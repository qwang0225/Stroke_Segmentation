import numpy as np
import matplotlib.pyplot as plt
import sys, os
import glob
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage
import cv2
import torch
from tqdm import tqdm
import random
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    RandAffined,
    ToTensord,
)
from monai.data import DataLoader, Dataset
from typing import List
from sklearn.model_selection import KFold
import requests
import zipfile


def download_data(url: str, destination_path: str):
    response = requests.get(url, stream=True)
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192): 
            file.write(chunk)
            
def unzip(zip_path: str, destination_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)


file_url = 'https://zenodo.org/record/7960856/files/ISLES-2022.zip'
zip_path = '.\data\ISLES-2022.zip'
data_dir = "./data/ISLES-2022"

# download datset from zenodo
if not os.path.exists('.\data'):
    os.mkdir('.\data')
if not os.path.exists(zip_path):
    download_data(file_url, zip_path)
if not os.path.exists(data_dir):
    unzip(zip_path, '.\data')

dwi_filepath_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*dwi.nii.gz")
seg_filepath_template = os.path.join(data_dir, "derivatives", "{filename}", "ses-0001", "*msk.nii.gz")
target_size=(128, 128)


train_transforms = Compose(
    [
        RandAffined(
        keys=['image', 'label'],
        mode=('bilinear', 'nearest'),
        prob=0.2, spatial_size=(128, 128),
        rotate_range=(0, 0, np.pi/15),
        scale_range=(0.1, 0.1, 0.1),
        translate_range=(0.1, 0.1, 0.1),
        padding_mode='border'),
        ToTensord(keys=["image", "label"], dtype=torch.float32),
    ]
)

val_transforms = Compose(
    [
        ToTensord(keys=["image", "label"], dtype=torch.float32),
    ]
)



def load_image(filepath: str) -> FileBasedImage:
    file = glob.glob(filepath)[0]
    return nib.load(file).get_fdata()

def load_images(filenames: List[str], filepath_template: str) -> List[FileBasedImage]:
    images = []
    for filename in filenames:
        filepath = filepath_template.format(filename=filename)
        file = load_image(filepath)
        images.append(file)
    return images

def get_stroke_filenames():
    all_files = sorted(glob.glob(os.path.join(data_dir, "*sub-strokecase*")))
    return [file.split(os.sep)[-1] for file in all_files]

def load_dwi_images(filenames: List[str]):
    filepath_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*dwi.nii.gz")
    dwi_3d_images = load_images(filenames, filepath_template)
    
    dwi_2d_images = []
    for dwi_3d_image in dwi_3d_images:

        for j in range(0, dwi_3d_image.shape[2]):
            dwi_2d_image = cv2.resize(dwi_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_LINEAR)
            dwi_2d_image_3channel = cv2.merge([dwi_2d_image, dwi_2d_image, dwi_2d_image])
            dwi_2d_image_3channel = np.moveaxis(dwi_2d_image_3channel, -1, 0)
            dwi_2d_images.append(dwi_2d_image_3channel)

    return dwi_2d_images

def create_dataset(images, labels):
    data = []
    for image, label in zip(images, labels):
        if np.sum(image) == 0 or np.sum(label) == 0:
            continue
        data.append({
            "image": image,
            "label": label
        })

    random.shuffle(data)
    # train = data[:int(0.8 * len(data))]
    # test = data[int(0.8 * len(data)):]
    return data    # train, test

def load_segmentation_images(filenames):
    filepath_template = os.path.join(data_dir, "derivatives", "{filename}", "ses-0001", "*msk.nii.gz")
    seg_3d_images = load_images(filenames, filepath_template)
    
    seg_2d_images = []
    for seg_3d_image in seg_3d_images:
        seg_3d_image[seg_3d_image > 0] = 1

        for j in range(0, seg_3d_image.shape[2]):
            seg_2d_image = cv2.resize(seg_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_NEAREST)
            seg_2d_image = np.expand_dims(seg_2d_image, axis=0)
            seg_2d_images.append(seg_2d_image)

    return seg_2d_images

def kfold_split(dataset):
    # Number of folds for cross-validation
    num_CV = 5
    kf = KFold(n_splits=num_CV, shuffle=True, random_state=42)

    # Lists to store training and validation sets for each fold
    folds = []

    # Loop through the folds
    for train_index, val_index in kf.split(dataset):

        train_data = [dataset[idx] for idx in train_index]
        val_data = [dataset[i] for i in val_index]

        # Append the training and validation sets to the lists
        folds.append({
            "train": Dataset(data=train_data, transform=train_transforms),
            "val": Dataset(data=val_data, transform=val_transforms)
        })

    return folds


filenames = get_stroke_filenames()
dwi_2d_images = load_dwi_images(filenames)
seg_2d_images = load_segmentation_images(filenames)
dataset = create_dataset(dwi_2d_images, seg_2d_images)
folds = kfold_split(dataset)


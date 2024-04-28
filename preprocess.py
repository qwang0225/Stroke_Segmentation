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
import requests
import zipfile
import pickle
import ants


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

# dwi_filepath_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*dwi.nii.gz")
# seg_filepath_template = os.path.join(data_dir, "derivatives", "{filename}", "ses-0001", "*msk.nii.gz")

dwi_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*dwi.nii.gz")
adc_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*adc.nii.gz")
registered_flair_template = os.path.join(data_dir, "{filename}", "ses-0001", "anat", "*registered.nii.gz")
mask_template = os.path.join(data_dir, "derivatives", "{filename}", "ses-0001", "*msk.nii.gz")
target_size=(128, 128)


# set random seed
set_determinism(seed=0)

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
    try:
        file = glob.glob(filepath)[0]
    except IndexError:
        print(f"File {filepath} not found")
        print(os.path.exists(file))
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
    """
    Load dwi images and resize them to target size
    """
    dwi_3d_images = load_images(filenames, dwi_template)
    
    dwi_2d_images = []
    for dwi_3d_image in dwi_3d_images:

        for j in range(0, dwi_3d_image.shape[2]):
            dwi_2d_image = cv2.resize(dwi_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_LINEAR)
            dwi_2d_image_3channel = cv2.merge([dwi_2d_image, dwi_2d_image, dwi_2d_image])
            dwi_2d_image_3channel = np.moveaxis(dwi_2d_image_3channel, -1, 0)
            dwi_2d_images.append(dwi_2d_image_3channel)

    return dwi_2d_images


def load_segmentation_images(filenames: List[str]):
    """
    Load segmentation images and resize them to target size
    """
    seg_3d_images = load_images(filenames, mask_template)
    
    seg_2d_images = []
    for seg_3d_image in seg_3d_images:
        seg_3d_image[seg_3d_image > 0] = 1

        for j in range(0, seg_3d_image.shape[2]):
            seg_2d_image = cv2.resize(seg_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_NEAREST)
            seg_2d_image = np.expand_dims(seg_2d_image, axis=0)
            seg_2d_images.append(seg_2d_image)

    return seg_2d_images

def register_image(fixed: str, moving: str, out: str):
    """
    Register two images using ANTs.
    fixed: path to the fixed image
    moving: path to the moving image
    out: path to save the registered image
    """
    fixed_image = ants.image_read(fixed)
    
    moving_image = ants.image_read(moving)
    transform = ants.registration(fixed_image, moving_image, 'Affine')

    reg3t = ants.apply_transforms(fixed_image, moving_image, transform['fwdtransforms'][0])
    ants.image_write(reg3t, out)

def register_flair_images(filenames: List[str]):
    """
    Register flair images to dwi images using ANTs."""
    flair_template = os.path.join(data_dir, "{filename}", "ses-0001", "anat", "*FLAIR.nii.gz")
    dwi_template = os.path.join(data_dir, "{filename}", "ses-0001", "dwi", "*dwi.nii.gz")
    
    for filename in filenames:
        dwi_img = glob.glob(dwi_template.format(filename=filename))[0]
        flair_img = glob.glob(flair_template.format(filename=filename))[0]
        registered_falir_img = flair_img.split(".nii.gz")[0] + "_registered.nii.gz"
        register_image(dwi_img, flair_img, registered_falir_img)
        
def load_full_images(filenames: List[str]):
    """
    load full 3-channel images with dwi, adc and flair
    """
    
    dwi_3d_images = load_images(filenames, dwi_template)
    adc_3d_images = load_images(filenames, adc_template)
    flair_3d_images = load_images(filenames, registered_flair_template)
    
    full_2d_images = []
    for dwi_3d_image, adc_3d_image, flair_3d_image in zip(dwi_3d_images, adc_3d_images, flair_3d_images):
        for j in range(0, dwi_3d_image.shape[2]):
            dwi_2d_image = cv2.resize(dwi_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_LINEAR)
            adc_2d_image = cv2.resize(adc_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_LINEAR)
            flair_2d_image = cv2.resize(flair_3d_image[:, :, j], dsize=target_size, interpolation=cv2.INTER_LINEAR)
            full_2d_image_3channel = cv2.merge([dwi_2d_image, adc_2d_image, flair_2d_image])
            full_2d_image_3channel = np.moveaxis(full_2d_image_3channel, -1, 0)
            full_2d_images.append(full_2d_image_3channel)

    return full_2d_images


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
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)):int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]
    return train, val, test

# def kfold_split(dataset):
#     # Number of folds for cross-validation
#     num_CV = 5
#     kf = KFold(n_splits=num_CV, shuffle=True, random_state=42)

#     # Lists to store training and validation sets for each fold
#     folds = []

#     # Loop through the folds
#     for train_index, val_index in kf.split(dataset):

#         train_data = [dataset[idx] for idx in train_index]
#         val_data = [dataset[i] for i in val_index]

#         # Append the training and validation sets to the lists
#         folds.append({
#             "train": Dataset(data=train_data, transform=train_transforms),
#             "val": Dataset(data=val_data, transform=val_transforms)
#         })

#     return folds

# set random seed
include_flair_adc = True 
random.seed(42) 
filenames = get_stroke_filenames()

if include_flair_adc:
    # register_flair_images(filenames)
    input_2d_images = load_full_images(filenames)
else:
    input_2d_images = load_dwi_images(filenames)

seg_2d_images = load_segmentation_images(filenames)
train, val, test = create_dataset(input_2d_images, seg_2d_images)
train_set = {'train': Dataset(data=train, transform=train_transforms), 'val': Dataset(data=val, transform=val_transforms)}
test_set = Dataset(data=test, transform=val_transforms)

with open('three_sequence_train_set.pkl', 'wb') as f:
    pickle.dump(train_set, f)
with open('three_sequence_test_set.pkl', 'wb') as f:
    pickle.dump(test_set, f)


# folds = kfold_split(train)
# print(2)

# with open('full_kfold_splits.pkl', 'wb') as f:
#     pickle.dump(folds, f)
    
# with open('full_test_set.pkl', 'wb') as f:
#     pickle.dump(test_set, f)
    

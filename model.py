import math
from einops import rearrange
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock, MLPBlock
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMerging

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=128, patch_size=4, embedding_size=768):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=embedding_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        num_patches = image_size//patch_size
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches**2, embedding_size))
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, image):
        # image: (B, C, H, W)
        split = self.conv(image)
        # split: (B, emb, H//patch_size, W//patch_size)

        linear_embedding = torch.flatten(split, -2, -1).permute((0,2,1))
        # linear_embedding: (B, num_patches, emb) where num_patches = (H//patch_size) * (W//patch_size)

        embedding = linear_embedding + self.positional_embedding

        return embedding

class PatchExpand(nn.Module):
    def __init__(self, scale, embedding_size=768, keep_embedding_size=False):
        super(PatchExpand, self).__init__()
        self.scale = scale
        self.embedding_size = embedding_size
        
        if keep_embedding_size:
            self.expand = nn.Linear(embedding_size, scale*scale*embedding_size, bias=False)
            self.norm = nn.LayerNorm(embedding_size)
        else:
            self.expand = nn.Linear(embedding_size, scale*embedding_size, bias=False)
            self.norm = nn.LayerNorm(embedding_size//scale)

    def forward(self, x):
        # x = (B, H*W, emb)
        x = self.expand(x)
        # x = (B, H*W, C)
        # if keep_embedding_size, C = scale*scale*embed, else C = scale*embed

        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))

        x = x.reshape(B, H, W, C)
        # x = (B, H, W, C)

        x = x.reshape(B, H, W, self.scale, self.scale, C//(self.scale**2))
        # x = (B, H, W, scale, scale, C//scale^2)

        x = x.permute((0, 1, 3, 2, 4, 5))
        # x = (B, H, scale, W, scale, C//scale^2)

        x = x.reshape(B, H*self.scale*W*self.scale, C//(self.scale**2))
        # x = (B, H*scale*W*scale, C//scale^2)

        return self.norm(x)
    
class Decoder(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Encoder-Decoder Attention
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.enc_dec_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)

        # MLP block
        self.norm_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input, skip):
        x = input
        x = self.self_attention(x, x, x, need_weights=False)[0]
        x = self.norm_1(input)
        x = self.dropout(x)
        x = x + input

        y = self.enc_dec_attn(x, skip, skip, need_weights=False)[0]
        y = self.norm_2(x)
        y = y + x

        z = self.mlp(y)
        z = self.norm_3(y)
        return z + y
    
class CustomEncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, num_encoders = 2):
        super(CustomEncoderBlock, self).__init__()
        self.encoders = nn.ModuleList()
        self.encoders.extend([
            EncoderBlock(num_heads, hidden_size, hidden_size*2, 0, 0)
            for _ in range(0, num_encoders)
        ])
        self.patch_merge = PatchMerging(hidden_size)

    def forward(self, x):
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))

        for encoder in self.encoders:
            x = encoder(x)

        x = x.reshape(B, H, W, C)
        x = self.patch_merge(x)
        return x.reshape(B, H*W//4, C*2)
    
class CustomDecoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, num_decoders = 2):
        super(CustomDecoderBlock, self).__init__()
        self.decoders = nn.ModuleList()
        self.decoders.extend([
            Decoder(num_heads, hidden_size, hidden_size*2, 0, 0)
            for _ in range(0, num_decoders)
        ])

        self.patch_expand = PatchExpand(2, 2*hidden_size)

    def forward(self, x, skip):
        x = self.patch_expand(x)
        for decoder in self.decoders:
            x = decoder(x, skip)
        return x

class TransformerAttentionUNet(nn.Module):
    """
    image_size: the height and width dimensions of the input image
    patch_size: how large we want each initial patch to 
    embedding_size: how large we want our initial patch embedding to be
    num_blocks: how many transformer+patch merging blocks we want
    num_heads: the number of heads in each transformer's multihead attention
    """
    def __init__(self,
            image_size: int = 128,
            patch_size: int = 4,
            embedding_size: int = 768,
            num_blocks: int = 3,
            num_heads: int = 4):
        super(TransformerAttentionUNet, self).__init__()
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks

        self.patch_embedding = PatchEmbedding(image_size, patch_size, embedding_size)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        hidden_size = embedding_size
        for i in range(0, num_blocks):
            self.encoder.append(CustomEncoderBlock(num_heads, hidden_size, 2))
            
            self.decoder.append(CustomDecoderBlock(num_heads, hidden_size, 2))

            hidden_size *= 2

        # the middle part of U-Net structure
        self.bottleneck = nn.Sequential(
            EncoderBlock(num_heads, hidden_size, 2*hidden_size, 0, 0),
            EncoderBlock(num_heads, hidden_size, 2*hidden_size, 0, 0),
        )

        # segmentation model head
        self.final_expand = PatchExpand(patch_size, embedding_size, keep_embedding_size=True)
        self.output = nn.Conv2d(embedding_size, 1, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embedding(x)
        #print("embed: ", x.shape)
        # x = (B, num_patches*num_patches, embed)

        skip = [] # holds the residual skip variables

        for encoder in self.encoder:
            skip.append(x)

            x = encoder(x)
            #print("enc: ", x.shape)

        x = self.bottleneck(x)
        #print("bottleneck: ", x.shape)
        # x = (B, (H//2^num_blocks) * (W//2^num_blocks), embed * 2^num_blocks)

        for decoder, s in reversed(list(zip(self.decoder, skip))):
            x = decoder(x, s)
            #print("Decode: ", x.shape)]
        
        #x = self.norm(x)
        x = self.final_expand(x)
        # x = (B, H*W, embed)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, self.embedding_size, H, W)

        x = self.output(x)
        #print(x.shape)
        # x = (B, H*W, 1)

        x = x.reshape(B, 1, H, W)
        return self.activation(x)
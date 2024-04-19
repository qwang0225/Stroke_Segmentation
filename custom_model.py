import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torchvision.models.vision_transformer import EncoderBlock

# wrapper for Layer/Batch Norm so we don't have to code the reshape every time
# Normalize seems to be standard for transformers, even though most uses vision uses BatchNorm2d
# Batch norm seems slightly more effective? I don't think it matters that much
class Normalize(nn.Module):
    def __init__(self, channels):
        super(Normalize, self).__init__()

        # self.norm = nn.LayerNorm(channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x = (B, C, H, W)

        #x = x.permute(0, 2, 3, 1)
        # x = (B, H, W, C)

        x = self.norm(x)
        #x = x.permute(0, 3, 1, 2)
        # x = (B, C, H, W)

        return x

# converts the (B, 3, 128, 128) image to a (B, C, H, W) embedding
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, out_channels: int, patch_size: int):
        super(PatchEmbedding, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = image_size//patch_size
        # std set to standard 0.02, see section 5 of this paper: https://aclanthology.org/D19-1083.pdf
        self.positional_embedding = nn.Parameter(torch.randn(1, out_channels, num_patches, num_patches).normal_(std=0.02))

        self.norm = Normalize(out_channels)

    def forward(self, image):
        # image = (B, 3, H, W)

        linear_embedding = self.conv(image)
        # linear_embedding = (B, C, H//patch_size, W//patch_size)

        embedding = linear_embedding + self.positional_embedding

        #return self.norm(linear_embedding)
        return self.norm(embedding)

# scales down the dimensions of the image by scale
class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super(PatchMerging, self).__init__()

        self.merge = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=scale,
            stride=scale,
            bias=False
        )

        self.norm = Normalize(out_channels)

    def forward(self, x):
        # x = (B, C, H, W)

        x = self.merge(x)
        # x = (B, C*scale, H//scale, W//scale)

        return self.norm(x)

# wrapper around MultiHeadAttention so we don't have to code the reshapes every time
class EncoderAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float):
        super(EncoderAttentionBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm = Normalize(channels)

    def forward(self, query, key, value):
        B, C, H, W = query.shape

        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        # (B, H, W, C)

        query = query.flatten(1, 2)
        key = key.flatten(1, 2)
        value = value.flatten(1, 2)
        # (B, H*W, C)

        output = self.attention(query, key, value)[0]

        output = output.unflatten(1, (H, W))
        # output = (B, H, W, C)

        output = output.permute(0, 3, 1, 2)
        # output = (B, C, H, W)

        return self.norm(output)

# inspired by Mix-FFN from SegFormer: https://arxiv.org/pdf/2105.15203.pdf
class MLPBlock(nn.Module):
    def __init__(self, channels, hidden_size):
        super(MLPBlock, self).__init__()

        self.dense = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

        self.depth_wise = nn.Conv2d(
            in_channels=channels,
            out_channels=hidden_size,
            kernel_size=3,
            groups=channels,
            padding=1,
        )

        self.nonlinear = nn.GELU()

        self.decrease = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=channels,
            kernel_size=1
        )

        self.norm = Normalize(channels)

    def forward(self, x):
        # x = (B, C, H, W)

        x = self.dense(x)
        # x = (B, C, H, W)

        x = self.depth_wise(x)
        # x = (B, C*scale, H, W), hidden_size = C*scale

        x = self.nonlinear(x)

        x = self.decrease(x)
        # x = (B, C, H, W)

        return self.norm(x)
    
# analogous to swin transformer block in the architecture diagram here: https://arxiv.org/pdf/2105.05537.pdf
class EncoderBlock(nn.Module):
    def __init__(self, channels: int, num_blocks: int, num_heads: int, mlp_hidden: int, dropout: float):
        super(EncoderBlock, self).__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self-attention": EncoderAttentionBlock(
                    channels=channels,
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "mlp": MLPBlock(
                    channels=channels, 
                    hidden_size=mlp_hidden
                )
            }) for _ in range(0, num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:

            # IMPORTANT NOTE: x += _ is an IN-PLACE operation, which can cause issues
            # make sure to use x = x + _
            x = x + block["self-attention"](x, x, x)
            x = x + block["mlp"](x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, channels: List[int], scale: List[int], num_blocks: List[int], num_heads: List[int], mlp_hidden: List[int], dropout: float):
        super(Encoder, self).__init__()

        # start with a embedding layer + transformers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "merge": PatchEmbedding(
                    image_size=128,
                    out_channels=channels[0],
                    patch_size=scale[0],
                ),
                "transformers": EncoderBlock(
                    channels=channels[0], 
                    num_blocks=num_blocks[0], 
                    num_heads=num_heads[0], 
                    mlp_hidden=mlp_hidden[0],
                    dropout=dropout
                )
            })
        ])
        
        for i in range(1, len(channels)):
            self.layers.append(nn.ModuleDict({
                "merge": PatchMerging(
                    in_channels=channels[i-1],
                    out_channels=channels[i],
                    scale=scale[i]
                ),
                "transformers": EncoderBlock(
                    channels=channels[i],
                    num_blocks=num_blocks[i],
                    num_heads=num_heads[i],
                    mlp_hidden=mlp_hidden[i],
                    dropout=dropout
                )
            }))
            # last transformers layer here is the bottleneck part of U-Net

    def forward(self, x):
        # x = (B, C, H, W)

        skip = []
        for layer in self.layers:
            x = layer["merge"](x)
            # x = (B, C', H', C')

            x = x + layer["transformers"](x)
            
            skip.append(x)

        return skip

class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super(Upsample, self).__init__()

        # self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

        # self.conv = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1
        # )
        
        self.convup = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
        

        self.norm = Normalize(out_channels)

    def forward(self, x):
        # x = (B, C, H, W)

        # x = self.up(x)
        # # x = (B, C, H*scale, W*scale)

        # x = self.conv(x)
        # # x = (B, C', H*scale, W*scale)
        
        x=self.convup(x)
        
        x = self.norm(x)

        return x

class Concat(nn.Module):
    def __init__(self, channels: int, num_skip: int):
        super(Concat, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channels*num_skip,
            out_channels=channels,
            kernel_size=1
        )
        self.nonlinear = nn.GELU()
        self.norm = Normalize(channels)

    def forward(self, skip):
        # skip is a list of tensors (B, C, H, W), with list length num_skip 

        x = torch.concat(skip, 1)
        # x = (B, C*num_skip, H, W)

        x = self.conv(x)
        # x = (B, C, H, W)

        x = self.nonlinear(x)
        x = self.norm(x)

        return x


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net, inspired by https://arxiv.org/pdf/1804.03999.pdf
    """
    def __init__(self, dim_g: int, dim_x: int, dim_int: int):
        """
        dim_x: number of channels in x, the input tensor from the skip connection
        dim_g: number of channels in g, the tensor from previous skip connection, which is resized to the same size as x
        dim_int: number of channels in the intermediate tensor
        g provides gating signal to control the flow of information from x to decoder
        """
        super().__init__()
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(dim_x, dim_int,
                                         kernel_size=1),
                                nn.BatchNorm2d(dim_int)
        )
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(dim_g, dim_int,
                                         kernel_size=1),
                                nn.BatchNorm2d(dim_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(dim_int, 1,
                                         kernel_size=1),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi) # (B, 1, H, W), gating signal
        out = x*psi
        
        return out 
    
class Decoder(nn.Module):
    def __init__(self, channels: List[int], hidden_size: int, attention: bool = False):
        """
        channels: list of channels for each layer of the encoder
        hidden_size: number of channels in the hidden layer of the decoder
        attention: whether to use attention in the decoder or not
        """
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(channels)-1, -1, -1):
            # print(channels[i], len(channels)-1-i)
            self.layers.append(nn.ModuleDict({
                "skip": nn.ModuleList([
                    nn.ModuleDict({
                        "upsample": Upsample(channels[j], channels[i], channels[j]//channels[i]),
                         "attention": AttentionGate(channels[i], channels[i], hidden_size) # added attention gate here
                    })
                    for j in range(len(channels)-1, i, -1)
                ]),                    
                "concat": Concat(channels[i], len(channels)-1-i) if len(channels)-i > 1 else None,
                #"self-attention": AttentionBlock(channels[i], num_heads[i], dropout),
                "upsample": Upsample(channels[i], hidden_size, 2 * channels[i]//channels[0]),
                "attention": AttentionGate(channels[i], channels[i], hidden_size) 
            }))

        self.concat = Concat(hidden_size, len(channels))
        self.attention = attention # attention flag 

    def forward(self, skip):
        # add in residuals to each layer to see if this helps training or not
        residuals = []

        # similar architecture as Segformer, in that all the transformer blocks with various receptive fields
        # would be concatenated together (gradients can reach all of them in same time despite different depths)
        layer_outputs = []

        # dense skip connections as inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10222451
        for i in range(0, len(skip)):
            layer = self.layers[i]

            upsampled_skip = []
            for j in range(0, len(residuals)):
                operations = layer["skip"][j]
               

                y = residuals[j]
                y = operations["upsample"](y)
                if self.attention:
                    y = operations["attention"](y, skip[i])  # dense attention  https://arxiv.org/pdf/2403.18180v1.pdf
                
                
                upsampled_skip.append(y)

            x = skip[i]
            residuals.append(x)
            # print("UPSAMPLED: ", len(upsampled_skip))
            if len(upsampled_skip) > 0:
                tmp = layer["concat"](upsampled_skip)
                # print(x.shape, tmp.shape)
                x = x + tmp

            #x = layer["self-attention"](x, x, x)
            x = layer["upsample"](x)
            # print("X: ", x.shape)
            layer_outputs.append(x)

        return self.concat(layer_outputs)

    
class CustomModel(nn.Module):
    def __init__(
        self,
        channels: List[int],
        scale: List[int],
        num_blocks: List[int],
        num_heads: List[int],
        mlp_hidden: List[int],
        dropout: float,
        decoder_hidden: int,
        attention: bool = False
    ):

        super().__init__()
        self.encoder = Encoder(
            channels=channels,
            scale=scale, 
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_hidden=mlp_hidden,
            dropout=dropout)
        
        self.decoder = Decoder( 
            channels=channels,
            hidden_size=decoder_hidden,
            attention=attention
        )
    
        self.final_upsample = nn.UpsamplingBilinear2d(128)        
        self.classifier = nn.Sequential(
            MLPBlock(decoder_hidden, decoder_hidden*4),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, 1, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        skip = self.encoder(x)
        skip.reverse()
        decoder_output = self.decoder(skip)
        
        upsampled_output = self.final_upsample(decoder_output)
        segmentation = self.classifier(upsampled_output)

        return self.activation(segmentation)
    
 
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from lora import LoRA_Linear, LoRA_Embedding, LoRA_Conv2d
# import loralib as lora
## Just apply LoRa in here for 6 Linear
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

        self.linear_1 = LoRA_Linear(n_embd, 4 * n_embd, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
        self.linear_2 = LoRA_Linear(4 * n_embd, 4 * n_embd, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        #self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_feature = LoRA_Conv2d(in_channels, out_channels, kernel_size=3, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)


        # self.linear_time = nn.Linear(n_time, out_channels)
        # self.linear_time = nn.Linear(n_time, out_channels)

        self.linear_time = LoRA_Linear(n_time, out_channels, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
        self.linear_time = LoRA_Linear(n_time, out_channels, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        #self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_merged = LoRA_Conv2d(out_channels, out_channels, kernel_size=3, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            #self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            self.residual_layer = LoRA_Conv2d(in_channels, out_channels, kernel_size=1, padding=0, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        # ERROR implement : should change time shape to : shape x out_channel x 1 x 1
        time = time.view(-1, time.shape[-1], 1, 1)  # Ensure it's (batch_size, out_channels, 1, 1)
        time = time.expand(-1, -1, feature.shape[2], feature.shape[3])  # Expand to match feature map size
        # merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = feature + time

        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        #self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_input = LoRA_Conv2d(channels, channels, kernel_size=1, padding=0, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)


        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        # self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        # self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.linear_geglu_1  = LoRA_Linear(channels, 4 * channels * 2, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
        self.linear_geglu_2  = LoRA_Linear(4 * channels, channels, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
        
        #self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_output = LoRA_Conv2d(channels, channels, kernel_size=1, padding=0, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        #self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv = LoRA_Conv2d(channels, channels, kernel_size=3, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            #SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(LoRA_Conv2d(4, 320, kernel_size=3, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)),
            
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            #SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(LoRA_Conv2d(320, 320, kernel_size=3, stride=2, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            #SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(LoRA_Conv2d(640, 640, kernel_size=3, stride=2, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            #SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(LoRA_Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = LoRA_Conv2d(in_channels, out_channels, kernel_size=3, padding=1, r=16, lora_alpha=1, lora_dropout=0.1, merge_weights=True)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)


        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)


        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        
        # (Batch, 4, Height / 8, Width / 8)
        return output
    


################################################################################

# class UNET(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoders = nn.ModuleList([
#             # Downscaling
#             SwitchSequential(nn.Conv2d(4, 64, kernel_size=3, padding=1)),  # Original: 320 channels might be too large for a smaller model
#             SwitchSequential(UNET_ResidualBlock(64, 64), UNET_AttentionBlock(4, 16)),
#             SwitchSequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
#             SwitchSequential(UNET_ResidualBlock(128, 128), UNET_AttentionBlock(4, 32)),
#             SwitchSequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
#             SwitchSequential(UNET_ResidualBlock(256, 256), UNET_AttentionBlock(4, 64)),
#         ])

#         self.bottleneck = SwitchSequential(
#             UNET_ResidualBlock(256, 256),
#             UNET_AttentionBlock(4, 64),
#             UNET_ResidualBlock(256, 256),
#         )

#         self.decoders = nn.ModuleList([
#             # Upscaling
#             SwitchSequential(UNET_ResidualBlock(512, 256)),
#             SwitchSequential(UNET_ResidualBlock(512, 256), Upsample(256)),
#             SwitchSequential(UNET_ResidualBlock(384, 128), UNET_AttentionBlock(4, 32)),
#             SwitchSequential(UNET_ResidualBlock(256, 128), Upsample(128)),
#             SwitchSequential(UNET_ResidualBlock(192, 64), UNET_AttentionBlock(4, 16)),
#         ])

#     def forward(self, x, context, time):
#         skip_connections = []
#         for layers in self.encoders:
#             x = layers(x, context, time)
#             skip_connections.append(x)

#         x = self.bottleneck(x, context, time)

#         for layers in self.decoders:
#             x = torch.cat((x, skip_connections.pop()), dim=1) 
#             x = layers(x, context, time)
        
#         return x

# class UNET_OutputLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.groupnorm = nn.GroupNorm(32, in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         x = self.groupnorm(x)
#         x = F.silu(x)
#         x = self.conv(x)
#         return x

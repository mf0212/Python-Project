from clip_model import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion_lora import Diffusion
import torch
from torch.nn import DataParallel
import os
from model_converter_2 import converter_model


def preload_models_from_standard_weights(checkpoint_dir, device):
    # Assuming 'results/' is your directory with checkpoint files
    best_model_list = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if len(best_model_list) == 0:
        print("No checkpoint files found in the directory.")
    best_model_file = best_model_list[0]
    # Find the checkpoint file with the highest epoch number
    best_model_dir = os.path.join(checkpoint_dir, best_model_file)
    tmp_state_dict = torch.load(best_model_dir, map_location='cpu')
    state_dict = converter_model(tmp_state_dict)
    

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['vae_encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['vae_decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion_model'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip_model'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
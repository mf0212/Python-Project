from contextlib import contextmanager
import torch
import os
import re
import glob
import shutil
from torchinfo import summary
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from clip_model import CLIP
import clip
from PIL import Image
import numpy as np
from model_converter_2 import converter_model

@contextmanager
def has_model_parameters_changed(model):
    initial_params = [p.clone().detach() for p in model.parameters()] # Dùng với with sẽ thực hiện câu lệnh này trước
    try: 
        yield # Sau đó thực hiện các dòng lệnh bên trong with
    finally:
        # Sau khi thực hiện các dòng lệnh trong with rồi thực hiện các dòng lệnh này
        for initial, new in zip(initial_params, model.parameters()):
            if not torch.equal(initial, new):
                print("Model parameters have changed.")
                return
        print("Model parameters have not changed.")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_gpu_memory_usage(device="cuda:0"):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

        # Report the current GPU memory usage by tensors in bytes for a given device.
        allocated = torch.cuda.memory_allocated()
        # Report the total GPU memory managed by the caching allocator in bytes for a given device.
        cached = torch.cuda.memory_reserved()
        print(f"GPU Memory - Allocated: {allocated / 1e9:.2f} GB, Cached: {cached / 1e9:.2f} GB")
    else:
        print("GPU is not available. GPU memory usage cannot be displayed.")


def save_gpu_memory_usage(filepath, epochs, t,loss, clip_score):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        with open(filepath, 'a') as f:
            f.write(f"GPU Memory - Allocated: {allocated / 1e9:.2f} GB, Cached: {cached / 1e9:.2f} GB, Time : {t} seconds , Loss : {loss}, CLIP Score : {clip_score}, Epoch: {epochs+1}\n")
    else:
        with open(filepath, 'a') as f:
            f.write("GPU is not available. GPU memory usage cannot be displayed.\n")

# # Example usage:
# save_gpu_memory_usage('/mnt/data/gpu_memory_usage.txt')


#run on gpu 
# def getDevice():
#   is_cuda = torch.cuda.is_available()
#   return "cuda" if is_cuda else "cpu"

#function to find latest epoch

#function to load the latest epoch file if it exists


def load_latest_checkpoint(path='./', device="cpu"):
    latest_checkpoint_path = os.path.join(path, 'latest_checkpoint.pt')
    if os.path.isfile(latest_checkpoint_path):
        print("Loading the latest checkpoint.")
        state_dict = torch.load(latest_checkpoint_path, map_location='cpu')
        new_state_dict = converter_model(state_dict)
        latest_epoch = new_state_dict['epoch']
        return latest_epoch, new_state_dict
    else:
        print("No checkpoint found, starting from beginning.")
        return 0, None



def copy_and_increment_gpu_memory_log(src_file, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find the highest index for existing 'gpu_memory_usage_i.txt' files
    existing_files = glob.glob(os.path.join(dest_dir, 'gpu_memory_usage_*.txt'))
    indices = [int(re.match(r'gpu_memory_usage_(\d+)\.txt', os.path.basename(f)).group(1)) for f in existing_files]
    next_index = max(indices) + 1 if indices else 1
    
    # Define the new file path
    new_file_name = f'gpu_memory_usage_{next_index}.txt'
    new_file_path = os.path.join(dest_dir, new_file_name)
    
    # Copy the contents from the source file to the new file
    shutil.copyfile(src_file, new_file_path)
    print(f"Copied GPU memory usage log to '{new_file_path}'")

def print_summary_model(name="clip", device="cuda"):
    if name == "clip":
        clip_model = CLIP().to(device)
        dummy_input = torch.randint(high=49408, size=(1, 77), dtype=torch.long, device=device)
        summary(clip_model, input_data=[dummy_input])
    elif name == "diffusion_model":
        # Assume 'Diffusion' is your model and it has been defined somewhere
        diffusion_model = Diffusion().to(device)

        # Create dummy inputs based on what you expect the model to need

        dummy_latent_input = torch.randn(1, 4, 16, 16, device=device)  # Adjust the shape according to your model's architecture

        # Dummy context input, assuming it's a sequence of embeddings (e.g., text embeddings)
        dummy_context_input = torch.randn(1, 77, 768, device=device)  # Adjust sequence length and embedding size as needed

        # Dummy time input, assuming it's a single vector
        dummy_time_input = torch.randn(1, 320, device=device)
        # Now summarize the model with these inputs
        summary(diffusion_model, input_data=[dummy_latent_input, dummy_context_input, dummy_time_input])
    elif name == "vae_encoder":
        vae_encoder = VAE_Encoder().to(device)

        dummy_noise_input = torch.randn(1, 4, 16, 16, device=device) 
        dummy_image_input = torch.randn(1, 3, 128, 128, device=device) 
        summary(vae_encoder, input_data=[dummy_image_input, dummy_noise_input])
    elif name == "vae_decoder":
        vae_decoder = VAE_Decoder().to(device)

        latent_dim = 16  # Example latent dimension
        dummy_latent_input = torch.randn(1, 256, latent_dim, latent_dim, device=device) 
        summary(vae_decoder, input_data=[dummy_latent_input])


def get_clip_scores(images, text_descriptions, device="cpu"):
    # images : bs x channel x height x width
    # text_des : a list , len(text_des) = bs
    # Load the pre-trained CLIP model along with the preprocessing function
    model, preprocess = clip.load('ViT-B/32', device='cpu')

    # Preprocess images - resize and normalize according to CLIP's requirements
    processed_images = []
    for img in images:
        img = img.detach().cpu().permute(1, 2, 0).numpy() * 255 # Convert c x h w -> h x w x c
        img = img.astype('uint8')
        pil_img = Image.fromarray(img)
        processed_img = preprocess(pil_img)
        processed_images.append(processed_img)

    images = torch.stack(processed_images)
    
    # Tokenize the text descriptions
    text_inputs = clip.tokenize(text_descriptions)

    # Move the inputs to GPU if available
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    images = images.to(device)
    text_inputs = text_inputs.to(device)
    model = model.to(device)
    
    # Generate embeddings for the images and texts
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity matrix
    if device != "cpu":
        # Calculate the cosine similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T)  # Keep this on GPU

        # Calculate the mean of the diagonal values of the similarity matrix
        mean_diagonal = torch.diag(similarity_matrix).mean().item()  # Keep the operation on GPU and convert to scalar at the end
    else:
        similarity_matrix = torch.matmul(image_features, text_features.T).cpu().numpy()
        mean_diagonal = np.diag(similarity_matrix).mean()
    
    return mean_diagonal

def metrics(fid_score, clip_score, scale_factor=1e-2):
    # Invert the FID score so that higher is better
    inverted_fid_score = 1 / (fid_score + 1e-6)  # Adding a small epsilon to avoid division by zero

    # Normalize the inverted FID score to match the scale of the CLIP scores
    normalized_fid_score = inverted_fid_score * scale_factor

    # Combine the normalized FID score and CLIP score
    combined = normalized_fid_score + clip_score

    return combined

def freeze_model(model, unfreeze_fraction=0.2):
    # Count total parameters and determine number of parameters to unfreeze
    total_params = sum(p.numel() for p in model.parameters())
    unfreeze_params = int(total_params * unfreeze_fraction)

    # Initialize counter and freeze parameters
    params_unfrozen = 0
    for param in reversed(list(model.parameters())):  # Reverse to freeze earlier layers first
        if params_unfrozen < unfreeze_params:
            param.requires_grad = True
            params_unfrozen += param.numel()
        else:
            param.requires_grad = False





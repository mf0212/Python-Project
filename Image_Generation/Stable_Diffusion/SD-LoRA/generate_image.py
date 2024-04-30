import torch
import os
import time
from PIL import Image
from torchvision import transforms
from diffusion import Diffusion
from clip_model import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from ddpm import DDPMSampler
from transformers import CLIPTokenizer
from pipeline import generate
import model_loader_2
# Assume you have the necessary imports and model definitions

device = "cuda:2" if torch.cuda.is_available() else "cpu"
# Load models
clip_model = CLIP().to(device)
vae_encoder = VAE_Encoder().to(device)
vae_decoder = VAE_Decoder().to(device)
diffusion_model = Diffusion().to(device)



# Define tokenizer and other components if needed
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  # Your tokenizer setup

# Inference settings
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Use an empty string for unconditional generation
strength = 0.75  # Adjust based on how much deviation from the input image is desired
n_inference_steps = 50
# Define the function to perform inference
def perform_inference():
    #input_image = Image.open('Generate_Image/generate_img.jpg')  # Optional: Load an image if you want to start with one
    #input_image_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
    input_image = None
    
    model_dir = "./best_model"
    models = model_loader_2.preload_models_from_standard_weights(model_dir, device)

    start = time.time()
    # Run generation
    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=n_inference_steps,
        models=models,
        seed=42,
        device=device,
        idle_device=device,
        tokenizer=tokenizer
    )
    end = time.time()
    output_dir = "Generate_Image"
    base_filename = "generated_image"
    file_extension = ".png"
    index = 1
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    output_file_path = os.path.join(output_dir, f"{base_filename}_{index}{file_extension}")
    # If the file exists, increment the index until it doesn't
    while os.path.exists(output_file_path):
        index += 1
        output_file_path = os.path.join(output_dir, f"{base_filename}_{index}{file_extension}")

    # Convert output to PIL Image for display or saving
    output_pil = Image.fromarray(output_image)
    output_pil.save(output_file_path)
    print(f"Image saved as {output_file_path}")
    print(f"Generating Time : {end - start} seconds")

perform_inference()

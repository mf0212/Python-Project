import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import os
import glob
import random
import time
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import MSELoss
from diffusion_lora import Diffusion
from clip_model import CLIP
from utilities import has_model_parameters_changed, print_gpu_memory_usage, save_gpu_memory_usage, \
load_latest_checkpoint, copy_and_increment_gpu_memory_log, \
print_summary_model, count_trainable_parameters, get_clip_scores, metrics, freeze_model
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from ddpm import DDPMSampler
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from torcheval.metrics.image.fid import FrechetInceptionDistance
import clip #pip install -U git+https://github.com/openai/CLIP.git
from torchvision.transforms import Resize, Normalize, ToTensor, Compose, Lambda


config = {
    "dataset_params": {
        "im_path": "data/CelebAMask-HQ",
        "im_channels": 3,
        "im_size": 256,
        "name": "celebhq"
    },
}


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


dataset_config = config['dataset_params']
latents_channel = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None, use_percentage=1):
        self.split = split
        if self.split != 'all':
          #self.split_filter = pd.read_pickle(f'/data/CelebAMask-HQ/{self.split}.pickle')
          self.split_filter = pickle.load(open(f'./data/CelebAMask-HQ/{self.split}.pickle', 'rb'))

        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = ['text'] #
        self.images, self.texts = self.load_images(im_path, use_percentage)

        # Whether to load images or to load latents
        # if use_latents and latent_path is not None:
        #     latent_maps = load_latents(latent_path)
        #     if len(latent_maps) >= len(self.images):
        #         self.use_latents = True
        #         self.latent_maps = latent_maps
        #         print('Found {} latents'.format(len(self.latent_maps)))
        #     else:
        #         print('Latents not found')

    def load_images(self, im_path, use_percentage=1):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []


        for fname in tqdm(fnames):
            im_name = os.path.split(fname)[1].split('.')[0]

            if self.split != 'all':
              if im_name not in self.split_filter:
                continue

            ims.append(fname)

            if 'text' in self.condition_types:
                captions_im = []
                with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
                    for line in f.readlines():
                        captions_im.append(line.strip())
                texts.append(captions_im)


        total_images = len(ims)
        use_count = int(total_images * use_percentage)

        ims = ims[:use_count]
        texts = texts[:use_count] if self.condition_types else None

        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"

        print('Found {} images'.format(len(ims)))
        print('Found {} captions'.format(len(texts)))

        return ims, texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
            

WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
BATCH_SIZE = 2 # Batch_size > 1 bị explode trên máy 96
USE_PERCENTAGE = 0.7
N_CLIP = 400  # Sau bao nhiêu vòng mới thực hiện tính toán clip_score trong 1 epoch
latents_shape = (BATCH_SIZE, 4, LATENTS_HEIGHT, LATENTS_WIDTH)



im_dataset = CelebDataset(split='train',
                          im_path=dataset_config['im_path'],
                          im_size=dataset_config['im_size'],
                          im_channels=dataset_config['im_channels'],
                          use_percentage=USE_PERCENTAGE)

data_loader = DataLoader(im_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)





# Define the transformation to preprocess the input images
# transform = transforms.Compose([
#     transforms.Resize((config['im_size'], config['im_size'])),#
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# Let's assume you have a dataset class for your data
# from your_dataset import YourDataset

# Initialize the dataset and data loader
# dataset = YourDataset(transform=transform)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize your model components
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = 'cpu'





# clip_model = CLIP()
# vae_encoder = VAE_Encoder()
# vae_decoder = VAE_Decoder()
# diffusion_model = Diffusion()

# clip_model = torch.nn.DataParallel(clip_model, device_ids=[0, 3])
# vae_encoder = torch.nn.DataParallel(vae_encoder, device_ids=[0, 3])
# vae_decoder = torch.nn.DataParallel(vae_decoder, device_ids=[0, 3])
# diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=[0, 3])

# clip_model.to(device)
# vae_encoder.to(device)
# vae_decoder.to(device)
# diffusion_model.to(device)

# Initialize your DDPMSampler
# ddpm_sampler = DDPMSampler(torch.Generator(device=device))

# clip_model.train()
# vae_encoder.train()
# vae_decoder.train()
# diffusion_model.train()

# Initialize optimizer
# params = list(clip_model.parameters()) + list(vae_encoder.parameters()) + list(vae_decoder.parameters()) + list(diffusion_model.parameters())
# use_mixed_precision = True
# scaler = GradScaler() if use_mixed_precision else None
# # optimizer = Adam(params, lr=2e-4)
# optimizer = AdamW(params, lr=2e-4, weight_decay=0.01) # sum 4 model parameters

# # Setup the learning rate scheduler
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# print_summary_model("clip", device=device)
# print_summary_model("diffusion_model", device=device)
# print_summary_model("vae_encoder",device=device)
# print_summary_model("vae_decoder",device=device)


# # Initialize loss function
# criterion = MSELoss()



def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# transform = Compose([
#     Resize((299, 299)),  # Resize images to match Inception input
#     Lambda(lambda x: x / 255.0),  # Ensure images are in [0, 1]
#     Lambda(lambda x: x.clamp(0, 1))
# ])
# def batch_transform(images):
#     # Batch transform images before sending them to the device
#     transformed = transform(images)  # Apply CPU-based transformations
#     return transformed.to(device).to(torch.float32)  # Convert to float32 and move to GPU


def train(num_epochs, ckpt_dir, model_dir, do_cfg=True, idle_device='cpu', load_checkpoint=True):

    best_score = -float('inf')  # Initialize with worst possible value
    # Score higher is better
    clip_model = CLIP().to(device)
    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)
    diffusion_model = Diffusion().to(device)   
    
    # Initialize the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # # Initialize metric
    # fid_metric = FrechetInceptionDistance()

    if load_checkpoint:
        start_epoch , state_dict = load_latest_checkpoint(ckpt_dir,device="cpu") # Load checkpoint bằng cpu, model converter
       
        vae_encoder.load_state_dict(state_dict['vae_encoder'])

        vae_decoder.load_state_dict(state_dict['vae_decoder'])

        diffusion_model.load_state_dict(state_dict['diffusion_model'])

        clip_model.load_state_dict(state_dict['clip_model'])
    else:
        start_epoch = 0

    # clip_model = torch.nn.DataParallel(clip_model, device_ids=[2, 3])
    # vae_encoder = torch.nn.DataParallel(vae_encoder, device_ids=[2, 3])
    # vae_decoder = torch.nn.DataParallel(vae_decoder, device_ids=[2, 3])
    # diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=[2, 3])
    


    # print("Diffusion model trainable parameter :", count_trainable_parameters(diffusion_model))
    # freeze_model(diffusion_model, unfreeze_fraction=0.1)
    # freeze_model(clip_model, unfreeze_fraction=0.1)
    # freeze_model(vae_decoder, unfreeze_fraction=0.1)
    # freeze_model(vae_encoder, unfreeze_fraction=0.1)


    #print("Diffusion model trainable parameter after freeze :", count_trainable_parameters(diffusion_model))

    clip_model.to(device)
    vae_encoder.to(device)
    vae_decoder.to(device)
    diffusion_model.to(device)

    

    clip_model.train()
    vae_decoder.train()
    vae_encoder.train()
    diffusion_model.train()
    ddpm_sampler = DDPMSampler(torch.Generator(device=device))
    total_param_training = count_trainable_parameters(clip_model) + count_trainable_parameters(vae_decoder) + count_trainable_parameters(vae_encoder) + count_trainable_parameters(diffusion_model)
    params = list(clip_model.parameters()) + list(vae_encoder.parameters()) + list(vae_decoder.parameters()) + list(diffusion_model.parameters())

    print("Diffusion parameter trainale : ", count_trainable_parameters(diffusion_model)) # 113M
    

    print("Total parameter trainale : ", total_param_training) # 332M

    use_mixed_precision = True
    scaler = GradScaler() if use_mixed_precision else None
    # optimizer = Adam(params, lr=2e-4)
    optimizer = AdamW(params, lr=2e-4, weight_decay=0.01) # sum 4 model parameters

    # Setup the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = MSELoss()
    if idle_device:
        to_idle = lambda x: x.to(idle_device)
    else:
        to_idle = lambda x: x

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        accumulation_steps = 2  # Accumulate gradients over 4 forward passes.
        #optimizer.zero_grad(set_to_none=True)
        start = time.time()
        epoch_clip_score = []
        for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, text_data = batch
            images = images.to(idle_device)
            batch_size = images.shape[0]
            # print('Before forward pass')
            # print_gpu_memory_usage()
            if do_cfg:
                # Convert into a list of length Seq_Len=77
                cond_tokens = tokenizer.batch_encode_plus(
                    text_data['text'], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=idle_device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                cond_context = clip_model(cond_tokens) #torch.Size([1, 77])
                # Convert into a list of length Seq_Len=77
                uncond_texts = [""] * batch_size
                uncond_tokens = tokenizer.batch_encode_plus(
                    uncond_texts, padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=idle_device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip_model(uncond_tokens)
                # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
                context = torch.cat([cond_context, uncond_context])
            else:
                # Convert into a list of length Seq_Len=77
                tokens = tokenizer.batch_encode_plus(
                    text_data['text'], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                tokens = torch.tensor(tokens, dtype=torch.long, device=idle_device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip_model(tokens)
                # context shape :
                # torch.Size([2, 77, 384]) with scale clip
                #  torch.Size([2, 77, 768]) orginal clip
            to_idle(clip_model)

            encoder_noise = torch.randn(latents_shape, dtype=torch.float32, device=idle_device)
            # Encode the input image
            latent = vae_encoder(images, encoder_noise)
            # Latent shape : torch.Size([bs, 4, 16, 16]) with img_size : 128

            # Simulate the diffusion process: add noise to the latent
            timesteps = torch.randint(0, ddpm_sampler.num_train_timesteps, (images.size(0),), device=idle_device)
            noisy_latent = ddpm_sampler.add_noise(latent.to(idle_device), timesteps)
            noisy_latent = noisy_latent.to(idle_device)
            context = context.to(idle_device)
            # Get time embeddings for the generated time steps
            time_embeddings = [get_time_embedding(timestep.item()).to(idle_device) for timestep in timesteps]
            time_embeddings = torch.stack(time_embeddings)
            time_embeddings = time_embeddings.squeeze(0)
            # Predict the noise using the diffusion model
            predicted_noise = diffusion_model(noisy_latent, context, time_embeddings)
            # Compute the loss as the difference between added and predicted noise
            diffusion_loss = criterion(predicted_noise, latent - noisy_latent)
            reconstructed_images = vae_decoder(latent)
            reconstruction_loss = nn.functional.mse_loss(reconstructed_images, images)
            total_loss = diffusion_loss + reconstruction_loss


            #Metric
            # For fid score
            # real_imgs = batch_transform(images).to(device)
            # fake_imgs = batch_transform(reconstructed_images).to(device)
            # Before updating the FID metric, convert images to float32
            # fid_metric.update(fake_imgs.float().cpu(), is_real=False)
            # fid_metric.update(real_imgs.float().cpu(), is_real=True)
            # print('After forward pass')
            # print_gpu_memory_usage()

            #for clip score
            if i % N_CLIP == 0 :
                #score_clip = 0
                #score_clip = get_clip_scores(reconstructed_images, text_data) # Use cpu calculate similarity matrix
                score_clip = get_clip_scores(reconstructed_images, text_data, device=idle_device)

                # Clip score need bs x channel x height x width 
                epoch_clip_score.append(score_clip)
                #print("CLIP score : ", score_clip)

            loss = total_loss / accumulation_steps  # Normalize our loss (if average 
            epoch_losses.append(loss.item())
            #with has_model_parameters_changed(diffusion_model): # Kiểm tra xem các model nào thực hiện cập nhập trọng số
            # Perform backpropagation and optimization
            scaler.scale(loss).backward()
            # print('After backward pass')
            # print_gpu_memory_usage()
            scaler.step(optimizer)
            # print('After step')
            # print_gpu_memory_usage()
            scaler.update()
            #optimizer.zero_grad(set_to_none=True) 
            # Backpropagation
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            if (i + 1) % accumulation_steps == 0:
                #optimizer.step()  # Perform the optimization step.
                optimizer.zero_grad()

        scheduler.step()
        # fid_score = fid_metric.compute()
        # Log the average loss for the epoch
        average_loss = np.mean(epoch_losses)
        average_clip_score = np.mean(epoch_clip_score)
        end = time.time()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss} - Time : {end-start} seconds - CLIP Score : {average_clip_score}")
        print_gpu_memory_usage(device=device)
        save_gpu_memory_usage(os.path.join(ckpt_dir, 'gpu_memory_usage.txt'), epochs=epoch, t=end-start, loss=average_loss, clip_score=average_clip_score)
        # Save model checkpoints``
 
        torch.save({
            'epoch': epoch+1,
            'clip_model': clip_model.state_dict(),
            'vae_encoder': vae_encoder.state_dict(),
            'vae_decoder': vae_decoder.state_dict(),
            'diffusion_model': diffusion_model.state_dict()
        }, os.path.join(ckpt_dir, 'latest_checkpoint.pt'))

        if average_clip_score > best_score:
            best_score = average_clip_score
            best_epoch = epoch + 1
            # Save the best model
            best_model_path = os.path.join(model_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch+1,
                'clip_model': clip_model.state_dict(),
                'vae_encoder': vae_encoder.state_dict(),
                'vae_decoder': vae_decoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict()
            }, best_model_path)
            print(f"New best model saved with Score: {average_clip_score} at epoch {best_epoch}")
    # Path to the original gpu_memory_usage.txt
    src_file_path = os.path.join(ckpt_dir, 'gpu_memory_usage.txt')
    # Directory where the incremented log files should be saved
    dest_dir_path = './gpu_result'
    # Call the function to copy and increment the log file
    copy_and_increment_gpu_memory_log(src_file_path, dest_dir_path)
    # Write gpu result to file 
    # Write to gpu_result folder with name gpu_memory_usage_i.txt 
    print('Training complete!')
    with open(src_file_path, 'w') as file:
        pass  # This will erase the content of the file


print("Device :",  device)

# # Now you can call the train function with the number of epochs and the directory where you want to save your models
train(num_epochs=10, ckpt_dir='./checkpoint/', model_dir="./best_model/",idle_device=device,load_checkpoint=False)

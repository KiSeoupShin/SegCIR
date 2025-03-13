import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from params import parse_args, parse_args_from_yaml, get_project_root
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import model.alpha_clip as alpha_clip

model, preprocess_val = alpha_clip.load("ViT-L/14", device='cpu', 
                                        alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
                                        lora_adapt=False, rank=-1)

preprocess_mask = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.26)
    ])

root_project = os.path.join(get_project_root(), 'data')
source_path = os.path.join(root_project, "imgnet", "imgnet_real_query_alpha.txt")

with open(source_path, 'r') as f:
    lines = f.readlines()

filenames = [line.strip() for line in lines]
images = [name.split(" ")[0] for name in filenames]
alphas = [name.split(" ")[1] for name in filenames]

img_path = os.path.join(root_project, str(images[0]))
image = preprocess_val(Image.open(img_path))

alpha_path = os.path.join(root_project, str(alphas[0]))
alpha = preprocess_mask(Image.open(alpha_path))

import pdb; pdb.set_trace()
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from data import get_data
from params import parse_args_from_yaml
from utils import convert_models_to_fp32

import model.alpha_clip as alpha_clip

import pandas as pd
import numpy as np
from tqdm import tqdm


def main(args):
    model, preprocess = alpha_clip.load("ViT-L/14", device='cpu', 
                                        alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
                                        lora_adapt=False, rank=-1)
    convert_models_to_fp32(model)
    model.cuda(args.gpu)

    data = get_data(args, (preprocess, preprocess))
    dataloader = data['train'].dataloader

    new_data = []

    for batch in tqdm(dataloader):
        images, texts, alphas, urls, nouns = batch
        images = images.cuda(args.gpu, non_blocking=True)
        alphas = alphas.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            image_features, _ = model.visual(images, alphas, return_attn=True)
        
        image_features = image_features.detach().cpu().numpy()

        for url, feature, noun, caption in zip(urls, image_features, nouns, texts):
            new_url = url.replace('images', 'image_reverse_features').replace('jpg', 'npy')
            if not os.path.exists(new_url):
                np.save(new_url, feature)
            new_data.append([new_url, noun, caption])
        
    new_csv = pd.DataFrame(new_data, columns=['url', 'noun', 'caption'])
    new_csv.to_csv('cc/GRIT_features_reverse_train_data.csv', index=False, sep='|')


if __name__ == "__main__":
    config_path = "./configs/train_alphaclip.yml"
    args = parse_args_from_yaml(config_path)
    main(args)

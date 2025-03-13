import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import model.alpha_clip as alpha_clip
from data import get_data
from params import parse_args_from_yaml

import numpy as np
import cv2
from third_party.open_clip.clip import tokenize
from tqdm import tqdm

def normalize_images(rgb_image, seg_image):
    rgb_min = rgb_image.min()
    rgb_max = rgb_image.max()
    normalized_rgb = ((rgb_image - rgb_min) * 255 / (rgb_max - rgb_min)).astype(np.uint8)

    unique_values = np.unique(seg_image)
    if len(unique_values) > 2:
        seg_min = seg_image.min()
        seg_max = seg_image.max()
        normalized_seg = (seg_image - seg_min) / (seg_max - seg_min)
    else:
        normalized_seg = (seg_image == seg_image.max()).astype(np.float32)
    
    return normalized_rgb, normalized_seg

def save_normalized_images(rgb_image, seg_image, rgb_output_path, seg_output_path):
    normalized_rgb, normalized_seg = normalize_images(rgb_image, seg_image)

    cv2.imwrite(rgb_output_path, cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR))

    seg_save = (normalized_seg * 255).astype(np.uint8)
    cv2.imwrite(seg_output_path, seg_save)


if __name__ == '__main__':
    config_path = "./configs/train_alphaclip.yml"
    args = parse_args_from_yaml(config_path)
    args.batch_size = 2

    model, preprocess = alpha_clip.load("ViT-L/14", device='cpu', 
                                        alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
                                        lora_adapt=False, rank=-1)

    data = get_data(args, (preprocess, preprocess))
    cnt = 0

    for i in tqdm(data["train"].dataloader):
        image, caption, alpha = i[0], i[1], i[2]
        # save_normalized_images(
        #     image.detach().cpu().numpy()[0].transpose((1, 2, 0)), 
        #     alpha.detach().cpu().numpy()[0].transpose((1, 2, 0)), 
        #     f"normalized_rgb_{idx}.png", 
        #     f"normalized_seg_{idx}.png"
        # )
        # print(f'caption_{idx}: {caption}')
        # if idx == 5:
        #     break
        try:
            tokenize(caption)
        except:
            cnt += 1
    
    print(cnt)
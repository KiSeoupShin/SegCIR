import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_llm import BLIP2
import model.alpha_clip as alpha_clip
from utils import convert_models_to_fp32, parse_args_from_yaml, get_project_root
from data import CIRR
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from PIL import Image

def main(args):
    ### Load Model
    blip_model = BLIP2()
    # model, preprocess = alpha_clip.load("ViT-L/14", device='cpu', 
    #                                         alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
    #                                         lora_adapt=False, rank=-1)

    # convert_models_to_fp32(model)
    # model.cuda(args.gpu)

    # cudnn.benchmark = True
    # cudnn.deterministic = False

    ### Setting DataLoader
    root_project = os.path.join(get_project_root(), 'data')
    dataset = CIRR(
        root=root_project,
        mode='caps',
        is_mask=True,
        )       
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    for batch in dataloader:
        ref_images, ref_alphas, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions = batch

        if args.gpu is not None:
            ref_images = ref_images.cuda(args.gpu, non_blocking=True)
            ref_alphas = ref_alphas.cuda(args.gpu, non_blocking=True)
        
        all_captions = []
        
        for i in range(ref_alphas.shape[1]):
            ref_alpha = ref_alphas[0, i].permute(1, 2, 0)
            ref_alpha = Image.fromarray(ref_alpha.cpu().numpy())
            captions = blip_model.create_caption(ref_alpha) 
            all_captions.append(captions)
        
        import pdb;pdb.set_trace()


if __name__ == "__main__":
    config_path = "./configs/llm_cirr.yml"
    args = parse_args_from_yaml(config_path)
    main(args)
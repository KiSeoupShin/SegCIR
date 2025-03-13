import os
import time
import logging
from time import gmtime, strftime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch import optim

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from params import parse_args, parse_args_from_yaml
from third_party.open_clip.scheduler import cosine_lr
from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, IM_TRANSFORMER, FiLMedIM2TEXT #IM_Transformer 추가
from trainer import train
from data import get_data
from params import parse_args, parse_args_from_yaml
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32
import model.alpha_clip as alpha_clip


# === TransformWrapper 추가 (pickle 문제 해결) ===
class TransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x)

    def __getstate__(self):
        # 빈 dict를 반환하여 pickle 시 내부 transform은 저장하지 않음.
        return {}

    def __setstate__(self, state):
        # 언피클 시 동일한 transform을 재생성.
        # (여기서는 alpha_clip.load를 이용해 transform을 새로 생성합니다.
        #  실제 환경에 맞게 필요한 인자들을 고정하여 호출해야 합니다.)
        _, transform = alpha_clip.load(
            "ViT-L/14", device='cpu', 
            alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
            lora_adapt=False, rank=-1
        )
        self.transform = transform
# ===================================================


def main_worker(gpu, ngpus_per_node, log_queue, args):
    try:
        args.gpu = gpu
        args.rank = int(os.environ.get("RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        setup_worker_logging(args.rank, log_queue, args.log_level)

        # 배치 사이즈 조정
        args.batch_size = args.batch_size // args.world_size

        if args.rank == 0:
            logging.info(f"Starting training with world_size: {args.world_size}")
            logging.info(f"Batch size per GPU: {args.batch_size}")

        # 프로세스 그룹 초기화
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank
        )

        torch.cuda.set_device(args.local_rank)

        # 모델 로드 및 설정
        model, preprocess = alpha_clip.load(
            "ViT-L/14", 
            device='cpu',
            alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth",
            lora_adapt=False, 
            rank=-1
        )
        
        preprocess_train = TransformWrapper(preprocess)
        preprocess_val = TransformWrapper(preprocess)

        img2text = FiLMedIM2TEXT(
            embed_dim=model.embed_dim,
            middle_dim=args.middle_dim,
            output_dim=model.token_embedding.weight.shape[1],
            n_layer=args.n_layer
        )

        if args.precision == "amp" or args.precision == "fp32":
            convert_models_to_fp32(model)

        model = model.cuda(args.local_rank)
        img2text = img2text.cuda(args.local_rank)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)

        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )
        
        img2text = torch.nn.parallel.DistributedDataParallel(
            img2text,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )

        # DataLoader 설정
        data = get_data(args, (preprocess_train, preprocess_val))
        
        # Optimizer 설정
        if args.train_data is not None:
            exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n : not exclude(n)
            named_parameters = list(img2text.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            
            total_steps = data["train"].dataloader.num_batches * args.epochs
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        else:
            optimizer = None
            scheduler = None

        # GradScaler 설정
        if args.precision == "amp":
            scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = None

        # Training loop
        for epoch in range(args.epochs):
            if args.rank == 0:
                logging.info(f'Start epoch {epoch}')
            
            if hasattr(data["train"].dataloader, 'sampler') and hasattr(data["train"].dataloader.sampler, 'set_epoch'):
                data["train"].dataloader.sampler.set_epoch(epoch)
            
            train(model, img2text, data, epoch, optimizer, scaler, scheduler, args)
            
            if args.rank == 0 and ((epoch + 1) % args.save_frequency == 0 or (epoch + 1) == args.epochs):
                save_dict = {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "state_dict_img2text": img2text.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                }
                torch.save(
                    save_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt")
                )

    except Exception as e:
        logging.error(f"Error in rank {args.rank}: {str(e)}")
        raise e
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main(args):
    # multiprocessing 방식을 'spawn'으로 설정
    mp.set_start_method('spawn', force=True)
    
    if args.name is None:
        args.name = (f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_workers={args.workers}")
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # Distributed 환경에서 gpu와 world_size 설정 (torchrun 등 사용 시)
    if args.distributed:
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            args.gpu = int(local_rank)
        else:
            args.gpu = 0
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    else:
        if args.gpu is None:
            args.gpu = 0

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    
    main_worker(args.gpu, None, log_queue, args)

if __name__ == "__main__":
    config_path = "./configs/train_alphaclip_multi.yml"
    args = parse_args_from_yaml(config_path)
    main(args)
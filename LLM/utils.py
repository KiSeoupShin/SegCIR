import yaml
import argparse
from pathlib import Path

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    args = argparse.Namespace(**config)

    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    
    return args

def get_project_root():
    return Path(__file__).parent.parent
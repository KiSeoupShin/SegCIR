import torch
torch.hub.set_dir("/home/work/gisub_conference/.cache/")

from lavis.models import load_model_and_preprocess

# model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_caption", "base_coco", device='cuda', is_eval=True)
model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device='cuda', is_eval=True)
import pdb; pdb.set_trace()


def rewrited_forward(self, x: torch.Tensor):
    global alpha
    if alpha is None: # better 
        print(f"[Warning] in {type(self)} forward: no alpha input when use alpha CLIP, alpha is expected!")
        alpha = torch.ones_like((x[:, [0], :, :])) * 1.9231
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x + self.conv1_alpha(alpha)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    return x

state_dict = torch.load('checkpoints/clip_l14_grit+mim_fultune_6xe.pth')
converted_dict = collections.OrderedDict()
for k, v in state_dict.items():
    # if "visual" in k:
    if 'in_proj.weight' in k:
        converted_dict[k.replace('in_proj.weight', 'in_proj_weight')] = v
    elif 'in_proj.bias' in k:
        converted_dict[k.replace('in_proj.bias', 'in_proj_bias')] = v
    else:
        converted_dict[k] = v

model.vision_model.conv1_alpha = torch.nn.Conv2d(in_channels=1,
                                                    out_channels=model.vision_model.conv1.out_channels, 
                                                    kernel_size=model.vision_model.conv1.kernel_size, 
                                                    stride=model.vision_model.conv1.stride, 
                                                    bias=False)
model.vision_model.forward = types.MethodType(rewrited_forward, model.vision_model)
model.vision_model.load_state_dict(converted_dict, strict=False)
model.vision_model = model.vision_model.half().cuda()
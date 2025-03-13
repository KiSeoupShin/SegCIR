import os
from torch.utils.data import Dataset
import json
from PIL import Image
from third_party.open_clip.clip import tokenize
import torch
import torchvision

class CIRR(Dataset):
    def __init__(self, transforms=None, transforms_mask=None, is_mask=False, mode='caps', 
    vis_mode=False, test=False, root='./data'):
        self.mode = mode
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
        ])
        self.transforms_mask = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(0.5, 0.26)
        ])
        self.is_mask = is_mask
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = os.path.join(root, 'CIRR')
        self.root_img = os.path.join(self.root, 'dev')
        if self.test:
            self.root_img = os.path.join(self.root, 'test1')
            if self.mode == 'caps':
                self.json = os.path.join(self.root , 'captions/cap.rc2.test1.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.test1.json')
        else:
            if self.mode == 'caps':
                self.json = os.path.join(self.root, 'captions/cap.rc2.val.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.val.json')
        data = json.load(open(self.json, "r"))                                
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []        
        if self.test:
            self.init_test(data)
        elif self.mode == 'caps':            
            self.init_val(data)                        
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]                    
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))    

    def init_test(self, data):
        self.pairids = []
        if self.mode == 'caps':
            for d in data:
                ref_path = d['reference']+ ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d['caption']) 
                self.pairids.append(d['pairid'])
                self.target_imgs.append('dummy')
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d['reference']+ ".png"
            tar_path = d['target_hard']+ ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d['caption'])            
    
    def return_testdata(self, idx):
        if self.mode == 'caps':
                ref_path = str(self.ref_imgs[idx])
                img_path = os.path.join(self.root_img, ref_path)
                ref_images = self.transforms(Image.open(img_path))
                target_cap = self.target_caps[idx]
                text_with_blank_raw = 'a photo of * , {}'.format(target_cap)    
                caption_only = tokenize(target_cap)[0]
                text_with_blank = tokenize(text_with_blank_raw)[0]                 
                return ref_images, text_with_blank, \
                    caption_only, str(self.ref_imgs[idx]), \
                        self.pairids[idx], text_with_blank_raw
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == 'caps' and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = 'a photo of * , {}'.format(target_cap)    
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0] 
            if self.is_mask:
                ref_alphas, cap_query = self.get_mask_and_caption(ref_images, img_path, target_cap)
                return ref_images, ref_alphas, ref_text_tokens, caption_only, \
                    str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                        target_cap  
            else:
                return ref_images, ref_text_tokens, caption_only, \
                    str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                        target_cap                       
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, img_path
    
    def apply_mask_to_image(self, image_tensor, mask_tensor):
        binary_mask = (mask_tensor[0, :, :] > 0).float()
        expanded_mask = binary_mask.expand_as(image_tensor)

        inverted_mask = 1 - expanded_mask
        
        dimming_factor = 0.0
        original_part = image_tensor * inverted_mask
        dimmed_part = image_tensor * dimming_factor * expanded_mask

        masked_image = original_part + dimmed_part

        masked_image = masked_image * 255
        masked_image = masked_image.clamp(0, 255).byte()
        return masked_image
    
    def get_mask_and_caption(self, ref_images, img_path, target_cap):
        file_name = os.path.basename(img_path).split('.')[0]
        print(os.path.dirname(img_path))
        mask_path = os.path.join(os.path.dirname(img_path).replace('dev', 'image_mask'), file_name)
        caption_path = os.path.join(os.path.dirname(img_path).replace('dev', 'image_captions'), file_name+'.txt')

        all_masked_image = []
        for mask_file in os.listdir(mask_path):
            mask = self.transforms_mask(Image.open(os.path.join(mask_path, mask_file)))
            masked_image = self.apply_mask_to_image(ref_images, mask)
            all_masked_image.append(masked_image)
        all_masked_image = torch.stack(all_masked_image)
        
        with open(caption_path, 'r') as f:
            all_captions = [cap[:-2] for cap in f.readlines()]
        
        all_cap_tokens = []
        for caption in all_captions:
            cap_tokens = tokenize('a photo of {} , {}'.format(caption, target_cap))[0]
            all_cap_tokens.append(cap_tokens)
        all_cap_tokens = torch.stack(all_cap_tokens)

        return all_masked_image, all_cap_tokens

    def __getitem__(self, idx):
        if self.test:                        
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)
    
    def __len__(self):
        return len(self.target_imgs)
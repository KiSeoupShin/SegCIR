import transformers
import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from PIL import Image
import re
import matplotlib.pyplot as plt
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

transformers.logging.set_verbosity_error()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("/home/work/gisub_conference/sam2")

from third_party.open_clip.clip import tokenize
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)
from transformers.image_utils import load_image
import torch

class SegmentImage():
    def __init__(self):
        self.device = 'cuda'

        self.checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_model = build_sam2(self.model_cfg, self.checkpoint)
        self.sam_model.to(self.device)
        self.sam_processor = SAM2ImagePredictor(self.sam_model)

        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(0.5, 0.26)
        ])
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), interpolation=BICUBIC),
            torchvision.transforms.CenterCrop((224, 224)),
            self._convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.dino_model, self.dino_processor = self.get_dino()
    
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")
    
    def get_dino(self):
        model_id = "IDEA-Research/grounding-dino-tiny"

        dino_processor = AutoProcessor.from_pretrained(model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        return dino_model, dino_processor

    def dino_process(self, images, texts):
        inputs = self.dino_processor(images=images, text=texts, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[images.size[::-1]]
        )

        return results[0]['boxes'].unsqueeze(0).cpu().numpy().tolist()
    
    def segment(self, img_path, noun):
        img_fullpath = os.path.join('data/cirr/dev', img_path+'.png')
        image = Image.open(img_fullpath).convert("RGB")
        
        input_boxes = self.dino_process(image, f'A {noun}.')
        if input_boxes[0] == []:
            mask = torch.ones([1] + list(np.array(image).shape[:2]))
        else:
            if len(input_boxes[0]) != 1:
                input_boxes = [[input_boxes[0][0]]]
            mask = self.get_masks(image, input_boxes)

        mask = self.transforms.transforms[0](mask)
        mask = self.transforms.transforms[1](mask)
        mask = np.array(mask)

        binary_maskes = (mask[0, :, :] != 0)
        alpha = self.mask_transform((binary_maskes * 255).astype(np.uint8))

        return alpha
    
    def get_masks(self, image, boxes):
        with torch.inference_mode():
            self.sam_processor.set_image(image)
            masks, scores, _ = self.sam_processor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
                )
        
        return torch.tensor(masks)

class LlamaSummarize():
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir='/home/work/gisub_conference/.cache/',
            use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir='/home/work/gisub_conference/.cache/',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    
    def summarize_with_token(self, caption):
        messages = [
            {"role": "system", "content": 
            '''
            Perform the following actions:

            1. Summarize the sentence so that the number of tokens does not exceed 77 while preserving its meaning.
            2. Ensure that <tok> is not removed.
            3. Assume that <tok> contains a noun referring to a specific object.
            4. Only return summerized sentence.
            '''
            },

            {"role": "user", "content": caption},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]['content']

    
    def summarize(self, caption):
        messages = [
            {"role": "system", "content": 
            '''
            Perform the following actions:

            1. Summarize the sentence so that the number of tokens does not exceed 70 while preserving its meaning.
            4. Only return summerized sentence.
            '''
            },

            {"role": "user", "content": caption},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]['content']
    

    def create_token(self, caption, noun, max_attempts=5):
        messages = [
            {"role": "system", "content": 
            '''
            Here are the rules you should follow:

            1. A sentence labeled as `caption` and a sentence or word labeled as `noun` will be provided as input.
            2. Identify the appropriate position in the `caption` where the `noun` can be inserted and replace it with `<tok>`.
            3. Instead of directly inserting the `noun`, replace it with `<tok>`.
            4. Only return created caption.

            Example Answer 1:
            input:
                caption: The Law Society awards Thomas Conway an honorary LLD for his exceptional advocacy skills, recognizing his contributions to Canada's legal profession.
                noun: Law Society Treasurer Paul Schabas (right)
            output:
                The Law Society awards Thomas Conway an honorary LLD for his exceptional advocacy skills, with contributions from <tok> recognized in Canada's legal profession.

            Example Answer 2:
            input:
                caption: She stands with her right leg crossed over her left, wearing a black drop waist dress, silvery grey tights, and black flats. Her hair is in a ponytail tied with a large black bow, and she has a finger to her lips.
                noun: Young girl
            output:
                <tok> stands with her right leg crossed over her left, wearing a black drop waist dress, silvery grey tights, and black flats. Her hair is in a ponytail tied with a large black bow, and she has a finger to her lips.

            Example Answer 3:
            input:
                caption: Vassiliou and Angeliki enjoy watermelon on a patio off their master bedroom, offering views of Mount Lycabettus and Athens, stretching towards the Acropolis in the distance.
                noun: the mountains
            output:
                Vassiliou and Angeliki enjoy watermelon on a patio off their master bedroom, offering views of <tok>, Mount Lycabettus, and Athens, stretching towards the Acropolis in the distance.
            '''
            },

            {"role": "user", "content": 
            f'''
            caption: {caption}
            noun: {noun}
            '''},
        ]
        
        attempts = 0
        while attempts < max_attempts:
            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
            )
            if '<tok>' in outputs[0]["generated_text"][-1]['content']:
                return outputs[0]["generated_text"][-1]['content']
            else:
                attempts += 1

        print(f"caption: {caption}")
        print(f"noun: {noun}")
        
        return outputs[0]["generated_text"][-1]['content']
    
    def extract_noun(self, caption, text):
        messages = [
            {"role": "system", "content": 
            '''
Given an image caption and accompanying text, output the most likely noun to focus on within the image, even if it's challenging to specify exactly. Do not provide any explanation for why the noun is selected.

- Analyze the given caption and text to identify the most likely related noun.
- The selected noun should attempt to clarify the image's theme, representing the key concept potentially.

# Steps

1. Carefully read the input image's caption and related text to understand the main theme.
2. Identify potential nouns within the text.
3. Select the noun that is most likely relevant, without providing any explanation.

# Output Format

- Output one noun that is most likely to focus on within the image.
- Write the noun separately as text.

# Examples

**Input:** 
- Image Caption: "A group of elephants in the wild."
- Text: "Elephants are social animals that live in groups. They are known for their intelligence and memory."

**Output:** 
- Elephants

**Input:**
- Image Caption: "A bustling marketplace with vendors."
- Text: "The marketplace is crowded with people buying fresh produce and handmade goods."

**Output:**
- Marketplace
            '''
            },

            {"role": "user", "content": f'''
            Image Caption: {caption}
            Text: {text}
            '''},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]['content']

    def summarize_noun_caption(self, noun, caption):
        messages = [
            {"role": "system", "content": 
            '''
## Task:
Modify the noun and caption so that the phrase **"a photo of 'noun' with 'caption'"** does not exceed **77 tokens** in length.

## Rules:
1. **Strict Output Format:**  
   - Your response must contain **only** the modified phrase in the format:  
     **"a photo of [Modified Noun] with '[Modified Caption]'"**  
   - No extra text, explanations, or formatting.

2. **Modification Criteria:**  
   - Shorten the **noun** and **caption** while preserving meaning.  
   - Ensure the phrase stays within the **77-token limit** when formatted.

3. **Validation:**  
   - Check that the total token count does not exceed **77 tokens**.

## Example:
### Input:  
Noun: "a majestic golden eagle soaring through the sky"  
Caption: "the breathtaking sight of nature's power and grace"  

### Output:  
**"a photo of golden eagle with 'nature’s power and grace'"**  
            '''
            },

            {"role": "user", "content": f'''
            noun: {noun}
            caption: {caption}
            '''},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]['content']


class PaliGemma2():
    def __init__(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_id = "google/paligemma2-10b-mix-224"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir='/home/work/gisub_conference/.cache/').eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir='/home/work/gisub_conference/.cache/')

    def extract_noun(self, img_path, caption):
        img_fullpath = os.path.join('data/cirr/dev', img_path+'.png')
        image = load_image(img_fullpath)
        model_inputs = self.processor(text="Create a caption for this image in English.", images=image, return_tensors="pt").to(torch.bfloat16).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded

# def process(summarizer, caption, max_attempts=5):
#     attempt = 0

#     while attempt < max_attempts:
#         try:
#             tmp = tokenize(caption)
#             return caption
#         except:
#             caption = summarizer.summarize(caption)
#             attempt += 1
    
#     return None

def process(summarizer, noun, caption, max_attempts=5):
    attempt = 0

    while attempt < max_attempts:
        try:
            tmp = tokenize('a photo of '+noun+' with '+caption)
            return noun, caption
        except:
            output_text = summarizer.summarize_noun_caption(noun, caption)
            noun_match = re.search(r'a photo of (.*?)\s*with', output_text)
            caption_match = re.search(r'with\s*[\'""]?(.*?)[\'""]?$', output_text)

            if noun_match is None or caption_match is None:
                attempt += 1
            else:
                noun = noun_match.group(1).strip()
                caption = caption_match.group(1).strip()
                attempt += 1
    
    return None, None
        
def create_caption():
    summarizer = LlamaSummarize()
    caption_data = pd.read_csv('./cc/GRIT_features_train_data_processed.csv', sep='|')
    new_caption_data = pd.DataFrame(columns=caption_data.columns)
    
    from tqdm import tqdm
    
    for idx, row in tqdm(caption_data.iterrows(), total=len(caption_data)):
        new_row = row.copy()
        caption = row['caption']

        new_noun, new_caption = process(summarizer, caption)
        if new_caption is None:
            import pdb; pdb.set_trace()
        
        new_row['caption'] = new_caption

        new_caption_data = pd.concat([new_caption_data, pd.DataFrame([new_row])], ignore_index=True)

    new_caption_data.to_csv('./cc/GRIT_features_train_data_caption_processed.csv', sep='|', index=False)

def test_length():
    summarizer = LlamaSummarize()
    caption_data = pd.read_csv('./cc/GRIT_features_train_data_caption_processed.csv', sep='|')
    new_caption_data = pd.read_csv('./cc/GRIT_features_train_data_caption_processed_tmp.csv', sep='|')

    exist_url = new_caption_data['url'].tolist()
    
    from tqdm import tqdm
    
    for idx, row in tqdm(caption_data.iterrows(), total=len(caption_data)):
        if row['url'] in exist_url:
            continue

        new_row = row.copy()
        caption = row['caption']
        noun = row['noun']

        new_noun, new_caption = process(summarizer, noun, caption)
        if new_caption is None:
            import pdb; pdb.set_trace()
        
        new_row['caption'] = new_caption
        new_row['noun'] = new_noun

        new_caption_data = pd.concat([new_caption_data, pd.DataFrame([new_row])], ignore_index=True)

    new_caption_data.to_csv('./cc/GRIT_features_train_data_caption_processed_v2.csv', sep='|', index=False)

def validate_single_noun_output(output: str) -> bool:
    # 단어를 추출
    words = re.findall(r'\w+', output)
    
    # 단어가 하나일 경우 통과
    if len(words) == 1:
        return True
    
    # 단어가 두 개일 경우 명사구(Noun Phrase)처럼 보이는지 확인
    if len(words) == 2:
        # 전치사, 관사, 대명사 등이 포함되면 문장으로 간주
        invalid_start_words = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were'}
        if words[0].lower() not in invalid_start_words:
            return True

    # 그 외의 경우 (문장처럼 보이거나 세 단어 이상일 경우) False
    return False

def process_noun(summarizer, caption, text, max_attempts=5):
    attempts = 0

    while attempts < max_attempts:
        noun = summarizer.extract_noun(caption, text)
        if not validate_single_noun_output(noun):
            attempts += 1
        else:
            return noun

    print('Cannot convert caption to noun.')
    return None

    # return summarizer.extract_noun(img_path, caption)

def noun_create():
    # summarizer = PaliGemma2()
    # summarizer = LlamaSummarize()
    segment_model = SegmentImage()

    # with open('data/cirr/captions/cap.rc2.val.json', 'r') as f:
    #     data = json.load(f)
        
    # with open('cirr_caption_path.txt', 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    #     data_dict = {}
    #     for line in lines:
    #         data_dict[line.split('|')[0]] = line.split('|')[1][:-2]
    
    # with open('cirr_noun_path.txt', 'w', encoding='utf-8') as file:
    #     file.write('img_path|caption|noun\n')

    with open('cirr_noun_path.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append([line.split('|')[0], line.split('|')[-1][:-2]])
    
    from tqdm import tqdm

    for row in tqdm(data[1:], total=len(data[1:])):
        img_path, noun = row[0], row[1]
        mask_path = os.path.join('data/cirr/mask/', img_path+'.png')
        if os.path.exists(mask_path):
            continue
        
        mask = segment_model.segment(img_path, noun)
        mask = mask.squeeze(0)
        if len(mask.shape) != 2 or mask.shape[0] != 224 or mask.shape[1] != 224:
            if len(mask.shape) == 3 and mask.shape[0] == 224 and mask.shape[1] == 224 and mask.shape[2] == 224:
                mask = mask[0, :, :]
            else:
                import pdb; pdb.set_trace()
        plt.imsave(mask_path, mask, cmap='gray')


if __name__ == "__main__":
    # noun_create()
    # create_caption()
    test_length()
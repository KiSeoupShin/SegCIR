from model_llm import BLIP2
import json
import os
from tqdm import tqdm
from PIL import Image

blip_model = BLIP2()
with open('/home/work/gisub_conference/2024_dna_conference/data/CIRR/captions/cap.rc2.val.json', 'r') as f:
    data = json.load(f)

img_path = '/home/work/gisub_conference/2024_dna_conference/data/CIRR/dev/'
save_path = '/home/work/gisub_conference/2024_dna_conference/data/CIRR/image_captions/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for row in tqdm(data):
    file_name = row['reference']
    ref_path = os.path.join(img_path, file_name+'.png')
    image = Image.open(ref_path)
    captions = blip_model.create_caption(image)
    
    with open(os.path.join(save_path, file_name+'.txt'), 'w') as f:
        for cap in captions:
            f.write(cap+'\n')
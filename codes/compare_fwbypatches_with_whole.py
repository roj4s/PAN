import argparse
import torch
from torch.nn import MSELoss
import options.options as option
from models import create_model
from PIL import Image
from numpy import asarray
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import time
import numpy as np
import matplotlib
import os

l1 = MSELoss()
toTensor = ToTensor()
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('images_folder', type=str, help='Images folder')
parser.add_argument('patch_size', type=int)
parser.add_argument('output_csv', type=str)

opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

args = vars(parser.parse_args())

model = create_model(opt)
model = model.netG
model.eval()

_dir = args['images_folder']
output_addr = args['output_csv']

columns = ('image_addr', 'image_width', 'image_height', 'patch_width',
           'patch_height', "scale", 'l1',
           'time_whole_image', 'time_patches')
cols_str = ','.join(columns)

with open(output_addr, 'wt') as f:
    f.write(f"{cols_str}\n")

for img_name in os.listdir(_dir):
    img_addr = os.path.join(_dir, img_name)
    print(f"Image addr: {img_addr}")

    image = Image.open(img_addr)
    it = toTensor(image).unsqueeze(0)

    scale = opt['scale']
    image_width = it.shape[3]
    image_height = it.shape[2]

    print(f"Processing {image_width}x{image_height} image")

    patch_size = args['patch_size']
    n_patches = int(image_width/patch_size) + int(image_height/patch_size)
    print(f"\tProcessing {n_patches} patches of {patch_size}x{patch_size}")
    output = np.zeros((1, 3, image_height * scale, image_width * scale))
    print("\tOutput Placeholder shape: ", output.shape)
    tp = time.time()
    for i in range(int(image_width/patch_size)):
        for j in range(int(image_height/patch_size)):
            y = i * patch_size
            x = j * patch_size

            ysr = y*scale
            xsr = x*scale

            w = patch_size
            h = patch_size
            wsr = patch_size*scale
            hsr = patch_size*scale

            if x + h >= image_height or x + 2*h >= image_height:
                h = image_height - x
                hsr = image_height * scale - xsr

            if y + w >= image_width or y + 2*w >= image_width:
                w = image_width - y
                wsr = image_width * scale - ysr

            p = it[:, :, x:x+h, y:y+w]
            o = model(p)
            oo = o.cpu().detach().numpy()
            del o
            output[0, :, xsr:xsr+hsr, ysr:ysr+wsr] = oo[0]

    output_by_patches = torch.from_numpy(output)
    save_image(output_by_patches, os.path.join('outputs', img_name))
    tp = time.time() - tp
    print(f"\tBy patches time: {tp:.2f} secs")
    t = time.time()
    output_whole_image = model(it)
    t = time.time() - t
    print(f"\tWhole image time: {t:.2f} secs")
    save_image(output_by_patches, os.path.join('outputs_wi', img_name))
    e = l1(output_by_patches, output_whole_image)
    print(f"\t L1: {e.item()}")

    del it
    del output_by_patches
    del output_whole_image

    columns = ('image_addr', 'image_width', 'image_height', 'patch_width',
           'patch_height', "scale", 'l1',
           'time_whole_image', 'time_patches')
    d = (img_addr, str(image_width), str(image_height), str(w), str(h),
         str(scale), str(e.item()), str(t), str(tp))
    ds = ",".join(d)
    with open(output_addr, "at") as f:
        f.write(f"{ds}\n")

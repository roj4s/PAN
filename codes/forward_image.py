import argparse
import torch
import options.options as option
from models import create_model
from PIL import Image
from numpy import asarray
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import time
import numpy as np

toTensor = ToTensor()
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('image', type=str, help='Path to input image.')
parser.add_argument('output_addr', type=str, help='Path to '\
                    'output image.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

args = vars(parser.parse_args())

model = create_model(opt)
model = model.netG
model.eval()

print(f"Image addr: {args['image']}")

image = Image.open(args['image'])
t = toTensor(image).unsqueeze(0)
print("Input shape:", t.shape)

ti = time.time()
output = model(t)
ti = time.time() - ti
print(f"Inference time: {ti:.2f} secs.")
print("Output shape:", output.shape)

save_image(output, args['output_addr'])

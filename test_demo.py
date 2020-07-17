import os
import glob
import time
import torch
import cv2
import numpy as np
import PANet_arch as PANet_arch
import util as util

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    ## test dataset
    test_d = sorted(glob.glob('/mnt/hyzhao/Documents/datasets/DIV2K_test/*.png'))

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## some functions
    def readimg(path):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = im.astype(np.float32) / 255.
        im = im[:, :, [2, 1, 0]]
        return im

    def img2tensor(img):
        imgt = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None, ...]
        return imgt

    ## load model
    model = PANet_arch.PANet(in_nc=3, out_nc=3, nf=40, nb=16)
    model_weight = torch.load('./pretrained_model/PANetx4_DF2K.pth')
    model.load_state_dict(model_weight, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))

    ## runnning
    print('-----------------Start Running-----------------')
    psnrs = []
    times = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(len(test_d)):
        im = readimg(test_d[i])
        img_LR = img2tensor(im)
        img_LR = img_LR.to(device)
        
        start.record()
        img_SR = model(img_LR)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
        sr_img = util.tensor2img(img_SR.detach())

        ### save image
        save_img_path = './test_results/%s'%(test_d[i].split('/')[-1])
        util.save_img(sr_img, save_img_path)
        
        print('Image: %03d'%(i+1))
    print('Paramters: %d, Mean Time: %.10f'%(number_parameters, np.mean(times)/1000.))
    
if __name__ == '__main__':

    main()
import argparse
import torch
import options.options as option
from models import create_model
from torchsummaryX import summary

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


model = create_model(opt)
model = model.netG
model.eval()

summary(model, torch.zeros((1, 3, 32, 32)))

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./mobile_model.pt")

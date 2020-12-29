import argparse
import torch
import options.options as option
from models import create_model
from torchsummaryX import summary
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


model_to_transfer = create_model(opt)
model_to_transfer = model_to_transfer.netG
model_to_transfer.eval()
input_tensor = torch.rand(1, 3, 100, 100)

# define input tensor
input_var = Variable(torch.FloatTensor(input_tensor))
torch.onnx.export(model_to_transfer, input_tensor, "model_100.onnx", )

import torch
import numpy as np

from approx_model import ApproxNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0001
batch_size = 128
# NUM_HIDDEN_LIST = [8, 64, 8]
NUM_HIDDEN_LIST = [128, 64]
log_dir = "/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/model/weights/l1/"
print(f'log_dir: {log_dir}')
print("using device: ", DEVICE)
input_dim = 1502
# input_dim = 502
approx_net = ApproxNet(learning_rate, log_dir, input_dim, NUM_HIDDEN_LIST)
approx_net.load("pruned")


def save_to_txt(filename, array):
    with open(filename, 'w') as f:
        for val in array.flatten():
            f.write(f"{val}\n")


for name, param in approx_net._network.named_parameters():
    filename = log_dir + name.split(".")[1] + "_" + name.split(".")[2] + "_pruned.txt"
    save_to_txt(filename, param.data.numpy())

#=============================================================================
# def quantize_model(model, num_bits=4):
#     for param in model.parameters():
#         param.data = torch.round(param.data * (2 ** (num_bits - 1))) / (2 ** (num_bits - 1))
# quantize_model(approx_net._network)
#
# for name, param in approx_net._network.named_parameters():
#     filename = log_dir + "quant/" + name.split(".")[1] + "_" + name.split(".")[2] + ".txt"
#     save_to_txt(filename, param.data.numpy())

################################################################################
# from torch.nn.utils import prune
#
#
# class ThresholdPruning(prune.BasePruningMethod):
#     PRUNING_TYPE = "unstructured"
#
#     def __init__(self, threshold):
#         super().__init__()
#         self.threshold = threshold
#
#     def compute_mask(self, tensor, default_mask):
#         return torch.abs(tensor) > self.threshold
#
# parameters_to_prune = ((approx_net._network.net.linear0, "weight"), (approx_net._network.net.linear0, "bias"))
# prune.global_unstructured(
#     parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-4
# )
# prune.remove(approx_net._network.net.linear0, name="weight")
# prune.remove(approx_net._network.net.linear0, name="bias")
#
# for name, param in approx_net._network.named_parameters():
#     filename = log_dir + name.split(".")[1] + "_" + name.split(".")[2] + "_pruned.txt"
#     save_to_txt(filename, param.data.numpy())
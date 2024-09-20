import torch
import torch.nn as nn
import numpy as np
from tpuop.tpuop import TpuOp

# fp32
input_tensor = torch.randn(1, 3136, 96).float()
weight_tensor = torch.randn(96).float()
bias_tensor = torch.randn(96).float()
output_tensor = torch.zeros(1, 3136, 96).float()
axis = 2
eps = 1e-5
print("fp32 test")
print(input_tensor)

# cmodel op
tpuop = TpuOp(device=0)
tpuop.tpuop_layernorm(
    input_tensor,
    weight_tensor,
    bias_tensor,
    output_tensor,
    axis,
    eps
)
print("cmodel LayerNorm output:")
print(output_tensor)

# torch op
ln = nn.LayerNorm([96], eps=eps)
ln.weight.data = weight_tensor
ln.bias.data = bias_tensor
torch_ln_output = ln(input_tensor)
print("torch LayerNorm output:")
print(torch_ln_output)

# fp16
input_tensor = torch.randn(1, 3136, 96).to(torch.float16)
weight_tensor = torch.randn(96).to(torch.float16)
bias_tensor = torch.randn(96).to(torch.float16)
output_tensor = torch.zeros(1, 3136, 96).to(torch.float16)
axis = 2
eps = 1e-5
print("fp16 test")
print(input_tensor)
print(input_tensor.numpy().view(np.uint16).flatten()[:32])
print(weight_tensor.numpy().view(np.uint16).flatten()[:32])
print(bias_tensor.numpy().view(np.uint16).flatten()[:32])

# cmodel op
tpuop = TpuOp(device=0)
tpuop.tpuop_layernorm(
    input_tensor,
    weight_tensor,
    bias_tensor,
    output_tensor,
    axis,
    eps
)
print("cmodel LayerNorm output:")
print(output_tensor)

# torch op
ln = nn.LayerNorm([96], eps=eps)
ln.weight.data = weight_tensor.float()
ln.bias.data = bias_tensor.float()
torch_ln_output = ln(input_tensor.float())
print("torch LayerNorm output:")
print(torch_ln_output)

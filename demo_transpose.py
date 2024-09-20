import torch
import torch.nn as nn
from tpuop.tpuop import TpuOp

import numpy as np
import argparse
from itertools import permutations

# for i in range(1):
#     input_tensor = torch.arange(64).reshape(1, 4, 4, 4).float()
#     output_tensor = torch.zeros(1, 4, 4, 4).float()
#     buffer_tensor = torch.zeros(256).float()
#     order = [0, 2, 1, 3]
#     order = [0, 3, 2, 1]
#     print("input:")
#     print(input_tensor)

#     # cmodel op
#     tpuop = TpuOp(device=0)
#     tpuop.tpuop_transpose(
#         input_tensor,
#         output_tensor,
#         buffer_tensor,
#         order
#     )
#     print("cmodel Transpose output:")
#     print(output_tensor)

# input_tensor = torch.arange(64).reshape(4, 4, 4).float()
# output_tensor = torch.zeros(4, 4, 4).float()
# order = [1, 2, 0]
# print("input:")
# print(input_tensor)

# # cmodel op
# tpuop = TpuOp(device=0)
# tpuop.tpuop_transpose(
#     input_tensor,
#     output_tensor,
#     order
# )
# print("cmodel Transpose output:")
# print(output_tensor)

def print_example_of_transpose(nD: int, minD: int, maxD: int):
    print(f'# Example for {nD}D-tensor with float type element')
    Dsize = np.random.randint(minD, maxD, size=nD)
    a = np.arange(np.multiply.reduce(Dsize), dtype=np.float32).reshape(Dsize)
    # a_torch = torch.tensor(a)
    print(f'Source:\n{a}')
    i = 0
    for T in permutations(np.arange(nD, dtype=np.int32)):
        print(f'Permutation {i}:')
        i += 1
        print(f'Result transpose {T}:')
        b = np.transpose(a, T)
        print(b)
        # a_torch = torch.from_numpy(a)
        a_torch = torch.arange(np.multiply.reduce(Dsize), dtype=torch.float32).reshape(tuple(Dsize))
        output_tensor = torch.zeros(b.shape, dtype=torch.float32)
        buffer_tensor = torch.zeros(np.multiply.reduce(Dsize), dtype=torch.float32)
        tpuop = TpuOp(device=0)
        tpuop.tpuop_transpose(
            a_torch,
            output_tensor,
            buffer_tensor,
            T
        )
        print("cmodel Transpose output:")
        print(output_tensor)
                

def main(args):
    if args.test1: print_example_of_transpose(nD=3, minD=4, maxD=6)
    if args.test2: print_example_of_transpose(nD=4, minD=4, maxD=6)
    if args.test3: print_example_of_transpose(nD=5, minD=4, maxD=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Transpose example')
    parser.add_argument('--test1', action='store_true')
    parser.add_argument('--test2', action='store_true')
    parser.add_argument('--test3', action='store_true')
    args = parser.parse_args()
    main(args)
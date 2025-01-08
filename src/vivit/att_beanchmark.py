####################################################################################
#
# From https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
#
####################################################################################

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

class MultiHeadAttBench(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()

        # assert emb_dim % n_heads == 0 torch implementation will verify this already
        self.head_size = emb_dim // n_heads
        self.n_heads = n_heads

        self.multihead_att = nn.MultiheadAttention(emb_dim, n_heads, 0.1, batch_first=True)

        self.avg_att_weights = None

    def forward(self, x:torch.Tensor):
        # B, S, E = x.shape
        with sdpa_kernel(SDPBackend.MATH):
            math_time=benchmark_torch_function_in_microseconds(self.multihead_att, x, x, x)
            print(f"The math implementation runs in {math_time:.3f} microseconds")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            try:
                flash_time=benchmark_torch_function_in_microseconds(self.multihead_att, x, x, x)
                print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
            except RuntimeError:
                print("FlashAttention is not supported. See warnings for reasons.")

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            try:
                efficient_time=benchmark_torch_function_in_microseconds(self.multihead_att, x, x, x)
                print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
            except RuntimeError:
                print("EfficientAttention is not supported. See warnings for reasons.")

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            try:
                efficient_time=benchmark_torch_function_in_microseconds(self.multihead_att, x, x, x)
                print(f"The CuDNN implementation runs in {efficient_time:.3f} microseconds")
            except RuntimeError:
                print("CuDNN is not supported. See warnings for reasons.")
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            # x is (B, S, E) & att is (B, S, E)
            x, self.avg_att_weights = self.multihead_att(query=x, key=x, value=x)

        return x
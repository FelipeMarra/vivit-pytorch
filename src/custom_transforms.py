import torch
import torch.nn as nn
from torchvision.transforms import v2

class ResizeSmallest(nn.Module):
    """
        Resizes frames so that `min(height, width)` is equal to `min_resize`.
        `max(height, width)` will be resized to `original_ratio * min_resize` where `original_ratio` is `max(height, width)/min(height, width)`.

        This function will do nothing if the `min(height, width)` is already equal to
        `min_resize`.

        Args:
            frames: A tensor of dimension [..., C, H, W].
            min_resize: Minimum size of the final image dimensions.

        Returns:
            A tensor of shape [..., C, H, W] of same type as input, where `min(output_h, output_w)` is `min_resize`.

        Motivation:
            The original ViViT [1] uses this with implementation from Deep Mind Video Readers [2]
            [1] https://github.com/google-research/scenic/blob/97d6ac5b65040621f266b0da3bf05066baa664f3/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py#L59
            [2] https://github.com/google-deepmind/dmvr/blob/77ccedaa084d29239eaeafddb0b2e83843b613a1/dmvr/processors.py#L374 
    """
    def __init__(self, min_resize:int):
        super().__init__()
        self.min_resize = min_resize

    def forward(self, frames:torch.Tensor):
        H = frames.shape[-2]
        W = frames.shape[-1]

        out_h = max(self.min_resize, (H * self.min_resize) // W)
        out_w = max(self.min_resize, (W * self.min_resize) // H)

        return v2.Resize((out_h, out_w))(frames)

class ZeroCenterNorm(nn.Module):
    """
        Normalize frames to the interval [-1, 1] with mean 0 and std 1 

        Args:
            frames: A tensor of dimension [..., C, H, W].

        Returns:
            A tensor of shape [..., C, H, W] of same type as input, where frames values are in interval [-1, 1].

        Motivation:
            The original ViViT [1] stes this to true, with implementation from Deep Mind Video Readers [2]
            [1] https://github.com/google-research/scenic/blob/97d6ac5b65040621f266b0da3bf05066baa664f3/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py#L62
            [2] https://github.com/google-research/scenic/blob/97d6ac5b65040621f266b0da3bf05066baa664f3/scenic/projects/vivit/data/video_tfrecord_dataset.py#L308
    """

    def forward(self, frames:torch.Tensor):
        mean = frames.mean(dim=(-2, -1), keepdim=True)
        std = frames.std(dim=(-2, -1), keepdim=True)

        frames = (frames - mean) / std

        return frames
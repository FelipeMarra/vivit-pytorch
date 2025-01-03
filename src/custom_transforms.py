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

    def forward(self, frames):
        H = frames.shape[-2]
        W = frames.shape[-1]
        print(f"ResizeSmallest frames {frames.shape}, {H}, {W}")

        out_h = max(self.min_resize, (H * self.min_resize) // W)
        out_w = max(self.min_resize, (W * self.min_resize) // H)

        return v2.Resize((out_h, out_w))(frames)
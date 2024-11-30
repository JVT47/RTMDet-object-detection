import torch
import torch.nn as nn
import torchvision.transforms.v2 as T_v2


class RTMDetPreprocessor(nn.Module):
    """
    Preprocessor that takes an input image tensor, resizes and normalizes
    it for inference. 
    """
    def __init__(self, dest_size: tuple[int, int], resize: bool, pad_color: list[float] = [114, 114, 114],
                 mean: list[float] = [103.53, 116.28, 123.675], std: list[float] = [57.375, 57.12, 58.395],
                 *args, **kwargs) -> None:
        """
        dest_size: the (H, W) output dimensions of the preprocessor
        resize: if true the input tensor is resized with aspect ratio to dest_size else the input is simply padded
        pad_color: the color used for potential padded regions
        mean: list of means used to normalize the input image. Default values from mmdet RMTDet implementation
        std: list of stds used to normalize the input image. Default values from mmdet RTMDet implementation
        """
        super().__init__(*args, **kwargs)

        assert dest_size[0] % 32 == 0 and dest_size[1] % 32 == 0, "RTMDet requires that input dimensions are divisible by 32"
        
        self.dest_size = dest_size
        self.resize = resize
        self.pad_color = pad_color
        self.normalize = T_v2.Normalize(mean, std)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input tensor (C, H, W) (RGB).
        Outputs a tensor of shape (C, H_dest, W_dest)
        """
        if self.resize:
            img = self._resize_with_aspect_ratio(img, self.dest_size)
        img = self._pad_to_size(img, self.dest_size, self.pad_color)
        img = T_v2.functional.center_crop(img, list(self.dest_size))
        img = self.normalize(img)

        return img
    
    @staticmethod
    def _pad_to_size(img: torch.Tensor, dest_size: tuple[int, int], pad_color: list[float]) -> torch.Tensor:
        """
        Pads a given image (C, H, W) to the destination size (H_dest, W_dest) with the given pad color. Padding is done to the 
        right and bottom of the image. 
        """
        target_height, target_width = dest_size
        height, width = img.shape[-2:]

        pad_right = max(0, target_width - width)
        pad_bottom = max(0, target_height - height)

        return T_v2.functional.pad(img, [0, 0, pad_right, pad_bottom], fill=pad_color)

    @staticmethod
    def _resize_with_aspect_ratio(img: torch.Tensor, dest_size: tuple[int, int]) -> torch.Tensor:
        """
        Resizes a given image (C, H, W) to the match at least one side in the destination size while keeping the aspect ratio.
        Should be used with padding to achieve the true destination size. 
        """
        target_height, target_width = dest_size
        height, width = img.shape[-2:]

        height_scale = target_height / height
        width_scale = target_width / width

        scale_factor = min(height_scale, width_scale)

        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        return T_v2.functional.resize(img, [new_height, new_width])
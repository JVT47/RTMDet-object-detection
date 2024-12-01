import torch
import torchvision.transforms.v2 as T_v2


class RTMDetPreprocessor:
    """
    Preprocessor that takes an input image tensor, resizes and normalizes
    it for inference. 
    """
    def __init__(self, dest_size: tuple[int, int], pad_color: list[float] = [114, 114, 114],
                 mean: list[float] = [103.53, 116.28, 123.675], std: list[float] = [57.375, 57.12, 58.395],
                 *args, **kwargs) -> None:
        """
        dest_size: the (H, W) output dimensions of the preprocessed images
        pad_color: the color used for potential padded regions
        mean: list of means used to normalize the input image. Default values from mmdet RMTDet implementation
        std: list of stds used to normalize the input image. Default values from mmdet RTMDet implementation
        """
        super().__init__(*args, **kwargs)

        assert dest_size[0] % 32 == 0 and dest_size[1] % 32 == 0, "RTMDet requires that input dimensions are divisible by 32"
        
        self.dest_size = dest_size
        self.pad_color = pad_color
        self.normalize = T_v2.Normalize(mean, std)
    
    def process_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input tensor (C, H, W) (RGB).
        Outputs a tensor of shape (C, H_dest, W_dest)
        """
        img = self.resize_with_aspect_ratio(img)
        img = self.normalize(img)

        return img
    
    def resize_with_aspect_ratio(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resizes a given image (C, H, W) to the match at least one side in the destination size while keeping the aspect ratio.
        The shorter side is padded if needed. Padding is done to the right and bottom side of the input image.
        """
        target_height, target_width = self.dest_size
        height, width = img.shape[-2:]

        height_scale = target_height / height
        width_scale = target_width / width

        scale_factor = min(height_scale, width_scale)

        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        img = T_v2.functional.resize(img, [new_height, new_width])

        return self._pad_to_size(img)

    def _pad_to_size(self, img: torch.Tensor) -> torch.Tensor:
        """
        Pads a given image (C, H, W) to the destination size (H_dest, W_dest) with the given pad color. Padding is done to the 
        right and bottom of the image. 
        """
        target_height, target_width = self.dest_size
        height, width = img.shape[-2:]

        pad_right = max(0, target_width - width)
        pad_bottom = max(0, target_height - height)

        return T_v2.functional.pad(img, [0, 0, pad_right, pad_bottom], fill=self.pad_color)
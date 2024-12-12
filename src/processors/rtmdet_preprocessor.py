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
        
        self.dest_size = torch.Size(dest_size)
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

    def process_bboxes(self, bboxes: torch.Tensor, img_shape: torch.Size) -> torch.Tensor:
        """
        Transform the bbox coordinates from the original image dimensions to the preprocessed image dimensions.
        bboxes: tensor of shape (n, 4) (x_min, y_min, x_max, y_max)
        img_shape: size of form (..., H, W). The original image's shape
        """
        height, width = img_shape[-2:]
        new_height, new_width = self._calc_new_height_width(img_shape)

        bboxes /= torch.tensor([width, height, width, height])
        bboxes *= torch.tensor([new_width, new_height, new_width, new_height])

        return bboxes
    
    def resize_with_aspect_ratio(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resizes a given image (C, H, W) to the match at least one side in the destination size while keeping the aspect ratio.
        The shorter side is padded if needed. Padding is done to the right and bottom side of the input image.
        """
        new_height, new_width = self._calc_new_height_width(img.shape)

        img = T_v2.functional.resize(img, [new_height, new_width])

        return self._pad_to_size(img)

    def _calc_new_height_width(self, img_shape: torch.Size) -> torch.Size:
        """
        Calculates the new height and width (unpadded) for a given input shape. The new dimensions
        have the same aspect ratio as the input.
        """
        target_height, target_width = self.dest_size
        height, width = img_shape[-2:]

        scale_factor = min(target_height / height, target_width / width)

        new_height = int(scale_factor * height)
        new_width = int(scale_factor * width)

        return torch.Size((new_height, new_width))

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
import torch

from src.processors.rtmdet_preprocessor import RTMDetPreprocessor


class TestRTMDetPreprocessor:
    preprocessor = RTMDetPreprocessor(
        dest_size=(32, 64),
        pad_color=[5, 15, 25],
        mean=[10, 10, 10],
        std=[1, 1, 1],
    )

    def test_forward(self) -> None:
        img = torch.ones((3, 32, 32)) * torch.tensor([0, 10, 20]).reshape((3, 1, 1))
        img = self.preprocessor.process_image(img)

        target_img = torch.ones((3, 32, 64)) * torch.tensor([-10, 0, 10]).reshape((3, 1, 1))
        target_img[:, :, 32:] = torch.ones((3, 32, 32)) * torch.tensor([-5, 5, 15]).reshape((3, 1, 1))

        torch.testing.assert_close(img, target_img)
    
    def test_pad_to_size_1(self) -> None:
        img = torch.ones((3, 10, 40))

        target_img = torch.ones((3, 32, 64)) * torch.Tensor(self.preprocessor.pad_color).reshape((3, 1, 1))
        target_img[:, :10, :40] = img

        img = self.preprocessor._pad_to_size(img)

        torch.testing.assert_close(img, target_img)
    
    def test_pad_to_size_2(self) -> None:
        img = torch.ones((3, 10, 5))

        target_img = torch.ones((3, 32, 64)) * torch.Tensor(self.preprocessor.pad_color).reshape(3, 1, 1)
        target_img[:, :10, :5] = img

        img = self.preprocessor._pad_to_size(img)

        torch.testing.assert_close(img, target_img)
    
    def test_resize_with_aspect_ratio_1(self) -> None:
        img = torch.ones((3, 10, 40))

        target_img = torch.ones((3, 32, 64)) * torch.tensor(self.preprocessor.pad_color).reshape((3, 1, 1))
        target_img[:, :16, :] = 1
        
        img = self.preprocessor.resize_with_aspect_ratio(img)

        torch.testing.assert_close(img, target_img)
    
    def test_resize_with_aspect_ratio_2(self) -> None:
        img = torch.ones((3, 32, 16))
        img = self.preprocessor.resize_with_aspect_ratio(img)

        target_img = torch.ones((3, 32, 64)) * torch.tensor(self.preprocessor.pad_color).reshape((3, 1, 1))
        target_img[:, :, :16] = 1

        torch.testing.assert_close(img, target_img)
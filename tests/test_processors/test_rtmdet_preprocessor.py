import torch

from src.processors.rtmdet_preprocessor import RTMDetPreprocessor


class TestRTMDetPreprocessor:
    preprocessor = RTMDetPreprocessor(
        dest_size=(32, 64),
        resize=True,
        pad_color=[5, 15, 25],
        mean=[10, 10, 10],
        std=[1, 1, 1],
    )

    def test_forward(self) -> None:
        img = torch.ones((3, 32, 32)) * torch.tensor([0, 10, 20]).reshape((3, 1, 1))
        img = self.preprocessor(img)

        target_img = torch.ones((3, 32, 64)) * torch.tensor([-10, 0, 10]).reshape((3, 1, 1))
        target_img[:, :, 32:] = torch.ones((3, 32, 32)) * torch.tensor([-5, 5, 15]).reshape((3, 1, 1))

        torch.testing.assert_close(img, target_img)
    
    def test_pad_to_size_1(self) -> None:
        img = torch.ones((3, 10, 20))
        dest_size = (20, 20)
        pad_color = [1.0, 2, 3]

        target_img = torch.ones((3, 20, 20)) * torch.Tensor(pad_color).reshape((3, 1, 1))
        target_img[:, :10, :] = img

        img = self.preprocessor._pad_to_size(img, dest_size, pad_color)

        torch.testing.assert_close(img, target_img)
    
    def test_pad_to_size_2(self) -> None:
        img = torch.ones((3, 10, 5))
        dest_size = (10, 10)
        pad_color = [5.0, 5, 5]

        target_img = torch.ones((3, 10, 10)) * torch.Tensor(pad_color).reshape(3, 1, 1)
        target_img[:, :, :5] = img

        img = self.preprocessor._pad_to_size(img, dest_size, pad_color)

        torch.testing.assert_close(img, target_img)
    
    def test_resize_with_aspect_ratio_1(self) -> None:
        img = torch.ones((3, 10, 20))
        dest_size = (20, 30)
        img = self.preprocessor._resize_with_aspect_ratio(img, dest_size)

        target_img = torch.ones((3, 15, 30))

        torch.testing.assert_close(img, target_img)
    
    def test_resize_with_aspect_ratio_2(self) -> None:
        img = torch.ones((3, 10, 20))
        dest_size = (40, 100)
        img = self.preprocessor._resize_with_aspect_ratio(img, dest_size)

        target_img = torch.ones((3, 40, 80))

        torch.testing.assert_close(img, target_img)
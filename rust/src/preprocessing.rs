use image::{DynamicImage, GenericImage, ImageBuffer, ImageResult, Rgb, RgbImage};
use ndarray::Array4;

/// Resizes the given rgb image to the given dimensions with the original aspect ratio.
/// Possible empty space is padded with values [114, 114, 114]. Additionally, normalizes
/// the image pixels with the RTMDet normalization values.
pub fn preprocess_image(image: &DynamicImage, width: u32, height: u32) -> Array4<f32> {
    let resized_image = resize_and_pad(image, width, height).unwrap();

    let mut result = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

    for (x, y, pixel) in resized_image.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        result[[0, 0, y as usize, x as usize]] = (r as f32 - 103.53) / 57.375;
        result[[0, 1, y as usize, x as usize]] = (g as f32 - 116.28) / 57.12;
        result[[0, 1, y as usize, x as usize]] = (b as f32 - 123.675) / 58.395;
    }

    result
}

/// Resizes the given rgb image to the specified width and target while preserving
/// the original aspect ratio. Possible empty space is padded with values [144, 144, 144]
fn resize_and_pad(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> ImageResult<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let resized_image = image.resize(width, height, image::imageops::FilterType::Lanczos3);
    let mut padded_image = RgbImage::from_pixel(width, height, Rgb([114, 114, 114]));

    padded_image.copy_from(&resized_image.to_rgb8(), 0, 0)?;

    Ok(padded_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};
    use ndarray::Array3;

    #[test]
    fn test_resize_and_pad_returns_correct_array() {
        let (width, height) = (10, 20);
        let img_buffer = RgbImage::from_raw(
            width,
            height,
            Array3::zeros((width as usize, height as usize, 3)).into_raw_vec(),
        )
        .unwrap();

        let image = DynamicImage::ImageRgb8(img_buffer);

        let (target_width, target_height) = (5, 5);
        let output = resize_and_pad(&image, target_width, target_height).unwrap();

        assert_eq!(output.width(), target_width);
        assert_eq!(output.height(), target_height);
    }
}

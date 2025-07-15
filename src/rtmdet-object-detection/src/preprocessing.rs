use image::{DynamicImage, GenericImage, ImageBuffer, Rgb, RgbImage};
use ndarray::Array4;

use crate::{errors::RTMDetResult, inference_engine::inference_config::PreprocessConfig};

/// Resizes the given rgb image to the given dimensions with the original aspect ratio.
/// Possible empty space is padded with values in padding_color. Additionally, normalizes
/// the image pixels with the RTMDet normalization values.
pub fn preprocess_image(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> RTMDetResult<Array4<f32>> {
    let resized_image = resize_and_pad(
        image,
        config.input_width,
        config.input_height,
        config.padding_color,
    )?;

    let mut result = Array4::<f32>::zeros((
        1,
        3,
        config.input_height as usize,
        config.input_width as usize,
    ));

    for (x, y, pixel) in resized_image.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        result[[0, 0, y as usize, x as usize]] =
            (r as f32 - config.color_mean[0]) / config.color_std[0];
        result[[0, 1, y as usize, x as usize]] =
            (g as f32 - config.color_mean[1]) / config.color_std[1];
        result[[0, 2, y as usize, x as usize]] =
            (b as f32 - config.color_mean[2]) / config.color_std[2];
    }

    Ok(result)
}

/// Resizes the given rgb image to the specified width and target while preserving
/// the original aspect ratio. Possible empty space is padded with values in padding_color
fn resize_and_pad(
    image: &DynamicImage,
    width: u32,
    height: u32,
    padding_color: [u8; 3],
) -> RTMDetResult<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let resized_image = image.resize(width, height, image::imageops::FilterType::Lanczos3);
    let mut padded_image = RgbImage::from_pixel(width, height, Rgb(padding_color));

    padded_image.copy_from(&resized_image.to_rgb8(), 0, 0)?;

    Ok(padded_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};
    use ndarray::{Array3, s};

    #[test]
    fn test_resize_and_pad_returns_correct_array() {
        let (width, height) = (10, 20);
        let img_buffer = RgbImage::from_raw(
            width,
            height,
            Array3::zeros((width as usize, height as usize, 3))
                .into_raw_vec_and_offset()
                .0,
        )
        .unwrap();

        let image = DynamicImage::ImageRgb8(img_buffer);

        let (target_width, target_height) = (5, 5);
        let output = resize_and_pad(&image, target_width, target_height, [114, 114, 114]).unwrap();

        assert_eq!(output.width(), target_width);
        assert_eq!(output.height(), target_height);
    }

    #[test]
    fn test_preprocess_image_resizes_and_normalizes_correctly() {
        let config = PreprocessConfig {
            input_width: 32,
            input_height: 32,
            padding_color: [2, 2, 2],
            color_mean: [2.0, 2.0, 2.0],
            color_std: [2.0, 2.0, 2.0],
        };

        let img_buffer =
            RgbImage::from_raw(10, 5, Array3::ones((10, 5, 3)).into_raw_vec_and_offset().0)
                .unwrap();

        let image = DynamicImage::ImageRgb8(img_buffer);

        let output = preprocess_image(&image, &config).unwrap();

        let img_slice = output.slice(s![.., .., 0..16, ..]);
        let padding_slice = output.slice(s![.., .., 16.., ..]);

        assert!(img_slice.iter().all(|&x| x == -0.5));
        assert!(padding_slice.iter().all(|&x| x == 0.0));
    }
}

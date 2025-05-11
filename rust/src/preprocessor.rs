use image::{DynamicImage, GenericImage, ImageBuffer, ImageResult, Rgb, RgbImage};
use ndarray::Array4;

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

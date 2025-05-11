use image::{DynamicImage, GenericImageView};
use itertools::Itertools;
use ndarray::{Axis, concatenate};

use crate::inference::run_inference;
use crate::postprocessor::{BBox, Detections, postprocess_outputs};
use crate::preprocessor::preprocess_image;

pub fn detect_from_images(images: Vec<DynamicImage>, width: u32, height: u32) -> Vec<Detections> {
    let original_shapes = get_original_shapes(&images);

    let images_iter = images
        .iter()
        .map(|image| preprocess_image(image, width, height))
        .into_iter();

    let batch_size = 2;

    let mut detections = Vec::with_capacity(images.len());

    for chunk in &images_iter.chunks(batch_size) {
        let batch_images = chunk.collect_vec();
        let batch_views: Vec<_> = batch_images.iter().map(|image| image.view()).collect();
        let batch = concatenate(Axis(0), &batch_views).unwrap();

        let model_outputs = run_inference(batch).unwrap();

        detections.extend(postprocess_outputs(model_outputs, 0.5, 0.3));
    }

    for (detections, original_shape) in detections.iter_mut().zip(original_shapes.iter()) {
        rescale_bbox_to_original_image(&mut detections.bboxes, *original_shape, (width, height));
    }

    detections
}

fn get_original_shapes(images: &Vec<DynamicImage>) -> Vec<(u32, u32)> {
    images.into_iter().map(|image| image.dimensions()).collect()
}

fn rescale_bbox_to_original_image(
    bboxes: &mut Vec<BBox>,
    original_shape: (u32, u32),
    rescaled_shape: (u32, u32),
) {
    let (orig_width, orig_height) = (original_shape.0 as f32, original_shape.1 as f32);
    let (rescaled_width, rescaled_height) = (rescaled_shape.0 as f32, rescaled_shape.1 as f32);

    let scale_factor = (rescaled_height / orig_height).min(rescaled_width / orig_width);

    let rescale_height = scale_factor * orig_height;
    let rescale_width = scale_factor * orig_width;

    for bbox in bboxes {
        bbox.top_left = (
            bbox.top_left.0 / rescale_width * orig_width,
            bbox.top_left.1 / rescale_height * orig_height,
        );
        bbox.bottom_right = (
            bbox.bottom_right.0 / rescale_width * orig_width,
            bbox.bottom_right.1 / rescale_height * orig_height,
        );
    }
}

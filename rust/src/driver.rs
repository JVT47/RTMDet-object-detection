use image::DynamicImage;
use itertools::Itertools;
use ndarray::{Axis, concatenate};

use crate::inference::run_inference;
use crate::postprocessor::{Detections, postprocess_outputs};
use crate::preprocessor::preprocess_image;

pub fn detect_from_images(images: Vec<DynamicImage>, width: u32, height: u32) -> Vec<Detections> {
    let images_iter = images
        .iter()
        .map(|image| preprocess_image(image, width, height))
        .into_iter();

    let batch_size = 2;

    let mut detections = vec![];

    for chunk in &images_iter.chunks(batch_size) {
        let batch_images = chunk.collect_vec();
        let batch_views: Vec<_> = batch_images.iter().map(|image| image.view()).collect();
        let batch = concatenate(Axis(0), &batch_views).unwrap();

        let model_outputs = run_inference(batch).unwrap();

        detections.extend(postprocess_outputs(model_outputs, 0.5, 0.3));
    }

    detections
}

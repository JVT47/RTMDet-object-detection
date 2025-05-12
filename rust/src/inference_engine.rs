mod inference;
pub mod inference_config;

use crate::{
    postprocessing::{bbox::BBox, detections::Detections, postprocess_outputs},
    preprocessing::preprocess_image,
};
use image::{DynamicImage, GenericImageView};
use inference::run_inference;
use inference_config::InferenceConfig;
use itertools::Itertools;
use ndarray::{Axis, concatenate};
use ort::{Environment, OrtError, Session, SessionBuilder};

pub struct InferenceEngine {
    session: Session,
    config: InferenceConfig,
}

impl InferenceEngine {
    pub fn new(model_path: &str, config: InferenceConfig) -> Result<Self, OrtError> {
        let environment = Environment::builder()
            .with_name("inference")
            .build()?
            .into_arc();

        let session = SessionBuilder::new(&environment)?.with_model_from_file(model_path)?;

        Ok(Self { session, config })
    }

    pub fn detect_from_images(&self, images: Vec<DynamicImage>) -> Vec<Detections> {
        let original_shapes = get_original_shapes(&images);

        let images_iter = images.iter().map(|image| {
            preprocess_image(image, self.config.input_width, self.config.input_height)
        });

        let mut detections = Vec::with_capacity(images.len());

        for chunk in &images_iter.chunks(self.config.batch_size) {
            let batch_images = chunk.collect_vec();
            let batch_views: Vec<_> = batch_images.iter().map(|image| image.view()).collect();
            let batch = concatenate(Axis(0), &batch_views).unwrap();

            let model_outputs = run_inference(&self.session, batch).unwrap();

            detections.extend(postprocess_outputs(model_outputs, 0.5, 0.3));
        }

        for (detections, original_shape) in detections.iter_mut().zip(original_shapes.iter()) {
            rescale_bbox_to_original_image(
                &mut detections.bboxes,
                *original_shape,
                (self.config.input_width, self.config.input_height),
            );
        }

        detections
    }
}

fn get_original_shapes(images: &[DynamicImage]) -> Vec<(u32, u32)> {
    images.iter().map(|image| image.dimensions()).collect()
}

fn rescale_bbox_to_original_image(
    bboxes: &mut [BBox],
    original_shape: (u32, u32),
    rescaled_shape: (u32, u32),
) {
    bboxes
        .iter_mut()
        .for_each(|bbox| bbox.rescale_to_original_image(original_shape, rescaled_shape));
}

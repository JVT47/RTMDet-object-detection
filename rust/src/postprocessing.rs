pub mod bbox;
pub mod detection_output;
use detection_output::DetectionOutput;
use ndarray::{Array4, ArrayView3, s};

fn get_batch_element(array: &Array4<f32>, index: usize) -> ArrayView3<f32> {
    array.slice(s![index, .., .., ..])
}

pub fn postprocess_outputs(
    outputs: Vec<Array4<f32>>,
    score_threshold: f32,
    iou_threshold: f32,
) -> Vec<DetectionOutput> {
    let (num_batches, _, _, _) = outputs[0].dim();

    let mut result = vec![];
    for i in 0..num_batches {
        result.push(DetectionOutput::new(
            (
                get_batch_element(&outputs[0], i),
                get_batch_element(&outputs[1], i),
                get_batch_element(&outputs[2], i),
            ),
            (
                get_batch_element(&outputs[3], i),
                get_batch_element(&outputs[4], i),
                get_batch_element(&outputs[5], i),
            ),
            score_threshold,
            iou_threshold,
        ));
    }

    result
}

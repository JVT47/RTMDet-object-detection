use ndarray::{ArrayView3, s};
use pyo3::pyclass;

use super::bbox::BBox;

/// A struct that holds all of the detected bounding boxes for an image
#[pyclass]
pub struct DetectionResult {
    /// A vec of the detected bounding boxes
    #[pyo3(get)]
    pub bboxes: Vec<BBox>,
}

impl DetectionResult {
    /// Creates a detection result from the given model predictions for the given image.
    /// The detection result contains a vector of bounding boxes sorted by score in
    /// descending order.
    pub fn new(
        cls_preds: (ArrayView3<f32>, ArrayView3<f32>, ArrayView3<f32>),
        reg_preds: (ArrayView3<f32>, ArrayView3<f32>, ArrayView3<f32>),
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Self {
        let mut bboxes = vec![];

        bboxes.extend(DetectionResult::extract_bboxes(
            &cls_preds.0,
            &reg_preds.0,
            8.0,
            score_threshold,
        ));
        bboxes.extend(DetectionResult::extract_bboxes(
            &cls_preds.1,
            &reg_preds.1,
            16.0,
            score_threshold,
        ));
        bboxes.extend(DetectionResult::extract_bboxes(
            &cls_preds.2,
            &reg_preds.2,
            32.0,
            score_threshold,
        ));

        bboxes = DetectionResult::nms(bboxes, iou_threshold);

        Self { bboxes }
    }

    /// Extract the bounding boxes from the model output layer that exceed
    /// the score threshold
    fn extract_bboxes(
        cls: &ArrayView3<f32>,
        reg: &ArrayView3<f32>,
        scale: f32,
        score_threshold: f32,
    ) -> Vec<BBox> {
        let (_, height, width) = cls.dim();

        let mut bboxes = vec![];
        for y in 0..height {
            for x in 0..width {
                let cls_preds = cls.slice(s![.., y, x]);
                if let Some((class, max_value)) = cls_preds
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                {
                    let score = sigmoid(*max_value);

                    if score < score_threshold {
                        continue;
                    }

                    let center_x = x as f32 * scale;
                    let center_y = y as f32 * scale;
                    let top_left = (center_x - reg[[0, y, x]], center_y - reg[[1, y, x]]);
                    let bottom_right = (center_x + reg[[2, y, x]], center_y + reg[[3, y, x]]);

                    bboxes.push(BBox {
                        top_left,
                        bottom_right,
                        class_num: class as u32,
                        score,
                    });
                }
            }
        }
        bboxes
    }

    /// Performs non maximum suppression to the given bounding boxes
    /// Return a vector of the selected bounding boxes sorted by
    /// score in descending order
    fn nms(mut bboxes: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
        bboxes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        let mut best_bboxes: Vec<BBox> = vec![];

        for bbox in bboxes.iter().rev() {
            let mut add = true;
            for selected_bbox in &best_bboxes {
                if bbox.class_num == selected_bbox.class_num
                    && bbox.iou(selected_bbox) > iou_threshold
                {
                    add = false;
                    break;
                }
            }
            if add {
                best_bboxes.push(*bbox);
            }
        }

        best_bboxes
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array3, CowArray};

    use super::*;

    #[test]
    fn test_new_returns_correct_bboxes() {
        let mut cls1 = Array3::<f32>::ones((3, 4, 4)) * -1.0;
        cls1.slice_mut(s![.., 1, 1])
            .assign(&CowArray::from(&[0.0, 10.0, 5.0]));
        let mut cls2 = Array3::<f32>::ones((3, 2, 2)) * -1.0;
        cls2.slice_mut(s![.., 1, 1])
            .assign(&CowArray::from(&[1.0, 0.0, -10.0]));
        let cls3 = Array3::<f32>::ones((3, 1, 1)) * -1.0;

        let mut reg1 = Array3::<f32>::zeros((4, 4, 4));
        reg1.slice_mut(s![.., 1, 1])
            .assign(&CowArray::from(&[1.0, 1.0, 0.0, 0.0])); // bbox [7, 7, 8, 8]
        let mut reg2 = Array3::<f32>::zeros((4, 2, 2));
        reg2.slice_mut(s![.., 1, 1])
            .assign(&CowArray::from(&[0.0, 0.0, 1.0, 1.0])); // bbox [16, 16, 17, 17]
        let reg3 = Array3::<f32>::zeros((4, 1, 1));

        let detection_result = DetectionResult::new(
            ((&cls1).into(), (&cls2).into(), (&cls3).into()),
            ((&reg1).into(), (&reg2).into(), (&reg3).into()),
            0.5,
            0.3,
        );

        let target_bboxes = vec![
            BBox {
                top_left: (7.0, 7.0),
                bottom_right: (8.0, 8.0),
                class_num: 1,
                score: 0.999,
            },
            BBox {
                top_left: (16.0, 16.0),
                bottom_right: (17.0, 17.0),
                class_num: 0,
                score: 0.731,
            },
        ];

        for (bbox, target) in detection_result.bboxes.iter().zip(&target_bboxes) {
            assert_eq!(bbox.top_left, target.top_left);
            assert_eq!(bbox.bottom_right, target.bottom_right);
            assert_eq!(bbox.class_num, target.class_num);
            assert!((bbox.score - target.score).abs() < 0.001);
        }
    }
}

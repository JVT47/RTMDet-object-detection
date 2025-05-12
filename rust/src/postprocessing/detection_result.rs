use ndarray::{ArrayView3, s};
use pyo3::pyclass;

use super::bbox::BBox;

#[pyclass]
pub struct DetectionResult {
    #[pyo3(get)]
    pub bboxes: Vec<BBox>,
}

impl DetectionResult {
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

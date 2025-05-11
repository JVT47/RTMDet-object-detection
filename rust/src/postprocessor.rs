use ndarray::{Array4, ArrayView3, s};
use pyo3::pyclass;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Copy)]
#[pyclass]
pub struct BBox {
    #[pyo3(get)]
    pub top_left: (f32, f32),
    #[pyo3(get)]
    pub bottom_right: (f32, f32),
    #[pyo3(get)]
    pub class_num: u32,
    #[pyo3(get)]
    pub score: f32,
}

impl BBox {
    fn iou(&self, other: &BBox) -> f32 {
        let union_area = self.union_area(other);
        if union_area == 0.0 {
            return 0.0;
        }
        self.intersection_area(other) / self.union_area(other)
    }

    fn area(&self) -> f32 {
        ((self.bottom_right.0 - self.top_left.0) * (self.bottom_right.1 - self.top_left.1)).max(0.0)
    }

    fn intersection_area(&self, other: &BBox) -> f32 {
        let i_width = (self.bottom_right.0.min(other.bottom_right.0)
            - self.top_left.0.max(other.top_left.0))
        .max(0.0);
        let i_height = (self.bottom_right.1.min(other.bottom_right.1)
            - self.top_left.1.max(other.top_left.1))
        .max(0.0);

        i_height * i_width
    }

    fn union_area(&self, other: &BBox) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }
}

#[pyclass]
pub struct Detections {
    #[pyo3(get)]
    pub bboxes: Vec<BBox>,
}

impl Detections {
    pub fn new(
        cls1: ArrayView3<f32>,
        cls2: ArrayView3<f32>,
        cls3: ArrayView3<f32>,
        reg1: ArrayView3<f32>,
        reg2: ArrayView3<f32>,
        reg3: ArrayView3<f32>,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Self {
        let mut bboxes = vec![];

        bboxes.extend(Detections::extract_bboxes(
            &cls1,
            &reg1,
            8.0,
            score_threshold,
        ));
        bboxes.extend(Detections::extract_bboxes(
            &cls2,
            &reg2,
            16.0,
            score_threshold,
        ));
        bboxes.extend(Detections::extract_bboxes(
            &cls3,
            &reg3,
            32.0,
            score_threshold,
        ));

        bboxes = Detections::nms(bboxes, iou_threshold);

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
                    let bottom_right = (center_x + reg[[2, y, x]], center_y + reg[[3, x, y]]);

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
                    && bbox.iou(&selected_bbox) > iou_threshold
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

fn get_batch_element(array: &Array4<f32>, index: usize) -> ArrayView3<f32> {
    array.slice(s![index, .., .., ..])
}

pub fn postprocess_outputs(
    outputs: Vec<Array4<f32>>,
    score_threshold: f32,
    iou_threshold: f32,
) -> Vec<Detections> {
    let (num_batches, _, _, _) = outputs[0].dim();

    let mut result = vec![];
    for i in 0..num_batches {
        result.push(Detections::new(
            get_batch_element(&outputs[0], i),
            get_batch_element(&outputs[1], i),
            get_batch_element(&outputs[2], i),
            get_batch_element(&outputs[3], i),
            get_batch_element(&outputs[4], i),
            get_batch_element(&outputs[5], i),
            score_threshold,
            iou_threshold,
        ));
    }

    result
}

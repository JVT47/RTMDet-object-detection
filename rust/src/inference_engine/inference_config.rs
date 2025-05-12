#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub batch_size: usize,
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

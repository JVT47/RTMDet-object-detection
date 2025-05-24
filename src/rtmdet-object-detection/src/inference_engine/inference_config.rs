#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub preprocess_config: PreprocessConfig,
    pub batch_size: usize,
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub padding_color: [u8; 3],
    pub color_mean: [f32; 3],
    pub color_std: [f32; 3],
}

use pyo3::pyclass;

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
    pub fn iou(&self, other: &BBox) -> f32 {
        let union_area = self.union_area(other);
        if union_area == 0.0 {
            return 0.0;
        }
        self.intersection_area(other) / self.union_area(other)
    }

    pub fn area(&self) -> f32 {
        ((self.bottom_right.0 - self.top_left.0) * (self.bottom_right.1 - self.top_left.1)).max(0.0)
    }

    pub fn intersection_area(&self, other: &BBox) -> f32 {
        let i_width = (self.bottom_right.0.min(other.bottom_right.0)
            - self.top_left.0.max(other.top_left.0))
        .max(0.0);
        let i_height = (self.bottom_right.1.min(other.bottom_right.1)
            - self.top_left.1.max(other.top_left.1))
        .max(0.0);

        i_height * i_width
    }

    pub fn union_area(&self, other: &BBox) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    pub fn rescale_to_original_image(
        &mut self,
        original_shape: (u32, u32),
        rescaled_shape: (u32, u32),
    ) {
        let (orig_width, orig_height) = (original_shape.0 as f32, original_shape.1 as f32);
        let (rescaled_width, rescaled_height) = (rescaled_shape.0 as f32, rescaled_shape.1 as f32);

        let scale_factor = (rescaled_height / orig_height).min(rescaled_width / orig_width);

        let rescale_width = scale_factor * orig_width;
        let rescale_height = scale_factor * orig_height;

        self.top_left = (
            self.top_left.0 / rescale_width * orig_width,
            self.top_left.1 / rescale_height * orig_height,
        );
        self.bottom_right = (
            self.bottom_right.0 / rescale_width * orig_width,
            self.bottom_right.1 / rescale_height * orig_height,
        )
    }
}

use pyo3::pyclass;

/// A struct that represents a detected bounding box
#[derive(Debug, Clone, Copy)]
#[pyclass]
pub struct BBox {
    /// top left (x,y) corner of the bounding box
    #[pyo3(get)]
    pub top_left: (f32, f32),
    /// bottom right (x,y) corner of the bounding box
    #[pyo3(get)]
    pub bottom_right: (f32, f32),
    /// the index of the detected class. Starts from 0
    #[pyo3(get)]
    pub class_num: u32,
    /// the confidence of the detected class
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

    /// Calculates the area of the bounding box.
    ///
    /// Returns 0.0 if the bounding box would have a negative area
    pub fn area(&self) -> f32 {
        ((self.bottom_right.0 - self.top_left.0) * (self.bottom_right.1 - self.top_left.1)).max(0.0)
    }

    /// Calculates the area of the intersection with the given bounding box
    pub fn intersection_area(&self, other: &BBox) -> f32 {
        let i_width = (self.bottom_right.0.min(other.bottom_right.0)
            - self.top_left.0.max(other.top_left.0))
        .max(0.0);
        let i_height = (self.bottom_right.1.min(other.bottom_right.1)
            - self.top_left.1.max(other.top_left.1))
        .max(0.0);

        i_height * i_width
    }

    /// Calculates the area of the union with the given bounding box
    pub fn union_area(&self, other: &BBox) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Rescales the top left and bottom right coordinates from the
    /// dimensions used in model inference to the original image
    /// dimensions
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_returns_correct_value() {
        let bbox1 = BBox {
            top_left: (0.0, 0.0),
            bottom_right: (2.0, 1.0),
            class_num: 0,
            score: 0.5,
        };

        assert_eq!(2.0, bbox1.area());
    }

    #[test]
    fn test_area_on_invalid_box_returns_zero() {
        let bbox1 = BBox {
            top_left: (3.0, 0.0),
            bottom_right: (2.0, 1.0),
            class_num: 0,
            score: 0.5,
        };

        assert_eq!(0.0, bbox1.area());
    }

    #[test]
    fn test_iou_returns_correct_value() {
        let bbox1 = BBox {
            top_left: (0.0, 0.0),
            bottom_right: (2.0, 1.0),
            class_num: 0,
            score: 0.5,
        };
        let bbox2 = BBox {
            top_left: (1.0, 0.0),
            bottom_right: (2.0, 2.0),
            class_num: 0,
            score: 0.5,
        };

        assert_eq!(1.0 / 3.0, bbox1.iou(&bbox2));
    }

    #[test]
    fn test_rescale_to_original_image() {
        let mut bbox1 = BBox {
            top_left: (0.0, 0.0),
            bottom_right: (2.0, 1.0),
            class_num: 0,
            score: 0.5,
        };

        bbox1.rescale_to_original_image((6, 8), (4, 4));

        assert_eq!((0.0, 0.0), bbox1.top_left);
        assert_eq!((4.0, 2.0), bbox1.bottom_right);
    }
}

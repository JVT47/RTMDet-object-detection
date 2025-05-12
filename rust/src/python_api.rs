use image::{DynamicImage, ImageBuffer};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{
    Py, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
};

use crate::{
    inference_engine::{InferenceEngine, inference_config::InferenceConfig},
    postprocessing::detection_result::DetectionResult,
};

#[pyclass]
pub struct RTMDetDetector {
    engine: InferenceEngine,
}

#[pymethods]
impl RTMDetDetector {
    #[new]
    #[pyo3(signature = (model_path, inference_shape, batch_size, score_threshold=0.5, iou_threshold=0.3))]
    pub fn new(
        model_path: &str,
        inference_shape: (u32, u32),
        batch_size: usize,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> PyResult<Self> {
        let config = InferenceConfig {
            input_width: inference_shape.0,
            input_height: inference_shape.1,
            batch_size,
            score_threshold,
            iou_threshold,
        };

        let engine = InferenceEngine::new(model_path, config)
            .map_err(|_| PyRuntimeError::new_err("Failed to initialize"))?;

        Ok(Self { engine })
    }

    pub fn detect_from_numpy(
        &self,
        py: Python<'_>,
        arrays: Vec<Py<PyArrayDyn<u8>>>,
    ) -> PyResult<Vec<DetectionResult>> {
        let mut images = Vec::with_capacity(arrays.len());

        for array in arrays {
            let array = array.bind(py);
            let readonly = array.readonly();
            let image = numpy_array_to_dynamic_image(readonly)?;
            images.push(image);
        }

        let detections = self.engine.detect_from_images(images);

        Ok(detections)
    }
}

fn numpy_array_to_dynamic_image(array: PyReadonlyArrayDyn<'_, u8>) -> PyResult<DynamicImage> {
    let shape = array.shape();

    match shape {
        [height, width, 3] => {
            let data = array.as_slice()?;
            let image = ImageBuffer::from_raw(*width as u32, *height as u32, data.to_vec())
                .ok_or_else(|| PyValueError::new_err("Failed to create ImageBuffer from NumPy"))?;
            Ok(DynamicImage::ImageRgb8(image))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unsupported image shape {:?}, expected (H, W, 3)",
            shape
        ))),
    }
}

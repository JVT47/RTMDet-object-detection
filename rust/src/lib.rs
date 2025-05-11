use std::path::Path;

use driver::detect_from_images;
use image::{DynamicImage, ImageBuffer};
use inference::build_session;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use postprocessor::{BBox, Detections};
use pyo3::{
    Bound, Py, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

pub mod driver;
pub mod inference;
pub mod postprocessor;
pub mod preprocessor;

#[pymodule]
fn rtmdet_object_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BBox>()?;
    m.add_class::<Detections>()?;
    m.add_function(wrap_pyfunction!(detect_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(init_session, m)?)?;

    Ok(())
}

#[pyfunction(name = "init_session")]
pub fn init_session<'a>(model_path: &str) -> PyResult<()> {
    let path = Path::new(model_path);
    build_session(path).map_err(|_| PyRuntimeError::new_err("Failed to initialize the session"))?;
    Ok(())
}

#[pyfunction(name = "detect_from_numpy")]
pub fn detect_from_numpy<'a>(
    py: Python<'a>,
    arrays: Vec<Py<PyArrayDyn<u8>>>,
) -> PyResult<Vec<Detections>> {
    let mut images = Vec::with_capacity(arrays.len());

    for array in arrays {
        let array = array.bind(py);
        let readonly = array.readonly();
        let image = numpy_array_to_dynamic_image(readonly)?;
        images.push(image)
    }

    let detections = detect_from_images(images, 640, 640);

    Ok(detections)
}

fn numpy_array_to_dynamic_image<'a>(array: PyReadonlyArrayDyn<'a, u8>) -> PyResult<DynamicImage> {
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

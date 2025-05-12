pub mod inference_engine;
pub mod postprocessing;
pub mod preprocessing;
pub mod python_api;

use postprocessing::{bbox::BBox, detections::Detections};
use pyo3::{
    Bound, PyResult, pymodule,
    types::{PyModule, PyModuleMethods},
};
use python_api::RTMDetDetector;

#[pymodule]
fn rtmdet_object_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BBox>()?;
    m.add_class::<Detections>()?;
    m.add_class::<RTMDetDetector>()?;

    Ok(())
}

use ndarray::{Array4, Ix4};
use ort::{
    session::{Session, SessionOutputs},
    value::{Tensor, Value},
};

use crate::errors::RTMDetResult;

pub fn run_inference(
    session: &mut Session,
    input_array: Array4<f32>,
) -> RTMDetResult<Vec<Array4<f32>>> {
    let input_tensor: Value<ort::value::TensorValueType<f32>> = Tensor::from_array(input_array)?;
    let model_outputs = session.run(ort::inputs![input_tensor])?;

    extract_model_outputs(model_outputs)
}

fn extract_model_outputs(model_outputs: SessionOutputs) -> RTMDetResult<Vec<Array4<f32>>> {
    model_outputs
        .iter()
        .map(|(_, out)| {
            let array = out.try_extract_array::<f32>()?;
            let array4 = array.view().to_owned().into_dimensionality::<Ix4>()?;
            Ok(array4)
        })
        .collect::<RTMDetResult<Vec<_>>>()
}

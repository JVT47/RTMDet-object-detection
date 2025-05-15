use ndarray::{Array4, CowArray, Ix4};
use ort::{Session, Value};

use crate::errors::RTMDetResult;

pub fn run_inference(
    session: &Session,
    input_array: Array4<f32>,
) -> RTMDetResult<Vec<Array4<f32>>> {
    let array = CowArray::from(input_array.into_dyn());
    let input_value = Value::from_array(session.allocator(), &array)?;

    let model_outputs = session.run(vec![input_value])?;

    extract_model_outputs(model_outputs)
}

fn extract_model_outputs(model_outputs: Vec<Value<'static>>) -> RTMDetResult<Vec<Array4<f32>>> {
    model_outputs
        .iter()
        .map(|out| {
            let array = out.try_extract::<f32>()?;
            let array4 = array.view().to_owned().into_dimensionality::<Ix4>()?;
            Ok(array4)
        })
        .collect::<RTMDetResult<Vec<_>>>()
}

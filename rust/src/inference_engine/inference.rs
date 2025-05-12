use ndarray::{Array4, CowArray, Ix4};
use ort::{Session, Value};

pub fn run_inference(
    session: &Session,
    input_array: Array4<f32>,
) -> Result<Vec<Array4<f32>>, String> {
    let array = CowArray::from(input_array.into_dyn());
    let input_value = Value::from_array(session.allocator(), &array).unwrap();

    let model_outputs = session.run(vec![input_value]).unwrap();

    Ok(extract_model_outputs(model_outputs))
}

fn extract_model_outputs(model_outputs: Vec<Value<'static>>) -> Vec<Array4<f32>> {
    model_outputs
        .iter()
        .map(|out| {
            out.try_extract::<f32>()
                .unwrap()
                .view()
                .to_owned()
                .into_dimensionality::<Ix4>()
                .unwrap()
        })
        .collect()
}

use ndarray::{Array4, CowArray, Ix4};
use ort::{Environment, Session, SessionBuilder, Value};
use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

static ENV: Mutex<Option<Arc<Environment>>> = Mutex::new(None);
static SESSION: Mutex<Option<Session>> = Mutex::new(None);

pub fn build_session(model_path: &Path) -> Result<(), String> {
    let mut env_guard = ENV.lock().map_err(|_| "Failed to lock ENV")?;
    let mut session_guard = SESSION.lock().map_err(|_| "Failed to lock SESSION")?;

    let environment = Environment::builder()
        .with_name("inference")
        .build()
        .map_err(|e| format!("Environment init error: {:?}", e))?
        .into_arc();

    let session = SessionBuilder::new(&environment)
        .map_err(|e| format!("SessionBuilder error: {:?}", e))?
        .with_model_from_file(model_path)
        .map_err(|e| format!("Model load error: {:?}", e))?;

    *env_guard = Some(environment);
    *session_guard = Some(session);

    Ok(())
}

pub fn get_session() -> Result<MutexGuard<'static, Option<Session>>, String> {
    SESSION
        .lock()
        .map_err(|_| "Failed to lock SESSION".to_string())
}

pub fn run_inference(input_array: Array4<f32>) -> Result<Vec<Array4<f32>>, String> {
    let guard = get_session()?;
    let session = guard.as_ref().ok_or("Session not initialized")?;

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

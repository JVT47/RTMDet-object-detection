use image::ImageError;
use ndarray::ShapeError;
use ort::Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RTMDetError {
    #[error("Image error: {0}")]
    Image(#[from] ImageError),

    #[error("ORT error: {0}")]
    Ort(#[from] Error),

    #[error("Shape error: {0}")]
    Shape(#[from] ShapeError),
}

pub type RTMDetResult<T> = Result<T, RTMDetError>;

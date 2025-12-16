//! ONNX Runtime Session - Simplified

use std::path::Path;
use ndarray::ArrayD;
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel};
use crate::error::{Result, ZipEnhancerError};

/// Session configuration
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    pub intra_op_num_threads: i32,
}

/// Dynamic tensor for ONNX I/O
#[derive(Debug, Clone)]
pub struct DynamicTensor {
    data: TensorData,
    shape: Vec<i64>,
}

#[derive(Debug, Clone)]
enum TensorData {
    Float32(Vec<f32>),
    Int16(Vec<i16>),
}

impl DynamicTensor {
    pub fn new_i16(data: Vec<i16>, shape: Vec<i64>) -> Self {
        Self { data: TensorData::Int16(data), shape }
    }

    pub fn shape(&self) -> &[i64] { &self.shape }

    pub fn into_ndarray(self) -> ArrayD<f32> {
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let data = match self.data {
            TensorData::Float32(d) => d,
            TensorData::Int16(d) => d.into_iter().map(|x| x as f32 / 32767.0).collect(),
        };
        ArrayD::from_shape_vec(shape, data).expect("Shape mismatch")
    }

    pub fn into_i16_ndarray(self) -> ArrayD<i16> {
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let data = match self.data {
            TensorData::Int16(d) => d,
            TensorData::Float32(d) => d.into_iter().map(|x| (x.clamp(-1.0, 1.0) * 32767.0) as i16).collect(),
        };
        ArrayD::from_shape_vec(shape, data).expect("Shape mismatch")
    }

    fn new_f32(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self { data: TensorData::Float32(data), shape }
    }
}

/// ONNX inference session
pub struct OnnxSession {
    session: Session<'static>,
    #[allow(dead_code)]
    env_holder: Box<Environment>,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession").finish()
    }
}

impl OnnxSession {
    pub fn new(model_path: &Path, config: SessionConfig) -> Result<Self> {
        if !model_path.exists() {
            return Err(ZipEnhancerError::onnx(format!("Model not found: {}", model_path.display())));
        }

        let env = Environment::builder()
            .with_name("zipenhancer")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .map_err(|e| ZipEnhancerError::onnx(format!("Environment error: {}", e)))?;

        let env_box = Box::new(env);
        let env_ref: &'static Environment = unsafe { &*(env_box.as_ref() as *const Environment) };
        let model_path_static: &'static Path = Box::leak(model_path.to_path_buf().into_boxed_path());

        let session = env_ref.new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_number_threads(config.intra_op_num_threads.max(1) as i16)?
            .with_model_from_file(model_path_static)?;

        log::info!("ONNX session created");

        Ok(Self { session, env_holder: env_box })
    }

    pub fn run(&mut self, inputs: Vec<DynamicTensor>) -> Result<Vec<DynamicTensor>> {
        let input_arrays: Vec<ArrayD<i16>> = inputs.into_iter().map(|t| t.into_i16_ndarray()).collect();

        let outputs = self.session.run(input_arrays)
            .map_err(|e| ZipEnhancerError::onnx(format!("Inference failed: {}", e)))?;

        Ok(outputs.into_iter().map(|tensor| {
            let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
            let data: Vec<f32> = tensor.as_slice().unwrap().iter().map(|&x: &i16| x as f32 / 32767.0).collect();
            DynamicTensor::new_f32(data, shape)
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_tensor() {
        let t = DynamicTensor::new_i16(vec![16384, -16384], vec![1, 2]);
        assert_eq!(t.shape(), &[1, 2]);
        let arr = t.into_ndarray();
        assert_eq!(arr.len(), 2);
        assert!((arr[[0, 0]] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.intra_op_num_threads, 0);
    }
}

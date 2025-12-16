//! ONNX Inference Engine - Simplified with retry and stats

use std::path::Path;
use std::time::{Duration, Instant};
use ndarray::ArrayD;
use crate::error::Result;
use super::{OnnxSession, SessionConfig, DynamicTensor};

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub intra_threads: i32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self { max_retries: 3, retry_delay_ms: 100, intra_threads: 1 }
    }
}

/// Inference statistics
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    pub total: u64,
    pub success: u64,
    pub failed: u64,
    pub total_time_ms: u64,
}

impl InferenceStats {
    pub fn record(&mut self, time_ms: u64, success: bool) {
        self.total += 1;
        if success {
            self.success += 1;
            self.total_time_ms += time_ms;
        } else {
            self.failed += 1;
        }
    }

    pub fn avg_time_ms(&self) -> f64 {
        if self.success > 0 { self.total_time_ms as f64 / self.success as f64 } else { 0.0 }
    }
}

/// Inference result
#[derive(Debug)]
pub struct InferenceResult {
    pub outputs: Vec<ArrayD<f32>>,
    pub time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

impl InferenceResult {
    pub fn ok(outputs: Vec<ArrayD<f32>>, time_ms: u64) -> Self {
        Self { outputs, time_ms, success: true, error: None }
    }

    pub fn err(msg: String, time_ms: u64) -> Self {
        Self { outputs: vec![], time_ms, success: false, error: Some(msg) }
    }

    pub fn first_output(&self) -> Option<&ArrayD<f32>> {
        self.outputs.first()
    }
}

/// Unified inference engine
pub struct InferenceEngine {
    session: OnnxSession,
    config: InferenceConfig,
    stats: InferenceStats,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("session", &self.session)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl InferenceEngine {
    pub fn new(model_path: &Path, config: InferenceConfig) -> Result<Self> {
        let session_config = SessionConfig {
            intra_op_num_threads: config.intra_threads,
            ..Default::default()
        };
        let session = OnnxSession::new(model_path, session_config)?;
        Ok(Self { session, config, stats: InferenceStats::default() })
    }

    pub fn from_session(session: OnnxSession, config: InferenceConfig) -> Self {
        Self { session, config, stats: InferenceStats::default() }
    }

    pub fn run(&mut self, inputs: Vec<DynamicTensor>) -> Result<InferenceResult> {
        let start = Instant::now();
        let mut last_err = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(self.config.retry_delay_ms));
                log::warn!("Retry {}/{}", attempt, self.config.max_retries);
            }

            match self.session.run(inputs.clone()) {
                Ok(outputs) => {
                    let time_ms = start.elapsed().as_millis() as u64;
                    self.stats.record(time_ms, true);
                    let arrays = outputs.into_iter().map(|t| t.into_ndarray()).collect();
                    return Ok(InferenceResult::ok(arrays, time_ms));
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                    log::error!("Inference failed (attempt {}): {}", attempt + 1, last_err.as_ref().unwrap());
                }
            }
        }

        let time_ms = start.elapsed().as_millis() as u64;
        self.stats.record(time_ms, false);
        Ok(InferenceResult::err(last_err.unwrap_or_else(|| "Unknown error".into()), time_ms))
    }

    pub fn warm_up(&mut self) -> Result<()> {
        log::info!("Warming up...");
        let dummy = DynamicTensor::new_i16(vec![0i16; 16000], vec![1, 1, 16000]);
        let _ = self.run(vec![dummy])?;
        log::info!("Warm-up complete");
        Ok(())
    }

    pub fn stats(&self) -> &InferenceStats { &self.stats }
    pub fn session(&self) -> &OnnxSession { &self.session }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let c = InferenceConfig::default();
        assert_eq!(c.max_retries, 3);
    }

    #[test]
    fn test_stats() {
        let mut s = InferenceStats::default();
        s.record(100, true);
        s.record(200, true);
        assert_eq!(s.avg_time_ms(), 150.0);
    }
}

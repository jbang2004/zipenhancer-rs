//! Audio Processing Pipeline - Serial mode

use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;
use crate::audio::WavAudio;
use crate::onnx::{InferenceEngine, InferenceConfig};
use crate::processing::{AudioPreprocessor, AudioPostprocessor, PreprocessingConfig, PostprocessingConfig, AudioSegment};
use crate::processing::common::{prepare_audio, to_onnx_input, apply_agc, normalize_output, build_audio_segment};
use crate::config::Config;
use crate::error::{ZipEnhancerError, Result};

#[derive(Debug)]
pub struct AudioProcessor {
    config: Config,
    engine: InferenceEngine,
    preprocessor: AudioPreprocessor,
    postprocessor: AudioPostprocessor,
}

impl AudioProcessor {
    pub fn new(config: Config) -> Result<Self> {
        if config.verbose() { println!("Initializing processor..."); }

        let engine = InferenceEngine::new(&config.model_path(), InferenceConfig {
            max_retries: config.max_retries(),
            intra_threads: config.inference_threads() as i32,
            ..Default::default()
        })?;

        if config.verbose() {
            println!("Model: {}", config.model_path().display());
            println!("Threads: {}", config.inference_threads());
        }

        Ok(Self {
            preprocessor: AudioPreprocessor::new(PreprocessingConfig {
                target_sample_rate: config.sample_rate(),
                segment_size: config.segment_size(),
                ..Default::default()
            }),
            postprocessor: AudioPostprocessor::new(PostprocessingConfig {
                output_sample_rate: config.sample_rate(),
                overlap_ratio: config.overlap_ratio(),
                output_format: crate::audio::AudioFormat::Int16,
            }),
            config,
            engine,
        })
    }

    pub fn process_file(&mut self, input: &Path, output: &Path) -> Result<ProcessingResult> {
        let start = Instant::now();

        let mut audio = WavAudio::from_file(input)?;
        if self.config.verbose() {
            println!("Audio: {:.2}s, {}Hz", audio.duration(), audio.sample_rate());
        }

        prepare_audio(&mut audio, &self.config)?;

        let segments = self.preprocessor.preprocess_and_segment(&audio)?;
        if self.config.verbose() { println!("Segments: {}", segments.len()); }

        let processed = self.run_inference(&segments)?;

        let audio_segments: Vec<_> = processed.iter()
            .map(|(idx, data, _)| build_audio_segment(*idx, data.clone(), &segments[*idx]))
            .collect();

        let mut output_data = self.postprocessor.reconstruct_from_segments(&audio_segments)?;
        normalize_output(&mut output_data, self.config.verbose());

        self.postprocessor.create_wav_audio(output_data)?.save_to_file(output)?;

        let processing_time = start.elapsed();
        let avg_inference = processed.iter().map(|(_, _, t)| *t as f64).sum::<f64>() / segments.len() as f64;

        Ok(ProcessingResult {
            input_path: input.to_path_buf(),
            output_path: output.to_path_buf(),
            processing_time,
            performance_metrics: PerformanceMetrics {
                input_duration_seconds: audio.duration(),
                processing_time_seconds: processing_time.as_secs_f64(),
                real_time_factor: processing_time.as_secs_f64() / audio.duration(),
                segment_count: segments.len(),
                average_inference_time_ms: avg_inference,
            },
        })
    }

    fn run_inference(&mut self, segments: &[AudioSegment]) -> Result<Vec<(usize, Vec<f32>, u64)>> {
        let segment_size = self.config.segment_size();
        let verbose = self.config.verbose();

        let inputs: Vec<_> = segments.par_iter()
            .map(|seg| seg.mono_data().map(|d| to_onnx_input(d, segment_size)))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ZipEnhancerError::processing("Not mono"))?;

        let mut results = Vec::with_capacity(segments.len());
        for (i, input) in inputs.into_iter().enumerate() {
            if verbose && segments.len() > 10 && i % (segments.len() / 10) == 0 {
                println!("Progress: {}/{}", i, segments.len());
            }

            let result = self.engine.run(vec![input])?;
            if !result.success {
                return Err(ZipEnhancerError::processing(result.error.unwrap_or_default()));
            }

            let data: Vec<f32> = result.first_output()
                .ok_or_else(|| ZipEnhancerError::processing("No output"))?
                .iter().cloned().collect();

            results.push((i, data, result.time_ms));
        }

        Ok(results.into_par_iter()
            .map(|(i, mut data, time)| { apply_agc(&mut data); (i, data, time) })
            .collect())
    }

    pub fn warm_up(&mut self) -> Result<()> {
        if self.config.verbose() { println!("Warming up..."); }
        self.engine.warm_up()?;
        if self.config.verbose() { println!("Warm-up complete"); }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub input_path: std::path::PathBuf,
    pub output_path: std::path::PathBuf,
    pub processing_time: std::time::Duration,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub input_duration_seconds: f64,
    pub processing_time_seconds: f64,
    pub real_time_factor: f64,
    pub segment_count: usize,
    pub average_inference_time_ms: f64,
}

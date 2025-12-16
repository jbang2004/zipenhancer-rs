//! Parallel Audio Processing - Multiple ONNX sessions

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use rayon::prelude::*;

use crate::audio::WavAudio;
use crate::onnx::{OnnxSession, SessionConfig, DynamicTensor};
use crate::processing::{AudioPreprocessor, AudioPostprocessor, PreprocessingConfig, PostprocessingConfig};
use crate::processing::common::{prepare_audio, to_onnx_input, apply_agc, normalize_output, build_audio_segment};
use crate::config::Config;
use crate::error::{ZipEnhancerError, Result};

struct Task { index: usize, tensor: DynamicTensor }
struct Output { index: usize, data: Vec<f32>, time_ms: u64 }

struct Worker {
    tx: Sender<Option<Task>>,
    rx: Receiver<Output>,
    handle: JoinHandle<()>,
}

impl Worker {
    fn new(model_path: PathBuf, threads: i32) -> Result<Self> {
        let (task_tx, task_rx) = channel::<Option<Task>>();
        let (out_tx, out_rx) = channel::<Output>();

        let handle = thread::spawn(move || {
            let config = SessionConfig { intra_op_num_threads: threads };
            let mut session = OnnxSession::new(&model_path, config).expect("Session failed");

            while let Ok(Some(task)) = task_rx.recv() {
                let start = Instant::now();
                let data = session.run(vec![task.tensor])
                    .ok()
                    .and_then(|o| o.into_iter().next())
                    .map(|t| t.into_ndarray().iter().cloned().collect())
                    .unwrap_or_default();

                let _ = out_tx.send(Output { index: task.index, data, time_ms: start.elapsed().as_millis() as u64 });
            }
        });

        Ok(Self { tx: task_tx, rx: out_rx, handle })
    }
}

pub struct ParallelAudioProcessor {
    config: Config,
    workers: Vec<Worker>,
    preprocessor: AudioPreprocessor,
    postprocessor: AudioPostprocessor,
}

impl ParallelAudioProcessor {
    pub fn new(config: Config, num_workers: usize) -> Result<Self> {
        let num_workers = num_workers.max(1);
        let threads_per = (config.inference_threads() / num_workers).max(1) as i32;

        let workers: Result<Vec<_>> = (0..num_workers)
            .map(|_| Worker::new(config.model_path().clone(), threads_per))
            .collect();

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
            workers: workers?,
        })
    }

    pub fn process_file(&mut self, input: &Path, output: &Path) -> Result<ParallelProcessingResult> {
        let start = Instant::now();

        let mut audio = WavAudio::from_file(input)?;
        prepare_audio(&mut audio, &self.config)?;

        let segments = self.preprocessor.preprocess_and_segment(&audio)?;
        let segment_size = self.config.segment_size();

        // Distribute tasks using common::to_onnx_input
        let mut tasks_per_worker = vec![0usize; self.workers.len()];
        for (i, seg) in segments.iter().enumerate() {
            let tensor = to_onnx_input(seg.mono_data().unwrap(), segment_size);
            let worker_idx = i % self.workers.len();
            
            self.workers[worker_idx].tx
                .send(Some(Task { index: i, tensor }))
                .map_err(|_| ZipEnhancerError::processing("Send failed"))?;
            tasks_per_worker[worker_idx] += 1;
        }

        // Collect results
        let mut results: Vec<(usize, Vec<f32>, u64)> = Vec::with_capacity(segments.len());
        for (worker_idx, &count) in tasks_per_worker.iter().enumerate() {
            for _ in 0..count {
                let out = self.workers[worker_idx].rx.recv()
                    .map_err(|_| ZipEnhancerError::processing("Recv failed"))?;
                results.push((out.index, out.data, out.time_ms));
            }
        }
        results.sort_by_key(|(idx, _, _)| *idx);

        // Post-process
        let processed: Vec<_> = results.into_par_iter()
            .map(|(idx, mut data, time)| { apply_agc(&mut data); (idx, data, time) })
            .collect();

        // Reconstruct
        let audio_segments: Vec<_> = processed.iter()
            .map(|(idx, data, _)| build_audio_segment(*idx, data.clone(), &segments[*idx]))
            .collect();

        let mut output_data = self.postprocessor.reconstruct_from_segments(&audio_segments)?;
        normalize_output(&mut output_data, self.config.verbose());

        self.postprocessor.create_wav_audio(output_data)?.save_to_file(output)?;

        let total_time = start.elapsed();
        Ok(ParallelProcessingResult {
            processing_time_secs: total_time.as_secs_f64(),
            segment_count: segments.len(),
            avg_inference_time_ms: processed.iter().map(|(_, _, t)| *t as f64).sum::<f64>() / segments.len() as f64,
            worker_count: self.workers.len(),
            rtf: total_time.as_secs_f64() / audio.duration(),
        })
    }
}

impl Drop for ParallelAudioProcessor {
    fn drop(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.tx.send(None);
            let _ = worker.handle.join();
        }
    }
}

#[derive(Debug)]
pub struct ParallelProcessingResult {
    pub processing_time_secs: f64,
    pub segment_count: usize,
    pub avg_inference_time_ms: f64,
    pub worker_count: usize,
    pub rtf: f64,
}

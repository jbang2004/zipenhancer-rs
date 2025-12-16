//! Audio Postprocessing - Simplified overlap-add reconstruction

use ndarray::Array1;
use crate::audio::{WavAudio, AudioData, AudioFormat, AudioHeader};
use crate::error::{ZipEnhancerError, Result};
use super::AudioSegment;

/// Postprocessing configuration
#[derive(Debug, Clone)]
pub struct PostprocessingConfig {
    pub output_sample_rate: u32,
    pub output_format: AudioFormat,
    pub overlap_ratio: f32,
}

impl Default for PostprocessingConfig {
    fn default() -> Self {
        Self {
            output_sample_rate: 16000,
            output_format: AudioFormat::Int16,
            overlap_ratio: 0.1,
        }
    }
}

/// Audio postprocessor
#[derive(Debug)]
pub struct AudioPostprocessor {
    config: PostprocessingConfig,
}

impl AudioPostprocessor {
    pub fn new(config: PostprocessingConfig) -> Self {
        Self { config }
    }

    /// Reconstruct audio from processed segments using overlap-add
    pub fn reconstruct_from_segments(&mut self, segments: &[AudioSegment]) -> Result<Array1<f32>> {
        if segments.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let segment_size = segments[0].length;
        let overlap_size = (segment_size as f32 * self.config.overlap_ratio) as usize;
        let hop_size = segment_size - overlap_size;

        let total_length = segments.last().map(|s| s.end_sample).unwrap_or(0);
        let mut output = Array1::zeros(total_length);

        let fade = Self::compute_crossfade(overlap_size);

        for (idx, segment) in segments.iter().enumerate() {
            let data = segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Segment must be mono"))?;

            if idx == 0 {
                for (i, &sample) in data.iter().enumerate() {
                    if segment.start_sample + i < output.len() {
                        output[segment.start_sample + i] = sample;
                    }
                }
            } else {
                let overlap_start = idx * hop_size;

                for i in 0..overlap_size {
                    let out_pos = overlap_start + i;
                    if out_pos < output.len() && i < data.len() && i < fade.len() {
                        let fade_out = fade[i];
                        output[out_pos] = output[out_pos] * fade_out + data[i] * (1.0 - fade_out);
                    }
                }

                let non_overlap_start = overlap_start + overlap_size;
                for (i, &sample) in data.iter().skip(overlap_size).enumerate() {
                    if non_overlap_start + i < output.len() {
                        output[non_overlap_start + i] = sample;
                    }
                }
            }
        }

        self.apply_end_fadeout(&mut output, segment_size);
        Ok(output)
    }

    fn compute_crossfade(size: usize) -> Vec<f32> {
        if size == 0 { return vec![]; }
        (0..size).map(|i| {
            let progress = i as f32 / (size - 1).max(1) as f32;
            0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }).collect()
    }

    fn apply_end_fadeout(&self, output: &mut Array1<f32>, segment_size: usize) {
        let fade_size = (segment_size as f32 * 0.15) as usize;
        if fade_size == 0 || output.len() <= fade_size { return; }

        let start = output.len() - fade_size;
        for i in 0..fade_size {
            let progress = i as f32 / fade_size as f32;
            let factor = (1.0 - progress * progress * std::f32::consts::PI / 2.0).cos();
            let factor = if progress > 0.8 {
                factor * (-(progress - 0.8) / 0.2 * 4.0).exp()
            } else { factor };
            output[start + i] *= factor;
        }

        for i in output.len().saturating_sub(5)..output.len() {
            output[i] = 0.0;
        }
    }

    pub fn create_wav_audio(&self, data: Array1<f32>) -> Result<WavAudio> {
        Ok(WavAudio {
            header: AudioHeader::new(self.config.output_sample_rate, 1, self.config.output_format, data.len() as u32),
            data: AudioData::Mono(data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let c = PostprocessingConfig::default();
        assert_eq!(c.output_sample_rate, 16000);
        assert_eq!(c.overlap_ratio, 0.1);
    }

    #[test]
    fn test_crossfade() {
        let fade = AudioPostprocessor::compute_crossfade(5);
        assert_eq!(fade.len(), 5);
        assert!((fade[0] - 1.0).abs() < 0.01);
        assert!(fade[4].abs() < 0.01);
    }

    #[test]
    fn test_create_wav() {
        let processor = AudioPostprocessor::new(PostprocessingConfig::default());
        let data = Array1::from(vec![0.1, 0.2, 0.3]);
        let audio = processor.create_wav_audio(data).unwrap();
        assert_eq!(audio.sample_rate(), 16000);
    }
}

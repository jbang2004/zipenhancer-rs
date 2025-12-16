//! Audio Preprocessing - Simplified segmentation

use ndarray::Array1;
use crate::audio::{WavAudio, AudioData};
use crate::error::{ZipEnhancerError, Result};

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_sample_rate: u32,
    pub segment_size: usize,
    pub overlap_ratio: f32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            segment_size: 16000,
            overlap_ratio: 0.1,
        }
    }
}

/// Audio segment
#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub index: usize,
    pub data: AudioData,
    pub start_sample: usize,
    pub end_sample: usize,
    pub length: usize,
    pub is_complete: bool,
}

impl AudioSegment {
    pub fn new(index: usize, data: AudioData, start: usize, end: usize, is_complete: bool) -> Self {
        Self {
            index,
            data,
            start_sample: start,
            end_sample: end,
            length: end - start,
            is_complete,
        }
    }

    pub fn mono_data(&self) -> Option<&Array1<f32>> {
        match &self.data {
            AudioData::Mono(d) => Some(d),
            AudioData::Stereo(_) => None,
        }
    }
}

/// Audio preprocessor
#[derive(Debug)]
pub struct AudioPreprocessor {
    config: PreprocessingConfig,
}

impl AudioPreprocessor {
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    /// Preprocess and segment audio
    pub fn preprocess_and_segment(&self, audio: &WavAudio) -> Result<Vec<AudioSegment>> {
        if audio.channels() != 1 {
            return Err(ZipEnhancerError::processing("Input must be mono"));
        }
        if audio.sample_rate() != self.config.target_sample_rate {
            return Err(ZipEnhancerError::processing(format!(
                "Sample rate mismatch: {} vs {}", audio.sample_rate(), self.config.target_sample_rate
            )));
        }
        self.segment(audio)
    }

    /// Segment audio into overlapping chunks
    fn segment(&self, audio: &WavAudio) -> Result<Vec<AudioSegment>> {
        let total = audio.data().len();
        let seg_size = self.config.segment_size;
        let hop = (seg_size as f32 * (1.0 - self.config.overlap_ratio)) as usize;

        if seg_size == 0 || hop == 0 {
            return Err(ZipEnhancerError::processing("Invalid segment/hop size"));
        }

        let mut segments = Vec::new();
        let mut idx = 0;

        for start in (0..total).step_by(hop) {
            let end = (start + seg_size).min(total);
            let is_complete = (end - start) == seg_size;

            let seg_data = match audio.data() {
                AudioData::Mono(d) => AudioData::Mono(d.slice(ndarray::s![start..end]).to_owned()),
                AudioData::Stereo(d) => AudioData::Stereo(d.slice(ndarray::s![start..end, ..]).to_owned()),
            };

            segments.push(AudioSegment::new(idx, seg_data, start, end, is_complete));
            idx += 1;
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioFormat;

    #[test]
    fn test_config_default() {
        let c = PreprocessingConfig::default();
        assert_eq!(c.target_sample_rate, 16000);
        assert_eq!(c.segment_size, 16000);
        assert_eq!(c.overlap_ratio, 0.1);
    }

    #[test]
    fn test_segment_creation() {
        let data = AudioData::Mono(Array1::from(vec![0.1, 0.2, 0.3]));
        let seg = AudioSegment::new(0, data, 0, 3, true);
        assert_eq!(seg.length, 3);
        assert!(seg.mono_data().is_some());
    }

    #[test]
    fn test_segmentation() {
        let config = PreprocessingConfig {
            segment_size: 4,
            overlap_ratio: 0.5,
            target_sample_rate: 16000,
        };
        let preprocessor = AudioPreprocessor::new(config);
        
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);
        
        let segments = preprocessor.segment(&audio).unwrap();
        assert!(segments.len() >= 2);
        assert_eq!(segments[0].length, 4);
    }
}

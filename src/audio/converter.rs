//! Audio Format Converter - Essential functions only

use ndarray::{Array1, Array2, ArrayView1};
use crate::audio::{AudioData, WavAudio};
use crate::error::{ZipEnhancerError, Result};

pub struct AudioConverter;

impl AudioConverter {
    /// Convert sample rate using linear interpolation
    pub fn convert_sample_rate(audio: &WavAudio, target_sample_rate: u32) -> Result<WavAudio> {
        if audio.sample_rate() == target_sample_rate {
            return Ok(audio.clone());
        }

        let ratio = target_sample_rate as f64 / audio.sample_rate() as f64;
        let new_length = (audio.data().len() as f64 * ratio) as usize;

        let new_data = match audio.data() {
            AudioData::Mono(data) => {
                AudioData::Mono(Self::resample_mono(data.view(), new_length, ratio)?)
            }
            AudioData::Stereo(data) => {
                let new_left = Self::resample_mono(data.column(0), new_length, ratio)?;
                let new_right = Self::resample_mono(data.column(1), new_length, ratio)?;
                let mut stereo = Array2::zeros((new_length, 2));
                stereo.column_mut(0).assign(&new_left);
                stereo.column_mut(1).assign(&new_right);
                AudioData::Stereo(stereo)
            }
        };

        let mut new_audio = audio.clone();
        new_audio.header.sample_rate = target_sample_rate;
        new_audio.header.total_samples = new_length as u32;
        new_audio.header.duration = new_length as f64 / target_sample_rate as f64;
        new_audio.data = new_data;
        Ok(new_audio)
    }

    fn resample_mono(data: ArrayView1<f32>, new_length: usize, ratio: f64) -> Result<Array1<f32>> {
        if data.is_empty() {
            return Err(ZipEnhancerError::audio("Input data is empty"));
        }

        let old_length = data.len();
        let mut new_data = Array1::zeros(new_length);

        for i in 0..new_length {
            let old_pos = i as f64 / ratio;
            let old_index = old_pos.floor() as usize;
            let fraction = old_pos - old_index as f64;

            new_data[i] = if old_index >= old_length - 1 {
                data[old_length - 1]
            } else {
                data[old_index] + (data[old_index + 1] - data[old_index]) * fraction as f32
            };
        }

        Ok(new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioFormat;

    #[test]
    fn test_resample_same_rate() {
        let data = Array1::from(vec![0.1, 0.2, 0.3]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);
        let result = AudioConverter::convert_sample_rate(&audio, 16000).unwrap();
        assert_eq!(result.sample_rate(), 16000);
    }

    #[test]
    fn test_resample_upsample() {
        let data = Array1::from(vec![0.0, 1.0]);
        let audio = WavAudio::new_mono(8000, data, AudioFormat::Float32);
        let result = AudioConverter::convert_sample_rate(&audio, 16000).unwrap();
        assert_eq!(result.sample_rate(), 16000);
        assert_eq!(result.data().len(), 4);
    }
}

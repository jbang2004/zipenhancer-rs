//! WAV audio file processing

use std::path::Path;
use std::fs::File;
use hound::{WavReader, WavWriter, SampleFormat};
use ndarray::{Array1, Array2};
use crate::error::{ZipEnhancerError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Int16,
    Float32,
}

impl AudioFormat {
    pub fn bytes_per_sample(&self) -> u16 {
        match self { AudioFormat::Int16 => 2, AudioFormat::Float32 => 4 }
    }

    pub fn to_sample_format(self) -> SampleFormat {
        match self { AudioFormat::Int16 => SampleFormat::Int, AudioFormat::Float32 => SampleFormat::Float }
    }
}

#[derive(Debug, Clone)]
pub struct AudioHeader {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: AudioFormat,
    pub total_samples: u32,
    pub bits_per_sample: u16,
    pub duration: f64,
}

impl AudioHeader {
    pub fn new(sample_rate: u32, channels: u16, format: AudioFormat, total_samples: u32) -> Self {
        Self {
            sample_rate,
            channels,
            format,
            total_samples,
            bits_per_sample: format.bytes_per_sample() * 8,
            duration: total_samples as f64 / sample_rate as f64,
        }
    }

    pub fn to_wav_spec(&self) -> hound::WavSpec {
        hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.format.to_sample_format(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AudioData {
    Mono(Array1<f32>),
    Stereo(Array2<f32>),
}

impl AudioData {
    pub fn len(&self) -> usize {
        match self { AudioData::Mono(d) => d.len(), AudioData::Stereo(d) => d.nrows() }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn channels(&self) -> u16 {
        match self { AudioData::Mono(_) => 1, AudioData::Stereo(_) => 2 }
    }

    pub fn to_mono(&self) -> Array1<f32> {
        match self {
            AudioData::Mono(d) => d.clone(),
            AudioData::Stereo(d) => d.mean_axis(ndarray::Axis(1)).expect("Empty stereo"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavAudio {
    pub header: AudioHeader,
    pub data: AudioData,
}

impl WavAudio {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| ZipEnhancerError::audio(format!("Cannot open {}: {}", path.display(), e)))?;
        
        let mut reader = WavReader::new(file)
            .map_err(|e| ZipEnhancerError::audio(format!("WAV read error: {}", e)))?;

        let spec = reader.spec();
        if spec.sample_rate == 0 { return Err(ZipEnhancerError::audio("Invalid sample rate")); }
        if spec.channels == 0 || spec.channels > 2 {
            return Err(ZipEnhancerError::audio("Only mono/stereo supported"));
        }

        let format = match spec.bits_per_sample {
            16 => AudioFormat::Int16,
            32 if spec.sample_format == SampleFormat::Float => AudioFormat::Float32,
            _ => return Err(ZipEnhancerError::audio(format!("Unsupported format: {} bit", spec.bits_per_sample))),
        };

        let samples: Vec<f32> = match spec.bits_per_sample {
            16 => reader.samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32767.0).map_err(|e| ZipEnhancerError::audio(e.to_string())))
                .collect::<Result<Vec<_>>>()?,
            32 => reader.samples::<f32>()
                .map(|s| s.map_err(|e| ZipEnhancerError::audio(e.to_string())))
                .collect::<Result<Vec<_>>>()?,
            _ => unreachable!(),
        };

        let data = if spec.channels == 1 {
            AudioData::Mono(Array1::from(samples))
        } else {
            let len = samples.len() / 2;
            let mut stereo = Array2::zeros((len, 2));
            for (i, chunk) in samples.chunks_exact(2).enumerate() {
                stereo[[i, 0]] = chunk[0];
                stereo[[i, 1]] = chunk[1];
            }
            AudioData::Stereo(stereo)
        };

        Ok(WavAudio {
            header: AudioHeader {
                sample_rate: spec.sample_rate,
                channels: spec.channels,
                format,
                total_samples: reader.len() as u32,
                bits_per_sample: spec.bits_per_sample,
                duration: reader.len() as f64 / spec.sample_rate as f64,
            },
            data,
        })
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ZipEnhancerError::audio(e.to_string()))?;
        }

        let file = File::create(path)
            .map_err(|e| ZipEnhancerError::audio(format!("Cannot create {}: {}", path.display(), e)))?;
        
        let spec = self.header.to_wav_spec();
        let mut writer = WavWriter::new(file, spec)
            .map_err(|e| ZipEnhancerError::audio(e.to_string()))?;

        let write_sample = |w: &mut WavWriter<_>, s: f32| -> Result<()> {
            let clamped = s.clamp(-1.0, 1.0);
            if spec.sample_format == SampleFormat::Float {
                w.write_sample(clamped)
            } else {
                w.write_sample((clamped * 32767.0) as i16)
            }.map_err(|e| ZipEnhancerError::audio(e.to_string()))
        };

        match &self.data {
            AudioData::Mono(d) => { for &s in d.iter() { write_sample(&mut writer, s)?; } }
            AudioData::Stereo(d) => { for row in d.rows() { for &s in row { write_sample(&mut writer, s)?; } } }
        }

        writer.finalize().map_err(|e| ZipEnhancerError::audio(e.to_string()))
    }

    pub fn new_mono(sample_rate: u32, data: Array1<f32>, format: AudioFormat) -> Self {
        WavAudio {
            header: AudioHeader::new(sample_rate, 1, format, data.len() as u32),
            data: AudioData::Mono(data),
        }
    }

    pub fn data(&self) -> &AudioData { &self.data }
    pub fn data_mut(&mut self) -> &mut AudioData { &mut self.data }
    pub fn sample_rate(&self) -> u32 { self.header.sample_rate }
    pub fn channels(&self) -> u16 { self.header.channels }
    pub fn total_samples(&self) -> u32 { self.header.total_samples }
    pub fn duration(&self) -> f64 { self.header.duration }
    pub fn format(&self) -> AudioFormat { self.header.format }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_audio_format() {
        assert_eq!(AudioFormat::Int16.bytes_per_sample(), 2);
        assert_eq!(AudioFormat::Float32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_wav_creation() {
        let data = Array1::from(vec![0.1, 0.2, 0.3]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.total_samples(), 3);
    }

    #[test]
    fn test_wav_roundtrip() {
        let data = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let original = WavAudio::new_mono(16000, data, AudioFormat::Float32);

        let temp = NamedTempFile::new().unwrap();
        original.save_to_file(temp.path()).unwrap();
        let loaded = WavAudio::from_file(temp.path()).unwrap();

        assert_eq!(loaded.sample_rate(), original.sample_rate());
        assert_eq!(loaded.total_samples(), original.total_samples());
    }

    #[test]
    fn test_to_mono() {
        let mono = AudioData::Mono(Array1::from(vec![0.5, 0.5]));
        assert_eq!(mono.to_mono().len(), 2);

        let stereo = AudioData::Stereo(Array2::from(vec![[0.2, 0.4], [0.6, 0.8]]));
        let mono_from_stereo = stereo.to_mono();
        assert_eq!(mono_from_stereo.len(), 2);
        assert!((mono_from_stereo[0] - 0.3).abs() < 1e-6);
    }
}

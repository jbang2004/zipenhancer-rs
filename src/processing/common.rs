//! Common processing utilities shared between serial and parallel processors

use crate::audio::{WavAudio, AudioConverter, AudioData};
use crate::onnx::DynamicTensor;
use crate::config::Config;
use crate::error::Result;
use super::AudioSegment;

/// Prepare audio: convert to mono and resample if needed
pub fn prepare_audio(audio: &mut WavAudio, config: &Config) -> Result<()> {
    if audio.channels() > 1 {
        *audio.data_mut() = AudioData::Mono(audio.data().to_mono());
        audio.header.channels = 1;
    }
    if audio.sample_rate() != config.sample_rate() {
        *audio = AudioConverter::convert_sample_rate(audio, config.sample_rate())?;
    }
    Ok(())
}

/// Convert f32 audio data to i16 ONNX input tensor
pub fn to_onnx_input(data: &ndarray::Array1<f32>, target_len: usize) -> DynamicTensor {
    let vec: Vec<f32> = if data.len() >= target_len {
        data.slice(ndarray::s![..target_len]).to_vec()
    } else {
        let mut v = vec![0.0f32; target_len];
        v[..data.len()].copy_from_slice(data.as_slice().unwrap());
        v
    };
    let i16_data: Vec<i16> = vec.iter().map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
    DynamicTensor::new_i16(i16_data, vec![1, 1, target_len as i64])
}

/// Apply automatic gain control to processed segment
pub fn apply_agc(data: &mut Vec<f32>) {
    for s in data.iter_mut() {
        *s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
    }
    let max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max < 0.3 && max > 0.001 {
        let gain = (1.0 / max).clamp(3.0, 10.0);
        for s in data.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}

/// Normalize final output audio
pub fn normalize_output(data: &mut ndarray::Array1<f32>, verbose: bool) {
    if data.is_empty() { return; }

    let (sum_sq, peak) = data.iter()
        .map(|&x| (x * x, x.abs()))
        .fold((0.0f32, 0.0f32), |(s, p), (s2, p2)| (s + s2, p.max(p2)));

    let rms = (sum_sq / data.len() as f32).sqrt();

    if rms < 0.1 && peak > 0.001 {
        let rms_gain = 0.2 / rms;
        let peak_gain = if peak > 0.0 { 0.95 / peak } else { 1.0 };
        let gain = rms_gain.min(peak_gain).clamp(1.5, 8.0);

        if verbose {
            println!("Normalization: RMS={:.4}, Peak={:.4}, gain={:.2}", rms, peak, gain);
        }

        for s in data.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}

/// Build AudioSegment from processed data
pub fn build_audio_segment(idx: usize, data: Vec<f32>, original: &AudioSegment) -> AudioSegment {
    AudioSegment {
        index: original.index,
        data: AudioData::Mono(ndarray::Array1::from(data)),
        start_sample: original.start_sample,
        end_sample: original.end_sample,
        length: original.length,
        is_complete: original.is_complete,
    }
}

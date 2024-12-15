use rustfft::FftNum;

use crate::{complex::Complex, fft::{istft, stft}, sample::AudioSample, signal::{SpectrumSignal, TimeDomainSignal}};

pub fn phase_vocoder<T, F>(
    signal: TimeDomainSignal<T>,
    scale_factor: f32,
    window_size: usize,
    hop_length: usize,
    window_fn: F,
) -> TimeDomainSignal<T>
where
    T: AudioSample + FftNum,
    F: Fn(f32, usize) -> T
{
    let synth_hop_length = hop_length;
    // let synth_hop_length = (hop_length as f32 * scale_factor) as usize;
    let new_len = (signal.len() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;

    let stft = stft(&signal, window_size, hop_length, window_fn);
    let interp_stft = SpectrumSignal::from_elem((new_len, window_size), Complex { re: T::zero(), im: T::zero() });

    istft(stft, window_size, hop_length, new_len)
}

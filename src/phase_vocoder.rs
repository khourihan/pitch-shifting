use rustfft::FftNum;

use crate::{fft::{istft, stft}, sample::AudioSample, signal::TimeDomainSignal};

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
    let new_len = (signal.len() as f32 / hop_length as f32).ceil() as usize * hop_length + window_size;
    let stft = stft(&signal, window_size, hop_length, window_fn);
    // let interp_stft = SpectrumSignal::default();

    let istft = istft(stft, window_size, hop_length, new_len);

    istft
}

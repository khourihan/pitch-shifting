use num_traits::Zero;
use rustfft::{FftNum, FftPlanner, num_complex::Complex as FftComplex};

use crate::{complex::Complex, sample::{AudioSample, SamplesRef}, signal::{SpectrumSignal, TimeDomainSignal}, windows::build_window};

/// Compute the Fast Fourier Transform of the given signal in the time domain.
pub fn fft<T>(signal: &[T]) -> Vec<Complex<T>>
where
    T: FftNum + AudioSample
{
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<_> = signal.iter().map(|&s| FftComplex { re: s, im: T::zero() }).collect();
    fft.process(&mut buffer);

    buffer.into_iter().map(|c| Complex { re: c.re, im: c.im }).collect()
}

/// Compute the inverse Fast Fourier Transform of the given signal in the frequency domain.
pub fn ifft<T>(signal: &[Complex<T>]) -> Vec<T>
where
    T: FftNum + AudioSample
{
    let mut planner = FftPlanner::<T>::new();
    let ifft = planner.plan_fft_inverse(signal.len());

    let mut buffer: Vec<_> = signal.iter().map(|&s| FftComplex { re: s.re, im: s.im }).collect();
    ifft.process(&mut buffer);

    buffer.into_iter().map(|c| c.re).collect()
}

/// Compute the short-time fourier transform of the given signal in the time domain.
pub fn stft<T, F>(
    signal: &TimeDomainSignal<T>,
    window_size: usize,
    hop_length: usize,
    window_fn: F,
) -> SpectrumSignal<T>
where
    T: FftNum + AudioSample,
    F: Fn(f32, usize) -> T,
{
    let mut spectrum_samples = Vec::new();

    let window = build_window(window_fn, window_size);

    for i in (0..signal.num_samples()).step_by(hop_length) {
        let len = window_size.min(signal.num_samples() - i);

        let window_signal = signal.window(i..i + len) * SamplesRef(&window[..len]);
        let spectrum = fft(&window_signal.0);

        spectrum_samples.push(spectrum);
    }

    SpectrumSignal { samples: spectrum_samples }
}

/// Compute the inverse short-time fourier transform of the given signal.
pub fn istft<T>(
    signal: SpectrumSignal<T>,
    window_size: usize,
    hop_length: usize,
    num_samples: usize,
) -> Vec<T>
where
    T: FftNum + AudioSample
{
    let mut samples = vec![T::zero(); num_samples];

    for (i, spectrum) in signal.samples.into_iter().enumerate() {
        let window_samples = ifft(&spectrum);
        let start = i * hop_length;
        
        samples[start..start + window_size]
            .iter_mut()
            .zip(window_samples.into_iter())
            .for_each(|(s, win_s)| *s = *s + win_s);
    }

    samples
}
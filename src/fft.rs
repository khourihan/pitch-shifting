use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use rustfft::{FftNum, FftPlanner, num_complex::Complex as FftComplex};

use crate::{complex::Complex, sample::AudioSample, signal::{SpectrumSignal, TimeDomainSignal}, windows::build_window};

/// Compute the Fast Fourier Transform of the given signal in the time domain.
pub fn fft<T>(signal: ArrayView1<T>) -> Array1<Complex<T>>
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
pub fn ifft<T>(signal: ArrayView1<Complex<T>>) -> Array1<T>
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
    let len = signal.len() / hop_length;
    let mut spectrum_samples = Array2::from_elem((len, window_size), Complex { re: T::zero(), im: T::zero() });

    let window = build_window(window_fn, window_size);

    for i in (0..signal.len()).step_by(hop_length) {
        let len = window_size.min(signal.len() - i);

        let window_signal = &signal.slice(s![i..i + len]) * &window.slice(s![..len]);
        let spectrum = fft(window_signal.view());

        let mut window_spectrum = spectrum_samples.slice_mut(s![i, ..]);
        window_spectrum += &spectrum;
    }

    spectrum_samples
}

/// Compute the inverse short-time fourier transform of the given signal.
pub fn istft<T>(
    signal: SpectrumSignal<T>,
    window_size: usize,
    hop_length: usize,
    num_samples: usize,
) -> Array1<T>
where
    T: FftNum + AudioSample
{
    let mut samples = Array1::from_elem(num_samples, T::zero());

    for (i, spectrum) in signal.axis_iter(Axis(0)).enumerate() {
        let window_samples = ifft(spectrum);
        let start = i * hop_length;
        
        samples.slice_mut(s![start..start + window_size])
            .iter_mut()
            .zip(window_samples.into_iter())
            .for_each(|(s, win_s)| *s += win_s);
    }

    samples
}

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
pub fn stft<T>(
    signal: &TimeDomainSignal<T>,
    window_size: usize,
    hop_length: usize,
    window: &Array1<T>,
) -> SpectrumSignal<T>
where
    T: FftNum + AudioSample,
{
    let len = signal.len().div_ceil(hop_length);
    let mut spectrum_samples = Array2::from_elem((len, window_size), Complex { re: T::zero(), im: T::zero() });

    for i in (0..signal.len()).step_by(hop_length) {
        let len = window_size.min(signal.len() - i);
        let index = i / hop_length;

        let window_signal = &signal.slice(s![i..i + len]) * &window.slice(s![..len]);
        let mut spectrum = fft(window_signal.view());

        if spectrum.len() < window_size {
            let _ = spectrum.append(Axis(0), Array1::from_elem(window_size - spectrum.len(), Complex { re: T::zero(), im: T::zero() }).view());
        }

        let mut window_spectrum = spectrum_samples.slice_mut(s![index, ..]);
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
    window: &Array1<T>,
) -> Array1<T>
where
    T: FftNum + AudioSample,
{
    let mut samples = Array1::from_elem(num_samples, T::zero());
    let mut weights = Array1::from_elem(num_samples, T::zero());

    for (i, spectrum) in signal.outer_iter().enumerate() {
        let mut window_samples = ifft(spectrum);
        let start = i * hop_length;

        window_samples *= &window.slice(s![..window_samples.len()]);

        let mut win_weights = weights.slice_mut(s![start..start + window_size]);
        win_weights += &window.slice(s![..window_samples.len()]);
        
        let mut win_samples = samples.slice_mut(s![start..start + window_size]);
        win_samples += &window_samples;
    }

    samples / weights.mapv(|v| if v == T::zero() { T::one() } else { v })
}

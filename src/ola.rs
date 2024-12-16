use ndarray::s;

use crate::{sample::AudioSample, signal::TimeDomainSignal, windows::build_window};

/// Overlap Add
pub fn ola<T, F>(
    signal: TimeDomainSignal<T>,
    scale_factor: f32,
    window_size: usize,
    hop_length: usize,
    window_fn: F,
) -> TimeDomainSignal<T>
where
    T: AudioSample,
    F: Fn(f32, usize) -> T, 
{
    let synth_hop_length = (hop_length as f32 * scale_factor) as usize;
    let synth_len = (signal.len() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut synth_signal = TimeDomainSignal::from_elem(synth_len, T::zero());
    let mut synth_weights = TimeDomainSignal::from_elem(synth_len, T::zero());

    let window = build_window(window_fn, window_size);

    for i in (0..signal.len()).step_by(hop_length) {
        let len = window_size.min(signal.len() - i);
        let index = (i as f32 * scale_factor) as usize;

        let window_f = window.slice(s![..len]);
        let window_value = &signal.slice(s![i..i + len]) * &window_f;

        let mut window_signal = synth_signal.slice_mut(s![index..index + len]);
        window_signal += &window_value;

        let mut window_weights = synth_weights.slice_mut(s![index..index + len]);
        window_weights += &window.slice(s![..len]);
    }

    synth_signal / synth_weights.mapv(|v| if v == T::zero() { T::one() } else { v })
}

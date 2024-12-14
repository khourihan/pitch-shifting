use crate::{sample::{AudioSample, SamplesRef}, signal::time_domain::TimeDomainSignal, windows::build_window};

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
    F: Fn(f32, usize) -> T
{
    let synth_hop_length = (hop_length as f32 * scale_factor) as usize;
    let new_len = (signal.num_samples() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut new_signal = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());
    let mut new_weights = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());

    let window = build_window(window_fn, window_size);

    for i in (0..signal.num_samples()).step_by(hop_length) {
        let len = window_size.min(signal.num_samples() - i);
        let index = (i as f32 * scale_factor) as usize;

        let mut window_signal = new_signal.window_mut(index..index + len);
        window_signal += signal.window(i..i + len) * SamplesRef(&window[..len]);

        let mut window_weights = new_weights.window_mut(index..index + len);
        window_weights += SamplesRef(&window[..len]);
    }

    new_signal.samples_mut()
        .zip(new_weights.into_samples())
        .for_each(|(s, w)| {
            if w != T::zero() {
                *s = *s / w;
            }
        });

    new_signal
}

use crate::{sample::AudioSample, signal::time_domain::TimeDomainSignal};

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
    let window_width = window_size as f32 / 2.0;
    let synth_hop_length = (hop_length as f32 * scale_factor) as usize;
    let new_len = (signal.num_samples() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut new_signal = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());
    let mut new_weights = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());

    for i in (0..signal.num_samples()).step_by(hop_length) {
        let len = window_size.min(signal.num_samples() - i);
        let index = (i as f32 * scale_factor) as usize;

        new_signal.window_mut(index..index + len)
            .iter_mut()
            .zip(new_weights.window_mut(index..index + len).iter_mut())
            .enumerate()
            .for_each(|(j, (s, w))| {
                let scale = window_fn(j as f32 - window_width, window_size);
                *w = *w + scale;
                *s = *s + signal[i + j] * scale;
            });
    }

    new_signal.samples_mut()
        .zip(new_weights.into_samples())
        .for_each(|(s, w)| {
            if w != T::zero() {
                *s = *s / w;
            } else {
                *s = T::zero();
            }
        });

    new_signal
}

use crate::{sample::{AudioSample, SamplesRef}, signal::TimeDomainSignal, windows::build_window};

/// Synchronized Overlap Add
pub fn sola<T, F>(
    signal: TimeDomainSignal<T>,
    scale_factor: f32,
    window_size: usize,
    hop_length: usize,
    window_fn: F,
) -> TimeDomainSignal<T>
where
    T: AudioSample + std::iter::Sum,
    F: Fn(f32, usize) -> T
{
    let synth_hop_length = (hop_length as f32 * scale_factor) as usize;
    let new_len = (signal.num_samples() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut new_signal = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());
    let mut norm_new_signal = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());
    let mut new_weights = TimeDomainSignal::from_zeros(new_len, signal.sample_rate());

    let window_f = build_window(window_fn, window_size);

    let mut last_index = 0;
    for i in (0..signal.num_samples()).step_by(hop_length) {
        let len = window_size.min(signal.num_samples() - i);
        let mut index = (i as f32 * scale_factor) as usize;

        let window = signal.window(i..i + len) * SamplesRef(&window_f[..len]);
        
        if index > 0 {
            index = index.saturating_sub(window_size / 10);
            let mut overlap = (window_size.saturating_sub(index) + last_index)
                .min(new_len.saturating_sub(index))
                .min(window.0.len());

            let mut min_area = T::max_value();
            let idx = index;
            for j in idx..(idx + window_size / 5).min(new_len - len).min(idx + overlap) {
                let area = -(SamplesRef(&window[..overlap]) * norm_new_signal.window(j..j + overlap)).into_iter().sum::<T>();

                if j >= new_len - len {
                    dbg!(j, new_len - len);
                }

                if area < min_area {
                    min_area = area;
                    index = j;
                }

                overlap -= 1;
                if overlap == 0 {
                    break;
                }
            }
        }

        let mut window_signal = new_signal.window_mut(index..index + len);
        window_signal += window;

        let mut window_weights = new_weights.window_mut(index..index + len);
        window_weights += SamplesRef(&window_f[..len]);

        norm_new_signal.window_mut(index..index + len)
            .into_iter()
            .zip(window_signal.0.iter())
            .zip(window_weights.0.iter())
            .for_each(|((norm, &s), &w)| {
                if w != T::zero() {
                    *norm = s / w;
                } else {
                    *norm = s;
                }
            });

        last_index = index;
    }

    norm_new_signal
}

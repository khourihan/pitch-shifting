use ndarray::s;

use crate::{sample::AudioSample, signal::TimeDomainSignal, windows::build_window};

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
    let synth_len = (signal.len() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut synth_signal = TimeDomainSignal::from_elem(synth_len, T::zero());
    let mut synth_norm_signal = TimeDomainSignal::from_elem(synth_len, T::zero());
    let mut synth_weights = TimeDomainSignal::from_elem(synth_len, T::zero());

    let window_f = build_window(window_fn, window_size);

    let mut last_index = 0;
    for i in (0..signal.len()).step_by(hop_length) {
        let len = window_size.min(signal.len() - i);
        let mut index = (i as f32 * scale_factor) as usize;

        let window = &signal.slice(s![i..i + len]) * &window_f.slice(s![..len]);
        
        if index > 0 {
            index = index.saturating_sub(window_size / 10);
            let mut overlap = (window_size.saturating_sub(index) + last_index)
                .min(synth_len.saturating_sub(index))
                .min(window.len());

            let mut min_area = T::max_value();
            let idx = index;
            for j in idx..(idx + window_size / 5).min(synth_len - len).min(idx + overlap) {
                let area = -(&window.slice(s![..overlap]) * &synth_norm_signal.slice(s![j..j + overlap])).sum();

                if j >= synth_len - len {
                    dbg!(j, synth_len - len);
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

        let mut window_signal = synth_signal.slice_mut(s![index..index + len]);
        window_signal += &window;

        let mut window_weights = synth_weights.slice_mut(s![index..index + len]);
        window_weights += &window_f.slice(s![..len]);

        synth_norm_signal.slice_mut(s![index..index + len])
            .assign(&(&synth_signal.slice(s![index..index + len]) / window_weights.mapv(|v| if v == T::zero() { T::one() } else { v })));

        last_index = index;
    }

    synth_norm_signal
}

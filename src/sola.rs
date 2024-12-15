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
    let new_len = (signal.len() as f32 / hop_length as f32).ceil() as usize * synth_hop_length + window_size;
    let mut new_signal = TimeDomainSignal::from_elem(new_len, T::zero());
    let mut norm_new_signal = TimeDomainSignal::from_elem(new_len, T::zero());
    let mut new_weights = TimeDomainSignal::from_elem(new_len, T::zero());

    let window_f = build_window(window_fn, window_size);

    let mut last_index = 0;
    for i in (0..signal.len()).step_by(hop_length) {
        let len = window_size.min(signal.len() - i);
        let mut index = (i as f32 * scale_factor) as usize;

        let window = &signal.slice(s![i..i + len]) * &window_f.slice(s![..len]);
        
        if index > 0 {
            index = index.saturating_sub(window_size / 10);
            let mut overlap = (window_size.saturating_sub(index) + last_index)
                .min(new_len.saturating_sub(index))
                .min(window.len());

            let mut min_area = T::max_value();
            let idx = index;
            for j in idx..(idx + window_size / 5).min(new_len - len).min(idx + overlap) {
                let area = -(&window.slice(s![..overlap]) * &norm_new_signal.slice(s![j..j + overlap])).sum();

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

        let mut window_signal = new_signal.slice_mut(s![index..index + len]);
        window_signal += &window;

        let mut window_weights = new_weights.slice_mut(s![index..index + len]);
        window_weights += &window_f.slice(s![..len]);

        norm_new_signal.slice_mut(s![index..index + len])
            .assign(&(&new_signal.slice(s![index..index + len]) / window_weights.mapv(|v| if v == T::zero() { T::one() } else { v })));

        last_index = index;
    }

    norm_new_signal
}

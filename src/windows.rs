use crate::sample::{AudioSample, ConvertSample, Samples};

pub fn build_window<T, F>(f: F, window_size: usize) -> Samples<T>
where 
    T: AudioSample,
    F: Fn(f32, usize) -> T
{
    let width = window_size as f32 / 2.0;
    let mut i = -width;

    std::iter::from_fn(move || {
        if i > width {
            None
        } else {
            let v = f(i, window_size);
            i += 1.0;
            Some(v)
        }
    }).collect()
}

pub fn hann_window<T>(x: f32, window_size: usize) -> T
where
    T: AudioSample,
    f32: ConvertSample<T>,
{
    let inv_win_size = 1.0 / window_size as f32;
    let cosx = f32::cos(std::f32::consts::PI * x * inv_win_size);
    (inv_win_size * cosx * cosx).convert_sample()
}

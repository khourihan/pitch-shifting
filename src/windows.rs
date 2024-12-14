use crate::sample::{AudioSample, ConvertSample};

pub fn hann_window<T>(x: f32, window_size: usize) -> T
where
    T: AudioSample,
    f32: ConvertSample<T>,
{
    let inv_win_size = 1.0 / window_size as f32;
    let cosx = f32::cos(std::f32::consts::PI * x * inv_win_size);
    (inv_win_size * cosx * cosx).convert_sample()
}

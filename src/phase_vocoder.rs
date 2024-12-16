use ndarray::{s, Array2, Axis, Zip};
use rustfft::FftNum;

use crate::{complex::Complex, fft::{istft, stft}, sample::AudioSample, signal::TimeDomainSignal, windows::build_window};

pub fn phase_vocoder<F>(
    signal: TimeDomainSignal<f32>,
    scale_factor: f32,
    window_size: usize,
    hop_length: usize,
    transience_cutoff: f32,
    window_fn: F,
) -> TimeDomainSignal<f32>
where
    f32: AudioSample + FftNum,
    F: Fn(f32, usize) -> f32
{
    let bins = signal.len().div_ceil(hop_length);
    let new_bins = (bins as f32 * scale_factor).ceil() as usize;

    let window = build_window(window_fn, window_size);
    let stft = stft(&signal, window_size, hop_length, &window);
    let mags = stft.mapv(|v| (v.re * v.re + v.im * v.im).sqrt());
    let phases = stft.mapv(|v| f32::atan2(v.im, v.re));

    let mut new_mags = Array2::from_elem((new_bins, window_size), 0f32);
    let mut new_phases = Array2::from_elem((new_bins, window_size), 0f32);
    let mut new_phase_diffs = Array2::from_elem((new_bins, window_size), 0f32);

    for (i, mut new_mags_win) in new_mags.outer_iter_mut().enumerate() {
        let index = i as f32 / scale_factor;
        let i0 = index.floor() as usize;
        let i1 = index.ceil() as usize;
        let d0 = (index - index.floor()).abs();
        let mut d1 = (index - index.ceil()).abs();

        if i0 == i1 {
            d1 = 1.0;
        }

        let mags_win_0 = mags.index_axis(Axis(0), i0.min(bins - 1));
        new_mags_win += &mags_win_0.mapv(|v| v * (1.0 - d0));

        let mags_win_1 = mags.index_axis(Axis(0), i1.min(bins - 1));
        new_mags_win += &mags_win_1.mapv(|v| v * (1.0 - d1));
    }

    let phase_dim = phases.raw_dim();
    let mut phase_shifted = Array2::from_elem((1, phase_dim[1]), 0f32);
    let _ = phase_shifted.append(Axis(0), phases.slice(s![..-1, ..]));
    let mut phase_diffs = &phases - phase_shifted; 
    phase_diffs.map_inplace(|v| *v = v.rem_euclid(std::f32::consts::TAU));

    for (i, mut new_phase_win) in new_phase_diffs.outer_iter_mut().enumerate() {
        let index = i as f32 / scale_factor;
        let i0 = index.floor() as usize;
        let i1 = index.ceil() as usize;
        let d0 = (index - index.floor()).abs();
        let mut d1 = (index - index.ceil()).abs();

        if i0 == i1 {
            d1 = 1.0;
        }

        let phase_win_0 = phase_diffs.index_axis(Axis(0), i0.min(bins - 1));
        new_phase_win += &phase_win_0.mapv(|v| v * (1.0 - d0));

        let phase_win_1 = phase_diffs.index_axis(Axis(0), i1.min(bins - 1));
        new_phase_win += &phase_win_1.mapv(|v| v * (1.0 - d1));
    }

    let mut unshifted_phases = Array2::from_elem((new_bins, window_size), 0f32); 

    for (i, mut phases_win) in unshifted_phases.outer_iter_mut().enumerate() {
        let index = (i as f32 / scale_factor).round() as usize;
        phases_win.assign(&phases.index_axis(Axis(0), index.min(bins - 1)));
    }

    new_phases.slice_mut(s![0, ..]).assign(&new_phase_diffs.slice(s![0, ..]));

    for t in 1..new_bins {
        let time_phase = &new_phases.slice(s![t - 1, ..]) + &new_phase_diffs.slice(s![t, ..]);
        let freq_phase = &unshifted_phases.slice(s![t, ..]);

        let mag0 = new_mags.slice(s![t, ..]);
        let mag1 = new_mags.slice(s![t - 1, ..]);
        let mut transient = (&mag0 - &mag1) / (&mag0 + &mag1);
        transient.mapv_inplace(|v| if v >= transience_cutoff { 1.0 } else { 0.0 });

        let mut new_phase = freq_phase * &transient + time_phase * (1.0 - transient);
        new_phase.map_inplace(|v| *v = v.rem_euclid(std::f32::consts::TAU));
        new_phases.slice_mut(s![t, ..])
            .assign(&new_phase);
    }

    let new_stft = Zip::from(&new_mags).and(&new_phases).map_collect(|&mag, &phase| {
        Complex {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    });

    istft(new_stft, window_size, hop_length, new_bins * hop_length + window_size, &window)
}

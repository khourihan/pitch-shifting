use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Axis, Zip};
use num_complex::{Complex, ComplexFloat};
use rustfft::FftNum;

use crate::{fft::{istft, stft}, sample::AudioSample, signal::TimeDomainSignal, windows::build_window};

pub fn phase_vocoder<F>(
    signal: TimeDomainSignal<f32>,
    scale_factor: f32,
    window_size: usize,
    hop_length: usize,
    transient_cutoff: f32,
    window_fn: F,
) -> TimeDomainSignal<f32>
where
    f32: AudioSample + FftNum,
    F: Fn(f32, usize) -> f32
{
    // Compute the number of frames in the original STFT.
    let frames = signal.len().div_ceil(hop_length);
    // Compute the number of frames in the synthesized STFT.
    let synth_frames = (frames as f32 * scale_factor).ceil() as usize;

    // Build the window function used in the STFT and inverse STFT.
    let window = build_window(window_fn, window_size);
    // Compute the original signal's STFT.
    let stft = stft(&signal, window_size, hop_length, &window);

    // Extract the magnitudes and phases from the complex output of the STFT.
    let mags = stft.mapv(|v| v.abs());
    let phases = stft.mapv(|v| f32::atan2(v.im, v.re));

    // Perform a linear interpolation of the magnitudes along the time axis.
    let mut shifted_mags = Array2::from_elem((synth_frames, window_size), 0f32);
    interpolate_time_linear(shifted_mags.view_mut(), mags.view(), frames, scale_factor);

    // Compute the phase differences per frame of the STFT (AKA the derivative of the phase with
    // respect to time).
    let mut phase_diffs = &phases - shift_time(phases.view()); 
    phase_diffs.map_inplace(|v| *v = v.rem_euclid(std::f32::consts::TAU));

    // Perform a linear interpolation of the phase differences along the time axis.
    let mut shifted_phase_diffs = Array2::from_elem((synth_frames, window_size), 0f32);
    interpolate_time_linear(shifted_phase_diffs.view_mut(), phase_diffs.view(), frames, scale_factor);

    // Also store the original phases, scaled to fit the new size.
    let mut unshifted_phases = Array2::from_elem((synth_frames, window_size), 0f32); 
    interpolate_time_nearest(unshifted_phases.view_mut(), phases.view(), frames, scale_factor);

    let mut shifted_phases = Array2::from_elem((synth_frames, window_size), 0f32);
    shifted_phases.slice_mut(s![0, ..]).assign(&shifted_phase_diffs.slice(s![0, ..]));

    // Sum the interpolated phase differences along the time axis.
    // The resulting phase for each frame is the sum of all the phase differences of the previous
    // frames.
    for t in 1..synth_frames {
        let time_phase = &shifted_phases.slice(s![t - 1, ..]) + &shifted_phase_diffs.slice(s![t, ..]);
        let freq_phase = &unshifted_phases.slice(s![t, ..]);

        let mag0 = shifted_mags.slice(s![t, ..]);
        let mag1 = shifted_mags.slice(s![t - 1, ..]);
        let mut transient = (&mag0 - &mag1) / (&mag0 + &mag1);
        transient.mapv_inplace(|v| if v >= transient_cutoff { 1.0 } else { 0.0 });

        let mut new_phase = freq_phase * &transient + time_phase * (1.0 - transient);
        new_phase.map_inplace(|v| *v = v.rem_euclid(std::f32::consts::TAU));
        shifted_phases.slice_mut(s![t, ..]).assign(&new_phase);
    }

    // Synthesize the new STFT by converting phase and magnitude back to cartesian coordinates.
    let synth_stft = Zip::from(&shifted_mags).and(&shifted_phases).map_collect(|&mag, &phase| {
        Complex {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    });

    istft(synth_stft, window_size, hop_length, synth_frames * hop_length + window_size, &window)
}

/// Perform linear interpolation on a component of the STFT along the time axis, 
/// stretching by `scale_factor` and storing the result in `shifted`.
fn interpolate_time_linear(
    mut shifted: ArrayViewMut2<f32>,
    original: ArrayView2<f32>,
    frames: usize,
    scale_factor: f32,
) {
    for (i, mut shift_win) in shifted.outer_iter_mut().enumerate() {
        let index = i as f32 / scale_factor;
        let i0 = index.floor() as usize;
        let i1 = index.ceil() as usize;
        let d0 = (index - index.floor()).abs();
        let mut d1 = (index - index.ceil()).abs();

        if i0 == i1 {
            d1 = 1.0;
        }

        let mags_win_0 = original.index_axis(Axis(0), i0.min(frames - 1));
        shift_win += &mags_win_0.mapv(|v| v * (1.0 - d0));

        let mags_win_1 = original.index_axis(Axis(0), i1.min(frames - 1));
        shift_win += &mags_win_1.mapv(|v| v * (1.0 - d1));
    }
}

/// Perform nearest-neighbor interpolation on a component of the STFT along the time axis, 
/// stretching by `scale_factor` and storing the result in `shifted`.
fn interpolate_time_nearest(
    mut interpolated: ArrayViewMut2<f32>,
    original: ArrayView2<f32>,
    frames: usize,
    scale_factor: f32,
) {
    for (i, mut phases_win) in interpolated.outer_iter_mut().enumerate() {
        let index = (i as f32 / scale_factor).round() as usize;
        phases_win.assign(&original.index_axis(Axis(0), index.min(frames - 1)));
    }
}

/// Shift the given component of the STFT forward along the time axis by one frame.
fn shift_time(
    original: ArrayView2<f32>,
) -> Array2<f32> {
    let dim = original.raw_dim();
    let mut shifted = Array2::from_elem((1, dim[1]), 0f32);
    let _ = shifted.append(Axis(0), original.slice(s![..-1, ..]));
    shifted
}

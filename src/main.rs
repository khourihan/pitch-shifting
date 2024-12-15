use signal::TimeDomainSignal;

mod signal;
mod sample;
mod merge;
mod ola;
mod sola;
mod phase_vocoder;
mod windows;
mod complex;
mod fft;

const WINDOW_SIZE_MS: f32 = 20.0;
const HOP_LENGTH_MS: f32 = 8.0;

fn main() {
    let (signal, sample_rate): (TimeDomainSignal<f32>, u32) = signal::read_mono("input/powerhse.wav").unwrap();

    let window_size = (sample_rate as f32 * WINDOW_SIZE_MS / 1000.0) as usize;
    let hop_length = (sample_rate as f32 * HOP_LENGTH_MS / 1000.0) as usize;

    let stretched = phase_vocoder::phase_vocoder(
        signal,
        2.0,
        window_size,
        hop_length,
        windows::hann_window,
    );

    signal::write(stretched, sample_rate, "output/powerhse.wav").unwrap();
}

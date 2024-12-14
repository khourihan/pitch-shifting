use signal::time_domain::TimeDomainSignal;

mod signal;
mod sample;
mod merge;
mod ola;
mod sola;
mod windows;

const WINDOW_SIZE_MS: f32 = 20.0;
const HOP_LENGTH_MS: f32 = 8.0;

fn main() {
    let signal: TimeDomainSignal<f32> = TimeDomainSignal::read_mono("input/powerhse.wav").unwrap();

    let window_size = (signal.sample_rate() as f32 * WINDOW_SIZE_MS / 1000.0) as usize;
    let hop_length = (signal.sample_rate() as f32 * HOP_LENGTH_MS / 1000.0) as usize;

    let stretched = sola::sola(
        signal,
        2.0,
        window_size,
        hop_length,
        windows::hann_window,
    );

    stretched.write("output/powerhse.wav").unwrap();
}

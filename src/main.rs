use signal::Signal;

mod signal;
mod sample;
mod merge;

fn main() {
    let signal1: Signal<f32> = Signal::from_fn(44100, 1.0, |t| (t * 440.0 * 2.0 * std::f32::consts::PI).sin());
    let signal2: Signal<f32> = Signal::from_fn(44100, 1.0, |t| (t * 880.0 * 2.0 * std::f32::consts::PI).sin());
    let signal: Signal<f32> = merge::merge(signal1, signal2, |a, b| a + b);
    signal.write("output/you-are-a-toy.wav").unwrap();
}

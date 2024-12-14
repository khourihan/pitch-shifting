use signal::Signal;

mod signal;
mod sample;

fn main() {
    let signal: Signal<f32> = Signal::from_fn(44100, 1.0, |t| (t * 440.0 * 2.0 * std::f32::consts::PI).sin());
    // let signal: Signal<f32> = Signal::read_mono("input/you-are-a-toy.wav").unwrap();
    signal.write("output/you-are-a-toy.wav").unwrap();
}

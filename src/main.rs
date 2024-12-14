use signal::Signal;

mod signal;
mod sample;

fn main() {
    let signal: Signal<f32> = Signal::read_mono("input/you-are-a-toy.wav").unwrap();
    signal.write("output/you-are-a-toy.wav").unwrap();
}

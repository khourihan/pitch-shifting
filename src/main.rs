use signal::time_domain::TimeDomainSignal;

mod signal;
mod sample;
mod merge;

fn main() {
    let signal1: TimeDomainSignal<f32> = TimeDomainSignal::from_fn(44100, 1.0, |t| (t * 440.0 * 2.0 * std::f32::consts::PI).sin());
    let signal2: TimeDomainSignal<f32> = TimeDomainSignal::from_fn(44100, 1.0, |t| (t * 880.0 * 2.0 * std::f32::consts::PI).sin());
    let mut signal: TimeDomainSignal<f32> = merge::merge(signal1, signal2, |a, b| a + b);
    signal.normalize();
    signal.write("output/you-are-a-toy.wav").unwrap();
}

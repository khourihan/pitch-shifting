use signal::time_domain::TimeDomainSignal;

mod signal;
mod sample;
mod merge;
mod ola;
mod windows;

fn main() {
    let signal: TimeDomainSignal<f32> = TimeDomainSignal::read_mono("input/you-are-a-toy.wav").unwrap();

    let ola = ola::ola(
        signal,
        2.0,
        5000,
        2000,
        windows::hann_window,
    );

    ola.write("output/you-are-a-toy.wav").unwrap();
}

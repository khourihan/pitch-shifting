use crate::{sample::AudioSample, signal::Signal};

/// Merge two signals into one by applying the given function `f` to each sample.
///
/// # Panics
///
/// Panics if the signals' sample rates or number of samples do not match.
pub fn merge<T, U, V, F>(a: Signal<T>, b: Signal<U>, f: F) -> Signal<V>
where
    T: AudioSample,
    U: AudioSample,
    V: AudioSample,
    F: Fn(T, U) -> V,
{
    assert!(a.sample_rate() == b.sample_rate() && a.num_samples() == b.num_samples());

    Signal::from_iter(a.sample_rate(), a.into_samples().zip(b.into_samples()).map(|(s, t)| f(s, t)))
}

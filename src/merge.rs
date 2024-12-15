use crate::{sample::AudioSample, signal::TimeDomainSignal};

/// Merge two [`TimeDomainSignal`]s into one by applying the given function `f` to each sample.
///
/// # Panics
///
/// Panics if the signals' sample rates or number of samples do not match.
pub fn merge<T, U, V, F>(a: TimeDomainSignal<T>, b: TimeDomainSignal<U>, f: F) -> TimeDomainSignal<V>
where
    T: AudioSample,
    U: AudioSample,
    V: AudioSample,
    F: Fn(T, U) -> V,
{
    assert!(a.sample_rate() == b.sample_rate() && a.num_samples() == b.num_samples());

    TimeDomainSignal::from_iter(a.sample_rate(), a.into_samples().zip(b.into_samples()).map(|(s, t)| f(s, t)))
}

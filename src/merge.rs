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
    assert!(a.len() == b.len() && a.len() == b.len());
    a.into_iter().zip(b.into_iter()).map(|(s, t)| f(s, t)).collect()
}

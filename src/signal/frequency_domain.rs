use crate::sample::AudioSample;

/// A single-channel audio signal stored in the frequency domain.
#[derive(Debug, Clone)]
pub struct FrequencyDomainSignal<T: AudioSample> {
    /// All the samples of the signal.
    samples: Vec<T>,
}

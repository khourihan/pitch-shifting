use thiserror::Error;

use crate::sample::{AudioSample, ConvertSample, SampleFormat};

/// A single-channel audio signal stored in the time domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signal<T: AudioSample> {
    /// All the samples of the signal.
    // TODO: Consider Arc<[T]>
    samples: Box<[T]>,
    /// The number of samples per second, in Hz.
    ///
    /// A value of 44100 represents a typical 44.1 kHz sample rate.
    sample_rate: u32,
}

impl<T: AudioSample> Signal<T> {
    /// Create a new signal from a given `sample_rate` in Hz, a given `length` in seconds, and a 
    /// given function `f` that returns the amplitude of the wave at a given time in seconds.
    pub fn from_fn<F>(sample_rate: u32, length: f32, f: F) -> Signal<T>
    where
        F: Fn(f32) -> T
    {
        Signal::<T> {
            samples: (0..(sample_rate as f32 * length).floor() as u32)
                .map(|t| f(t as f32 / sample_rate as f32))
                .collect(),
            sample_rate,
        }
    }

    /// Read the WAV file at the given `path`, converting to the appropriate type and adding
    /// multiple channels into a single one if necessary.
    pub fn read_mono<P>(path: P) -> Result<Signal<T>, SignalReadError>
    where
        P: AsRef<std::path::Path>,
        T: hound::Sample,
        f32: ConvertSample<T>,
        i16: ConvertSample<T>,
        i32: ConvertSample<T>,
    {
        let reader = hound::WavReader::open(path)?;
        
        let spec = reader.spec();

        let same_format = if let hound::SampleFormat::Int = spec.sample_format {
            T::sample_format() == SampleFormat::Int
        } else {
            T::sample_format() == SampleFormat::Float
        };

        let samples = if same_format {
            reader.into_samples::<T>().collect::<Result<Vec<T>, _>>()?
        } else {
            match T::sample_format() {
                SampleFormat::Int => {
                    let samples = reader.into_samples::<f32>().collect::<Result<Vec<f32>, _>>()?;
                    samples.into_iter().map(|s| s.convert_sample()).collect::<Vec<T>>()
                },
                SampleFormat::Float => {
                    if spec.bits_per_sample == 16 {
                        let samples = reader.into_samples::<i16>().collect::<Result<Vec<i16>, _>>()?;
                        samples.into_iter().map(|s| s.convert_sample()).collect::<Vec<T>>()
                    } else {
                        let samples = reader.into_samples::<i32>().collect::<Result<Vec<i32>, _>>()?;
                        samples.into_iter().map(|s| s.convert_sample()).collect::<Vec<T>>()
                    }
                },
            }
        };

        let samples_mono = samples.chunks_exact(spec.channels as usize)
            .map(|c| c.iter().fold(T::zero(), |acc, &s| acc + s))
            .collect::<Box<[T]>>();

        Ok(Signal {
            samples: samples_mono,
            sample_rate: spec.sample_rate,
        })
    }
    
    pub fn write<P>(&self, path: P) -> Result<(), SignalWriteError>
    where
        P: AsRef<std::path::Path>,
        T: hound::Sample,
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: T::bits_per_sample(),
            sample_format: if let SampleFormat::Int = T::sample_format() { hound::SampleFormat::Int } else { hound::SampleFormat::Float },
        };

        let mut writer = hound::WavWriter::create(path, spec)?;

        for sample in self.samples.iter() {
            writer.write_sample(*sample)?;
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SignalReadError {
    #[error(transparent)]
    Wav(#[from] hound::Error),
}

#[derive(Debug, Error)]
pub enum SignalWriteError {
    #[error(transparent)]
    Wav(#[from] hound::Error),
}

use std::{alloc::Layout, ops::{AddAssign, DivAssign, MulAssign, SubAssign}};

pub trait AudioSample:
    Copy
    + num_traits::Num
    + num_traits::Bounded
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + ConvertSample<i16>
    + ConvertSample<i32>
    + ConvertSample<f32>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + Sync
    + Send
{
    fn sample_format() -> SampleFormat;

    fn bits_per_sample() -> u16;
}

impl AudioSample for i16 {
    fn sample_format() -> SampleFormat {
        SampleFormat::Int
    }

    fn bits_per_sample() -> u16 {
        16
    }
}

impl AudioSample for i32 {
    fn sample_format() -> SampleFormat {
        SampleFormat::Int
    }

    fn bits_per_sample() -> u16 {
        16
    }
}

impl AudioSample for f32 {
    fn sample_format() -> SampleFormat {
        SampleFormat::Float
    }

    fn bits_per_sample() -> u16 {
        32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Int,
    Float,
}

pub trait ConvertSample<T: AudioSample> {
    fn convert_sample(self) -> T
    where
        Self: Sized + AudioSample;
}

pub trait ConvertSamples<T: AudioSample> {
    fn convert_samples(self) -> Box<[T]>;
}

impl<T: AudioSample, U> ConvertSamples<T> for Box<[U]>
where
    U: AudioSample + ConvertSample<T>,
{
    fn convert_samples(self) -> Box<[T]> {
        let len = self.len();
        let mut out: Box<[T]> = {
            if len == 0 {
                <Box<[T]>>::default()
            } else {
                let layout = match Layout::array::<T>(len) {
                    Ok(layout) => layout,
                    Err(_) => panic!("Failed to allocate buffer of size {}", len),
                };

                let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
                let slice_ptr: *mut [T] = core::ptr::slice_from_raw_parts_mut(ptr, len);
                unsafe { Box::from_raw(slice_ptr) }
            }
        };

        for i in 0..len {
            out[i] = self[i].convert_sample();
        }

        out
    }
}

impl ConvertSample<i16> for i16 {
    #[inline(always)]
    fn convert_sample(self) -> i16 {
        self
    }
}

impl ConvertSample<i32> for i16 {
    #[inline(always)]
    fn convert_sample(self) -> i32 {
        (self as i32) << 16
    }
}

impl ConvertSample<f32> for i16 {
    #[inline(always)]
    fn convert_sample(self) -> f32 {
        ((self as f32) / (i16::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertSample<i16> for i32 {
    #[inline(always)]
    fn convert_sample(self) -> i16 {
        (self >> 16) as i16
    }
}

impl ConvertSample<i32> for i32 {
    #[inline(always)]
    fn convert_sample(self) -> i32 {
        self
    }
}

impl ConvertSample<f32> for i32 {
    #[inline(always)]
    fn convert_sample(self) -> f32 {
        ((self as f32) / (i32::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertSample<i16> for f32 {
    #[inline(always)]
    fn convert_sample(self) -> i16 {
        ((self * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16
    }
}

impl ConvertSample<i32> for f32 {
    #[inline(always)]
    fn convert_sample(self) -> i32 {
        ((self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32
    }
}

impl ConvertSample<f32> for f32 {
    #[inline(always)]
    fn convert_sample(self) -> f32 {
        self
    }
}

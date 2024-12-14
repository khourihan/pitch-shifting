use std::{alloc::Layout, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign}, slice::SliceIndex};

pub trait AudioSample:
    Copy
    + num_traits::Num
    + num_traits::Bounded
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + ConvertSample<i16>
    + ConvertSample<i32>
    + ConvertSample<f32>
    + Sync
    + Send
{
    fn sample_format() -> SampleFormat;

    fn bits_per_sample() -> u16;
}

#[derive(Debug, Default, Clone)]
pub struct Samples<T: AudioSample>(pub Vec<T>);

#[derive(Debug, Default, Clone, Copy)]
pub struct SamplesRef<'a, T: AudioSample>(pub &'a [T]);

#[derive(Debug, Default)]
pub struct SamplesMut<'a, T: AudioSample>(pub &'a mut [T]);

impl<T: AudioSample> FromIterator<T> for Samples<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Samples<T> {
        Samples(Vec::from_iter(iter))
    }
}

impl<T: AudioSample> IntoIterator for Samples<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: AudioSample> IntoIterator for SamplesRef<'a, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T: AudioSample> IntoIterator for SamplesMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T, U, I> Index<I> for Samples<T>
where 
    T: AudioSample,
    I: SliceIndex<[T], Output = U>,
    U: ?Sized,
{
    type Output = U;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, U, I> IndexMut<I> for Samples<T>
where 
    T: AudioSample,
    I: SliceIndex<[T], Output = U>,
    U: ?Sized,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<'a, T, U, I> Index<I> for SamplesRef<'a, T>
where 
    T: AudioSample,
    I: SliceIndex<[T], Output = U>,
    U: ?Sized,
{
    type Output = U;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, U, I> Index<I> for SamplesMut<'a, T>
where 
    T: AudioSample,
    I: SliceIndex<[T], Output = U>,
    U: ?Sized,
{
    type Output = U;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, U, I> IndexMut<I> for SamplesMut<'a, T>
where 
    T: AudioSample,
    I: SliceIndex<[T], Output = U>,
    U: ?Sized,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

macro_rules! impl_samples_op {
    ($opt:ident, $opf:ident) => {
        impl<T: AudioSample> $opt<Samples<T>> for Samples<T> {
            type Output = Samples<T>;

            fn $opf(self, rhs: Samples<T>) -> Self::Output {
                self.0.into_iter().zip(rhs.0.into_iter()).map(|(a, b)| a.$opf(b)).collect()
            }
        }

        impl<'a, T: AudioSample> $opt<SamplesRef<'a, T>> for Samples<T> {
            type Output = Samples<T>;

            fn $opf(self, rhs: SamplesRef<'a, T>) -> Self::Output {
                self.0.into_iter().zip(rhs.0.into_iter()).map(|(a, &b)| a.$opf(b)).collect()
            }
        }

        impl<'a, T: AudioSample> $opt<Samples<T>> for SamplesRef<'a, T> {
            type Output = Samples<T>;

            fn $opf(self, rhs: Samples<T>) -> Self::Output {
                self.0.into_iter().zip(rhs.0.into_iter()).map(|(&a, b)| a.$opf(b)).collect()
            }
        }

        impl<'a, 'b, T: AudioSample> $opt<SamplesRef<'a, T>> for SamplesRef<'b, T> {
            type Output = Samples<T>;

            fn $opf(self, rhs: SamplesRef<'a, T>) -> Self::Output {
                self.0.into_iter().zip(rhs.0.into_iter()).map(|(&a, &b)| a.$opf(b)).collect()
            }
        }
    };
}

impl_samples_op!(Add, add);
impl_samples_op!(Sub, sub);
impl_samples_op!(Mul, mul);
impl_samples_op!(Div, div);

macro_rules! impl_samples_mut_op {
    ($opt:ident, $opf:ident, $op:tt) => {
        impl<'a, T: AudioSample> $opt<Samples<T>> for SamplesMut<'a, T> {
            fn $opf(&mut self, rhs: Samples<T>) {
                self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(a, b)| *a = *a $op b);
            }
        }

        impl<'a, 'b, T: AudioSample> $opt<SamplesRef<'a, T>> for SamplesMut<'b, T> {
            fn $opf(&mut self, rhs: SamplesRef<'a, T>) {
                self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(a, b)| *a = *a $op *b);
            }
        }
    };
}

impl_samples_mut_op!(AddAssign, add_assign, +);
impl_samples_mut_op!(SubAssign, sub_assign, -);
impl_samples_mut_op!(MulAssign, mul_assign, *);
impl_samples_mut_op!(DivAssign, div_assign, /);

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

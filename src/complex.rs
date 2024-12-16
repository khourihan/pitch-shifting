use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::sample::AudioSample;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Complex<T: AudioSample> {
    pub re: T,
    pub im: T,
}

macro_rules! impl_complex_op {
    ($opt:ident, $opf:ident, $op:tt) => {
        impl<T: AudioSample> $opt<Complex<T>> for Complex<T> {
            type Output = Complex<T>;

            fn $opf(self, rhs: Complex<T>) -> Self::Output {
                Complex::<T> {
                    re: self.re $op rhs.re,
                    im: self.im $op rhs.im,
                }
            }
        }
    }
}

impl_complex_op!(Add, add, +);
impl_complex_op!(Sub, sub, -);
impl_complex_op!(Mul, mul, *);
impl_complex_op!(Div, div, /);

macro_rules! impl_complex_op_assign {
    ($opt:ident, $opf:ident, $op:tt) => {
        impl<T: AudioSample> $opt<Complex<T>> for Complex<T> {
            fn $opf(&mut self, rhs: Complex<T>) {
                self.re $op rhs.re;
                self.im $op rhs.im;
            }
        }
    }
}

impl_complex_op_assign!(AddAssign, add_assign, +=);
impl_complex_op_assign!(SubAssign, sub_assign, -=);
impl_complex_op_assign!(MulAssign, mul_assign, *=);
impl_complex_op_assign!(DivAssign, div_assign, /=);

macro_rules! impl_complex_op_scale {
    ($opt:ident, $opf:ident, $op:tt) => {
        impl<T: AudioSample> $opt<T> for Complex<T> {
            type Output = Complex<T>;

            fn $opf(self, rhs: T) -> Self::Output {
                Complex::<T> {
                    re: self.re $op rhs,
                    im: self.im $op rhs,
                }
            }
        }
    }
}

impl_complex_op_scale!(Mul, mul, *);
impl_complex_op_scale!(Div, div, /);

macro_rules! impl_complex_op_scale_assign {
    ($opt:ident, $opf:ident, $op:tt) => {
        impl<T: AudioSample> $opt<T> for Complex<T> {
            fn $opf(&mut self, rhs: T) {
                self.re $op rhs;
                self.im $op rhs;
            }
        }
    }
}

impl_complex_op_scale_assign!(MulAssign, mul_assign, *=);
impl_complex_op_scale_assign!(DivAssign, div_assign, /=);

impl<T: AudioSample + 'static> ndarray::ScalarOperand for Complex<T> {}

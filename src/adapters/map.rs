use crate::{NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::Axis;

/// A producer that maps the falues of the inner iterator with the closure.
///
/// This struct is created by the `map` method on `NdProducer`. See its
/// documentation for more.
pub struct Map<T, F> {
    inner: T,
    f: F,
}

impl<T, F> Map<T, F> {
    pub(crate) fn new(inner: T, f: F) -> Self {
        Map { inner, f }
    }
}

impl<T, F, B> NdProducer for Map<T, F>
where
    F: FnMut(T::Item) -> B,
    T: NdProducer,
{
    type Item = B;
    type Source = Map<T::Source, F>;
    fn into_source(self) -> Self::Source {
        Map {
            inner: self.inner.into_source(),
            f: self.f,
        }
    }
}

impl<T, F> NdReshape for Map<T, F>
where
    T: NdReshape,
{
    type Dim = T::Dim;
    impl_ndreshape_methods_for_wrapper!(inner);
}

impl<T, F, B> NdSource for Map<T, F>
where
    F: FnMut(T::Item) -> B,
    T: NdSource,
{
    type Item = B;
    unsafe fn read_once_unchecked(&mut self, ptr: &T::Ptr) -> B {
        (self.f)(self.inner.read_once_unchecked(ptr))
    }
}

unsafe impl<T, F> NdAccess for Map<T, F>
where
    T: NdAccess,
{
    type Dim = T::Dim;
    type Ptr = T::Ptr;
    impl_ndaccess_methods_for_wrapper!(inner);
}

unsafe impl<T, F, B> NdSourceRepeat for Map<T, F>
where
    F: Fn(T::Item) -> B,
    T: NdSourceRepeat,
{
}

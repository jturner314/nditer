use crate::{NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::Axis;

/// A producer that calls a closure with a reference to each element before
/// yielding it.
///
/// This struct is created by the `inspect` method on `NdProducer`. See its
/// documentation for more.
pub struct Inspect<T, F> {
    inner: T,
    f: F,
}

impl<T, F> Inspect<T, F> {
    pub(crate) fn new(inner: T, f: F) -> Self {
        Inspect { inner, f }
    }
}

impl<T, F> NdProducer for Inspect<T, F>
where
    F: FnMut(&T::Item),
    T: NdProducer,
{
    type Item = T::Item;
    type Source = Inspect<T::Source, F>;
    fn into_source(self) -> Self::Source {
        Inspect {
            inner: self.inner.into_source(),
            f: self.f,
        }
    }
}

impl<T, F> NdReshape for Inspect<T, F>
where
    T: NdReshape,
{
    type Dim = T::Dim;
    impl_ndreshape_methods_for_wrapper!(inner);
}

impl<T, F> NdSource for Inspect<T, F>
where
    F: FnMut(&T::Item),
    T: NdSource,
{
    type Item = T::Item;
    unsafe fn read_once_unchecked(&mut self, ptr: &T::Ptr) -> T::Item {
        let item = self.inner.read_once_unchecked(ptr);
        (self.f)(&item);
        item
    }
}

unsafe impl<T, F> NdAccess for Inspect<T, F>
where
    T: NdAccess,
{
    type Dim = T::Dim;
    type Ptr = T::Ptr;
    impl_ndaccess_methods_for_wrapper!(inner);
}

unsafe impl<T, F> NdSourceRepeat for Inspect<T, F>
where
    F: FnMut(&T::Item),
    T: NdSourceRepeat,
{
}

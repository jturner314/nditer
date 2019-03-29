use crate::{NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::Axis;

/// A producer that clones the elements of the underlying producer.
///
/// This struct is created by the `cloned` method on `NdProducer`. See its
/// documentation for more.
pub struct Cloned<T> {
    inner: T,
}

impl<T> Cloned<T> {
    pub(crate) fn new(inner: T) -> Self {
        Cloned { inner }
    }
}

impl<'a, T, B> NdProducer for Cloned<T>
where
    T: NdProducer<Item = &'a B>,
    B: 'a + Clone,
{
    type Item = B;
    type Source = Cloned<T::Source>;
    fn into_source(self) -> Self::Source {
        Cloned {
            inner: self.inner.into_source(),
        }
    }
}

impl<T> NdReshape for Cloned<T>
where
    T: NdReshape,
{
    type Dim = T::Dim;
    impl_ndreshape_methods_for_wrapper!(inner);
}

impl<'a, T, B> NdSource for Cloned<T>
where
    T: NdSource<Item = &'a B>,
    B: 'a + Clone,
{
    type Item = B;
    unsafe fn read_once_unchecked(&mut self, ptr: &T::Ptr) -> B {
        self.inner.read_once_unchecked(ptr).clone()
    }
}

unsafe impl<T> NdAccess for Cloned<T>
where
    T: NdAccess,
{
    type Dim = T::Dim;
    type Ptr = T::Ptr;
    impl_ndaccess_methods_for_wrapper!(inner);
}

unsafe impl<'a, T, B> NdSourceRepeat for Cloned<T>
where
    T: NdSourceRepeat<Item = &'a B>,
    B: 'a + Clone,
{
}

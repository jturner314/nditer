//! Error types.

use crate::AxesFor;
use ndarray::Dimension;
use std::error::Error;
use std::fmt;

/// Error that indicates the shape could not be broadcast.
#[derive(Debug)]
pub struct BroadcastError {
    /// Original shape before broadcasting.
    original: Box<[usize]>,
    /// Desired shape after broadcasting.
    desired: Box<[usize]>,
    /// Mapping of original to desired axes.
    axes_mapping: Box<[usize]>,
}

impl BroadcastError {
    pub(crate) fn new<O, D>(original: &O, desired: &D, axes_mapping: &AxesFor<D, O>) -> BroadcastError
    where
        O: Dimension,
        D: Dimension,
    {
        BroadcastError {
            original: Box::from(original.slice()),
            desired: Box::from(desired.slice()),
            axes_mapping: Box::from(axes_mapping.slice()),
        }
    }
}

impl fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "Could not broadcast {:?} to {:?} using axes mapping {:?}",
            self.original, self.desired, self.axes_mapping
        )
    }
}

impl Error for BroadcastError {}

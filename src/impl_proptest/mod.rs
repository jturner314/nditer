use self::layout::{AxesConstraints, LayoutValueTree};
pub use self::layout::{AxesConstraintsConstNdim, LayoutStrategy};
use ndarray::{Array, Dimension};
use proptest::strategy::{NewTree, Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::fmt::Debug;

// TODO: How to generate arrays with stride 0?
#[derive(Clone, Debug)]
pub struct ArrayStrategy<T, C>
where
    C: AxesConstraints,
{
    pub elem: T,
    pub layout: LayoutStrategy<C>,
}

impl<T, C> Default for ArrayStrategy<T, C>
where
    T: Default,
    C: AxesConstraints,
    LayoutStrategy<C>: Default,
{
    fn default() -> ArrayStrategy<T, C> {
        ArrayStrategy {
            elem: T::default(),
            layout: LayoutStrategy::default(),
        }
    }
}

impl<T, C> Strategy for ArrayStrategy<T, C>
where
    T: Strategy,
    C: AxesConstraints,
{
    type Tree = ArrayValueTree<T::Tree, C::Dim>;
    type Value = Array<T::Value, C::Dim>;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let layout = self.layout.new_tree(runner);
        let all_trees = (0..layout.all_trees_len())
            .map(|_| self.elem.new_tree(runner))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ArrayValueTree {
            all_trees,
            layout,
            simplify_action: Some(Action::Layout),
            complicate_action: None,
        })
    }
}

/// A simplify/complicate action for an `ArrayValueTree`.
#[derive(Clone, Copy, Debug)]
enum Action {
    /// Simplify/complicate the layout.
    Layout,
    /// Simplify/complicate the element at the given index in `all_trees`.
    Element(usize),
}

/// `ValueTree` corresponding to `ArrayStrategy`.
#[derive(Clone, Debug)]
pub struct ArrayValueTree<A, D: Dimension> {
    all_trees: Vec<A>,
    layout: LayoutValueTree<D>,
    /// Action to perform on next `simplify` call.
    simplify_action: Option<Action>,
    /// Action to perform on next `complicate` call.
    complicate_action: Option<Action>,
}

impl<A: ValueTree, D: Dimension> ValueTree for ArrayValueTree<A, D> {
    type Value = Array<A::Value, D>;

    fn current(&self) -> Array<A::Value, D> {
        self.layout.current(&self.all_trees)
    }

    fn simplify(&mut self) -> bool {
        if let Some(action) = self.simplify_action {
            match action {
                Action::Layout => {
                    if self.layout.simplify() {
                        self.complicate_action = Some(action);
                        true
                    } else {
                        self.complicate_action = None;
                        if !self.all_trees.is_empty() {
                            self.simplify_action = Some(Action::Element(0));
                            true
                        } else {
                            self.simplify_action = None;
                            false
                        }
                    }
                }
                Action::Element(index) => {
                    if self.all_trees[index].simplify() {
                        self.complicate_action = Some(action);
                        true
                    } else {
                        self.complicate_action = None;
                        let next_index = index + 1;
                        if next_index < self.all_trees.len() {
                            self.simplify_action = Some(Action::Element(next_index));
                            true
                        } else {
                            self.simplify_action = None;
                            false
                        }
                    }
                }
            }
        } else {
            false
        }
    }

    fn complicate(&mut self) -> bool {
        if let Some(action) = self.complicate_action {
            match action {
                Action::Layout => {
                    if self.layout.complicate() {
                        true
                    } else {
                        self.complicate_action = None;
                        false
                    }
                }
                Action::Element(index) => {
                    if self.all_trees[index].complicate() {
                        true
                    } else {
                        self.complicate_action = None;
                        false
                    }
                }
            }
        } else {
            false
        }
    }
}

mod layout;

#[cfg(test)]
mod tests {
    use super::{ArrayStrategy, AxesConstraintsConstNdim, LayoutStrategy};
    use ndarray::prelude::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn example2(
            (arr1, arr2) in ArrayStrategy {
                elem: 0..10,
                layout: LayoutStrategy {
                    axes_constraints: AxesConstraintsConstNdim::<Ix3>::default(),
                    ..Default::default()
                }
            }
            .prop_flat_map(|arr1| {
                let nrows = arr1.len_of(Axis(0));
                let ncols = arr1.len_of(Axis(1));
                (
                    Just(arr1),
                    ArrayStrategy {
                        elem: 10..20,
                        layout: LayoutStrategy {
                            axes_constraints: AxesConstraintsConstNdim::<Ix2> {
                                axis_lens: vec![nrows..nrows+1, ncols..ncols+1],
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    },
                )
            })
        ) {
            prop_assert_eq!(&arr1.shape()[..2], arr2.shape());
            // prop_assert_ne!(arr1, arr2);
        }
    }

    proptest! {
        #[test]
        fn example(arr in ArrayStrategy {
            elem: 0..10,
            layout: LayoutStrategy::<AxesConstraintsConstNdim<Ix3>>::default(),
        }) {
            prop_assert_ne!(arr.sum() % 5, 4);
        }
    }

    #[test]
    fn exampleslfjkl() {
        let mut runner = proptest::test_runner::TestRunner::default();
        let strategy = ArrayStrategy {
            elem: 0..10,
            layout: LayoutStrategy::<AxesConstraintsConstNdim<Ix3>>::default(),
        };
        runner.run(&strategy, |arr| Ok(())).unwrap();
    }

    #[test]
    fn test_add2() {
        let mut runner = proptest::test_runner::TestRunner::default();
        // Combine our two inputs into a strategy for one tuple. Our test
        // function then destructures the generated tuples back into separate
        // `a` and `b` variables to be passed in to `add()`.
        runner
            .run(&(0..1000i32, 0..1000i32), |(a, b)| {
                let sum = a + b;
                assert!(sum >= a);
                assert!(sum >= b);
                Ok(())
            })
            .unwrap();
    }

    proptest! {
        #[test]
        fn test_add(a in 0..1000i32, b in 0..1000i32) {
            let sum = a + b;
            assert!(sum >= a);
            assert!(sum >= b);
        }
    }
}

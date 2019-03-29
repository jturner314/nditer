# nditer

[![Build status](https://travis-ci.org/jturner314/nditer.svg?branch=master)](https://travis-ci.org/jturner314/nditer)
[![Coverage](https://codecov.io/gh/jturner314/nditer/branch/master/graph/badge.svg)](https://codecov.io/gh/jturner314/nditer)

This is an experimental, unstable crate for high-performance iteration over
n-dimensional arrays. `nditer::NdProducer` is intended to be a replacement for
`ndarray::NdProducer` that provides

* more sophisticated optimizations for better performance,

* more adapters other than just zipping arrays together, and

* the ability to collect the results into an array instead of having to always
  manually pre-allocate the result array.

The public API is likely to undergo significant changes in the near future, the
current implementation depends on undocumented (unstable) features of
`ndarray`, and a lot more tests need to be added, so this crate isn't ready for
production use.

In the future, this may be merged into `ndarray`, but it's a fairly large crate
(and will get bigger as more features are added), so we'll see how things go.

## Documentation

View the documentation with

```sh
cargo doc --open
```

## License

Copyright 2019 Jim Turner

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.

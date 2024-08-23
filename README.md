<h1 align="center">
  <code>async-tensorrt</code>
</h1>
<p align="center">Asynchronous wrapper for TensorRT.</p>
<div align="center">

[![version](https://img.shields.io/crates/v/async-tensorrt)](https://crates.io/crates/async-tensorrt)
[![license](https://img.shields.io/crates/l/async-tensorrt)](#license)
[![docs](https://img.shields.io/docsrs/async-tensorrt)](https://docs.rs/async-tensorrt)

</div>

## ℹ️ Introduction

`async-tensorrt` is an async wrapper for [TensorRT](https://developer.nvidia.com/tensorrt).

## 🛠 S️️tatus

This project is still a work-in-progress, and will contain bugs. Some parts of the API have not
been flushed out yet. Use with caution.

## 📦 Setup

Make sure you have the necessary dependencies installed:

* CUDA toolkit 11 or later.
* TensorRT 8 or later.

Then, add the following to your dependencies in `Cargo.toml`:

```toml
async-tensorrt = "0.8"
```

## ⚠️ Safety warning

This crate is **intentionally unsafe**. Due to the limitations of how async Rust currently works,
usage of the async interface of this crate can cause undefined behavior in some rare cases. It is up
to the user of this crate to prevent this from happening by following these rules:

* No futures produced by functions in this crate may be leaked (either by `std::mem::forget` or
  otherwise).
* Use a well-behaved runtime (one that will not forget your future) like Tokio or async-std.

Internally, the `Future` type in this crate schedules a CUDA call on a separate runtime thread. To
make the API as ergonomic as possible, the lifetime bounds of the closure (that is sent to the
runtime) are tied to the future object. To enforce this bound, the future will block and wait if it
is dropped. This mechanism relies on the future being driven to completion, and not forgotten. This
is not necessarily guaranteed. Unsafety may arise if either the runtime gives up on or forgets the
future, or the caller manually polls the future, then forgets it.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

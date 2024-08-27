#![recursion_limit = "256"]

#[cfg(not(feature = "lean"))]
pub mod builder;

pub mod engine;
pub mod error;
pub mod ffi;
pub mod runtime;

#[cfg(test)]
mod tests;

#[cfg(not(feature = "lean"))]
pub use builder::Builder;

pub use engine::{Engine, ExecutionContext};
pub use error::Error;

#[cfg(not(feature = "lean"))]
pub use ffi::builder_config::BuilderConfig;

pub use ffi::memory::HostBuffer;
pub use ffi::network::{NetworkDefinition, NetworkDefinitionCreationFlags, Tensor};

#[cfg(not(feature = "lean"))]
pub use ffi::optimization_profile::OptimizationProfile;

pub use ffi::parser::Parser;
pub use runtime::Runtime;

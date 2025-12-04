//! Dataset loading, validation, and statistics utilities.
//!
//! Provides comprehensive tools for working with IR evaluation datasets.

#[cfg(feature = "serde")]
mod loaders;
#[cfg(feature = "serde")]
mod validator;
#[cfg(feature = "serde")]
mod statistics;

#[cfg(feature = "serde")]
pub use loaders::*;
#[cfg(feature = "serde")]
pub use validator::*;
#[cfg(feature = "serde")]
pub use statistics::*;


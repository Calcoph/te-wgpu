//! Test utilities for the wgpu repository.

mod config;
pub mod image;
mod init;
mod isolation;
pub mod native;
mod params;
mod report;
mod run;

#[cfg(target_arch = "wasm32")]
pub use init::initialize_html_canvas;

pub use self::image::ComparisonType;
pub use config::GpuTestConfiguration;
#[doc(hidden)]
pub use ctor::ctor;
pub use init::{initialize_adapter, initialize_device, initialize_instance};
pub use params::{FailureCase, FailureReasons, TestParameters};
pub use run::{execute_test, TestingContext};
pub use wgpu_macros::gpu_test;

/// Run some code in an error scope and assert that validation fails.
pub fn fail<T, E: std::error::Error>(callback: impl FnOnce() -> Result<T, E>) {
    assert!(callback().is_err());
}

/// Run some code in an error scope and assert that validation succeeds.
pub fn valid<T, E: std::error::Error>(callback: impl FnOnce() -> Result<T, E>) -> T {
    let result = callback();
    assert!(result.is_ok());

    result.unwrap()
}

/// Run some code in an error scope and assert that validation succeeds or fails depending on the
/// provided `should_fail` boolean.
pub fn fail_if<T, E: std::error::Error>(should_fail: bool, callback: impl FnOnce() -> Result<T, E>) -> Option<T> {
    if should_fail {
        fail(callback);
        None
    } else {
        Some(valid(callback))
    }
}

/// Adds the necissary main function for our gpu test harness.
#[macro_export]
macro_rules! gpu_test_main {
    () => {
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
        #[cfg(target_arch = "wasm32")]
        fn main() {}

        #[cfg(not(target_arch = "wasm32"))]
        fn main() -> $crate::native::MainResult {
            $crate::native::main()
        }
    };
}

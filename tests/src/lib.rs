//! Test utilities for the wgpu repository.

#![allow(clippy::arc_with_non_send_sync)] // False positive on wasm

mod config;
mod expectations;
pub mod image;
mod init;
mod isolation;
pub mod native;
mod params;
mod poll;
mod report;
mod run;

use std::fmt::Debug;

#[cfg(target_arch = "wasm32")]
pub use init::initialize_html_canvas;

pub use self::image::ComparisonType;
pub use config::GpuTestConfiguration;
#[doc(hidden)]
pub use ctor::ctor;
pub use expectations::{FailureApplicationReasons, FailureBehavior, FailureCase, FailureReason};
pub use init::{initialize_adapter, initialize_device, initialize_instance};
pub use params::TestParameters;
pub use run::{execute_test, TestingContext};
pub use wgpu_macros::gpu_test;

/// Run some code in an error scope and assert that validation fails.
pub fn fail<T, E: Debug + ToString>(callback: impl FnOnce() -> Result<T, E>, expected_msg_substring: Option<&'static str>,) -> E {
    let result = callback();
    let validation_error = result.err()
        .expect("expected validation error in callback, but no validation error was emitted");
    if let Some(expected_msg_substring) = expected_msg_substring {
        let lowered_expected = expected_msg_substring.to_lowercase();
        let lowered_actual = validation_error.to_string().to_lowercase();
        assert!(
            lowered_actual.contains(&lowered_expected),
            concat!(
                "expected validation error case-insensitively containing {:?}, ",
                "but it was not present in actual error message:\n{:?}"
            ),
            expected_msg_substring,
            validation_error
        );
    }

    validation_error
}

/// Run some code in an error scope and assert that validation succeeds.
#[track_caller]
pub fn valid<T, E: Debug>(callback: impl FnOnce() -> Result<T, E>) -> T {
    let result = callback();
    assert!(result.is_ok());

    result.unwrap()
}

/// Run some code in an error scope and assert that validation succeeds or fails depending on the
/// provided `should_fail` boolean.
pub fn fail_if<T, E: Debug + ToString>(
    should_fail: bool,
    callback: impl FnOnce() -> Result<T, E>,
    expected_msg_substring: Option<&'static str>,
) -> Result<T, E> {
    if should_fail {
        Err(fail(callback, expected_msg_substring))
    } else {
        Ok(valid(callback))
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

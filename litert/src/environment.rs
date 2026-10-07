//! LiteRT execution environment — the hardware context shared across models.

use std::ptr::NonNull;
use std::sync::Arc;

use litert_sys as sys;

use crate::{check, Result};

/// Holds the hardware context (device handles, caches, delegates).
///
/// One [`Environment`] is typically created at application startup and shared
/// across every [`CompiledModel`](crate::CompiledModel) and [`Model`](crate::Model).
/// Cloning is cheap: the handle is `Arc`-backed and the underlying LiteRT
/// object is destroyed when the last clone is dropped.
#[derive(Clone)]
pub struct Environment {
    inner: Arc<EnvironmentInner>,
}

struct EnvironmentInner {
    ptr: NonNull<sys::LiteRtEnvironmentT>,
}

impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Environment")
            .field("ptr", &self.inner.ptr.as_ptr())
            .field("strong_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

impl Environment {
    /// Creates a new environment with no extra options. Enough for CPU and
    /// the default GPU/Metal backends on desktop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`](crate::Error::Status) if LiteRT reports an
    /// initialisation failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// let env = litert::Environment::new()?;
    /// # Ok::<(), litert::Error>(())
    /// ```
    pub fn new() -> Result<Self> {
        let mut raw: sys::LiteRtEnvironment = std::ptr::null_mut();
        check(unsafe { sys::LiteRtCreateEnvironment(0, std::ptr::null(), &mut raw) })?;
        let ptr = NonNull::new(raw).ok_or(crate::Error::NullPointer)?;
        Ok(Self {
            inner: Arc::new(EnvironmentInner { ptr }),
        })
    }

    pub(crate) fn as_raw(&self) -> sys::LiteRtEnvironment {
        self.inner.ptr.as_ptr()
    }
}

impl Drop for EnvironmentInner {
    fn drop(&mut self) {
        unsafe { sys::LiteRtDestroyEnvironment(self.ptr.as_ptr()) }
    }
}

// Safety: the environment is an opaque refcounted context safe for concurrent
// read access across threads. Destruction happens only when the last `Arc`
// handle is dropped, which Rust's ownership rules already serialise.
unsafe impl Send for EnvironmentInner {}
unsafe impl Sync for EnvironmentInner {}

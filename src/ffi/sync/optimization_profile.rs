use cpp::cpp;

use crate::ffi::result;
use crate::ffi::MAX_DIMS;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::OptimizationProfile`].
///
/// Refer to [`crate::OptimizationProfile`] for documentation.
pub struct OptimizationProfile(*mut std::ffi::c_void);

/// Implements [`Send`] for [`OptimizationProfile`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`OptimizationProfile`].
unsafe impl Send for OptimizationProfile {}

/// Implements [`Sync`] for [`OptimizationProfile`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`OptimizationProfile`].
unsafe impl Sync for OptimizationProfile {}

#[derive(Copy, Clone, Debug)]
#[repr(i32)]
enum OptimizationProfileSelector {
    Min = 0,
    Opt = 1,
    Max = 2,
}

impl OptimizationProfile {
    #[inline]
    pub(crate) fn wrap(internal: *mut std::ffi::c_void) -> Self {
        OptimizationProfile(internal)
    }

    pub fn set_min_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.set_shape_values(input_name, OptimizationProfileSelector::Min as i32, values);
    }

    pub fn set_opt_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.set_shape_values(input_name, OptimizationProfileSelector::Opt as i32, values);
    }

    pub fn set_max_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.set_shape_values(input_name, OptimizationProfileSelector::Max as i32, values);
    }

    fn set_shape_values(&mut self, input_name: &str, select: i32, values: &[i32]) -> bool {
        let internal = self.as_mut_ptr();
        let input_name_cstr = std::ffi::CString::new(input_name).unwrap();
        let input_name_ptr = input_name_cstr.as_ptr();
        let nb_values = values.len() as i32;
        let values_ptr = values.as_ptr();

        let res = cpp!(unsafe [
            internal as "void*",
            input_name_ptr as "const char*",
            select as "OptProfileSelector",
            values_ptr as "const int32_t*",
            nb_values as "int32_t"
        ] -> bool as "bool" {
            return ((IOptimizationProfile*) internal)->setShapeValues(input_name_ptr, select, values_ptr, nb_values);
        });
        res
    }

    pub fn get_min_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.get_shape_values(input_name, OptimizationProfileSelector::Min as i32);
    }

    pub fn get_opt_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.get_shape_values(input_name, OptimizationProfileSelector::Opt as i32);
    }

    pub fn get_max_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.get_shape_values(input_name, OptimizationProfileSelector::Max as i32);
    }

    fn get_shape_values(&self, input_name: &str, select: i32) -> Result<Option<Vec<i32>>> {
        let internal = self.as_ptr();
        let input_name_cstr = std::ffi::CString::new(input_name).unwrap();
        let input_name_ptr = input_name_cstr.as_ptr();

        let nb_shape_values = cpp!(unsafe [
            internal as "void*",
            input_name_ptr as "const char*"
        ] -> i32 as "int32_t" {
            return ((const IOptimizationProfile*) internal)->getNbShapeValues(input_name_ptr);
        });

        if nb_shape_values < 0 {
            return Ok(None);
        }
        let nb_shape_values = nb_shape_values as usize;
        let mut values = Vec::with_capacity(nb_shape_values);

        let shape_values = cpp!(unsafe [
            internal as "void*",
            input_name_ptr as "const char*",
            select as "OptProfileSelector"
        ] -> *const i32 as "const int32_t*" {
            return ((const IOptimizationProfile*) internal)->getShapeValues(input_name_ptr, select);
        });
        let shape_values = result!(shape_values, shape_values)?;
        for i in 0..nb_shape_values {
            let dim = unsafe { *shape_values.add(i) };
            values.push(dim)
        }
        return Ok(Some(values));
    }

    pub fn set_min_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.set_dimensions(input_name, OptimizationProfileSelector::Min as i32, dims);
    }

    pub fn set_opt_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.set_dimensions(input_name, OptimizationProfileSelector::Opt as i32, dims);
    }

    pub fn set_max_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.set_dimensions(input_name, OptimizationProfileSelector::Max as i32, dims);
    }

    fn set_dimensions(&mut self, input_name: &str, select: i32, dims: &[i32]) -> bool {
        let internal = self.as_mut_ptr();
        let input_name_cstr = std::ffi::CString::new(input_name).unwrap();
        let input_name_ptr = input_name_cstr.as_ptr();
        let nb_dims = dims.len() as i32;
        let dims_ptr = dims.as_ptr();

        let res = cpp!(unsafe [
            internal as "void*",
            input_name_ptr as "const char*",
            select as "OptProfileSelector",
            dims_ptr as "const int32_t*",
            nb_dims as "int32_t"
        ] -> bool as "bool" {
            nvinfer1::Dims xdims;
            xdims.nbDims = nb_dims;
            for (int i = 0; i < xdims.nbDims; ++i) {
                xdims.d[i] = dims_ptr[i];
            }

            return ((IOptimizationProfile*) internal)->setDimensions(input_name_ptr, select, xdims);
        });
        res
    }

    pub fn get_min_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        return self.get_dimensions(input_name, OptimizationProfileSelector::Min as i32);
    }

    pub fn get_opt_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        return self.get_dimensions(input_name, OptimizationProfileSelector::Opt as i32);
    }

    pub fn get_max_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        return self.get_dimensions(input_name, OptimizationProfileSelector::Max as i32);
    }

    fn get_dimensions(&self, input_name: &str, select: i32) -> Option<Vec<i32>> {
        let internal = self.as_ptr();
        let input_name_cstr = std::ffi::CString::new(input_name).unwrap();
        let input_name_ptr = input_name_cstr.as_ptr();
        let mut dims = Vec::with_capacity(MAX_DIMS);
        let dims_ptr = dims.as_mut_ptr();

        let num_dimensions = cpp!(unsafe [
            internal as "void*",
            input_name_ptr as "const char*",
            select as "OptProfileSelector",
            dims_ptr as "int32_t*"
        ] -> i32 as "int32_t" {
            auto dims = ((const IOptimizationProfile*) internal)->getDimensions(input_name_ptr, select);
            if (dims.nbDims > 0) {
                for (int i = 0; i < dims.nbDims; ++i) {
                    dims_ptr[i] = dims.d[i];
                }
            }
            return dims.nbDims;
        });
        if num_dimensions >= 0 {
            // Safety: The vec has been initialized up until num_dimensions elements
            unsafe {
                dims.set_len(num_dimensions as usize);
            }
            return Some(dims);
        } else {
            return None;
        }
    }

    pub fn set_extra_memory_target(&mut self, target: f32) -> bool {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*",
            target as "float"
        ] -> bool as "bool" {
            return ((IOptimizationProfile*) internal)->setExtraMemoryTarget(target);
        })
    }

    pub fn get_extra_memory_target(&self) -> f32 {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> f32 as "float" {
            return ((const IOptimizationProfile*) internal)->getExtraMemoryTarget();
        })
    }

    pub fn is_valid(&self) -> bool {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> bool as "bool" {
            return ((const IOptimizationProfile*) internal)->isValid();
        })
    }

    /// Get internal readonly pointer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let OptimizationProfile(internal) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let OptimizationProfile(internal) = *self;
        internal
    }
}

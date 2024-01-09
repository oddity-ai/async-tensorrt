use std::marker::PhantomData;

use cpp::cpp;

use crate::ffi::result;
use crate::ffi::sync::builder::Builder;

/// Defined in `NvInferRuntimeBase.h`
const MAX_DIMS: usize = 8;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::OptimizationProfile`].
///
/// Refer to [`crate::OptimizationProfile`] for documentation.
pub struct OptimizationProfile<'builder>(*mut std::ffi::c_void, PhantomData<&'builder ()>);

/// Implements [`Send`] for [`OptimizationProfile`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`OptimizationProfile`].
unsafe impl<'builder> Send for OptimizationProfile<'builder> {}

/// Implements [`Sync`] for [`OptimizationProfile`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`OptimizationProfile`].
unsafe impl<'builder> Sync for OptimizationProfile<'builder> {}

/// Optimization profile selector.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#afd20e1d227abd394fdd3af0cb1525104)
#[derive(Copy, Clone, Debug)]
#[repr(i32)]
enum OptimizationProfileSelector {
    /// This is used to set or get the minimum permitted value for dynamic dimensions etc.
    Min = 0,
    /// This is used to set or get the value that is used in the optimization (kernel selection).
    Opt = 1,
    /// This is used to set or get the maximum permitted value for dynamic dimensions etc.
    Max = 2,
}

impl<'builder> OptimizationProfile<'builder> {
    /// Wrap internal pointer as [`OptimizationProfile`].
    ///
    /// # Arguments
    ///
    /// * `internal` - Pointer to wrap.
    /// * `_builder` - Reference to builder to tie lifetime of optimization profile to.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `IOptimizationProfile` object.
    #[inline]
    pub(crate) fn wrap(internal: *mut std::ffi::c_void, _builder: &'builder Builder) -> Self {
        OptimizationProfile(internal, PhantomData)
    }

    /// Set the minimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d701934850
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `values` - Shape values.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    #[inline]
    pub fn set_min_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        self.set_shape_values(input_name, OptimizationProfileSelector::Min as i32, values)
    }

    /// Set the optimium values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d701934850
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `values` - Shape values.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    #[inline]
    pub fn set_opt_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        self.set_shape_values(input_name, OptimizationProfileSelector::Opt as i32, values)
    }

    /// Set the maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d701934850
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `values` - Shape values.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    #[inline]
    pub fn set_max_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        self.set_shape_values(input_name, OptimizationProfileSelector::Max as i32, values)
    }

    /// Set the minimum / optimum / maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d701934850
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `select` - Optimization profile selector as integer.
    /// * `values` - Shape values.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
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

    /// Get the minimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    ///
    /// # Return value
    ///
    /// Input shape if previously set.
    pub fn get_min_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        self.get_shape_values(input_name, OptimizationProfileSelector::Min as i32)
    }

    /// Get the optimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    ///
    /// # Return value
    ///
    /// Input shape if previously set.
    pub fn get_opt_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        self.get_shape_values(input_name, OptimizationProfileSelector::Opt as i32)
    }

    /// Get the maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    ///
    /// # Return value
    ///
    /// Input shape if previously set.
    pub fn get_max_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        self.get_shape_values(input_name, OptimizationProfileSelector::Max as i32)
    }

    /// Get the minimum / optimum / maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `select` - Optimization profile selector as integer.
    ///
    /// # Return value
    ///
    /// Input shape if previously set.
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
        Ok(Some(values))
    }

    /// Set the minimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    pub fn set_min_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        self.set_dimensions(input_name, OptimizationProfileSelector::Min as i32, dims)
    }

    /// Set the optimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    pub fn set_opt_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        self.set_dimensions(input_name, OptimizationProfileSelector::Opt as i32, dims)
    }

    /// Set the maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
    pub fn set_max_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        self.set_dimensions(input_name, OptimizationProfileSelector::Max as i32, dims)
    }

    /// Set the minimum / optimum / maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `select` - Optimization profile selector as integer.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// `false` if an inconsistency was detected.
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

    /// Get the minimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// Dimensions if they have been previously set.
    pub fn get_min_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        self.get_dimensions(input_name, OptimizationProfileSelector::Min as i32)
    }

    /// Get the optimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// Dimensions if they have been previously set.
    pub fn get_opt_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        self.get_dimensions(input_name, OptimizationProfileSelector::Opt as i32)
    }

    /// Get the maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// Dimensions if they have been previously set.
    pub fn get_max_dimensions(&self, input_name: &str) -> Option<Vec<i32>> {
        self.get_dimensions(input_name, OptimizationProfileSelector::Max as i32)
    }

    /// Get the minimum / optimum / maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    ///
    /// # Arguments
    ///
    /// * `input_name` - Name of input tensor.
    /// * `select` - Optimization profile selector as integer.
    /// * `dims` - Dimensions.
    ///
    /// # Return value
    ///
    /// Dimensions if they have been previously set.
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
            Some(dims)
        } else {
            None
        }
    }

    /// Set a target for extra GPU memory that may be used by this profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#abc9215e02ad6b5d911b35d45d59236e7)
    ///
    /// # Arguments
    ///
    /// * `target` - Additional memory that the builder should aim to maximally allocate for this profile, as a fraction of the memory it would use if the user did not impose any constraints on memory.
    ///
    /// # Return value
    ///
    /// `true` if the input is in the valid range (between 0 and 1 inclusive), else `false`.
    pub fn set_extra_memory_target(&mut self, target: f32) -> bool {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*",
            target as "float"
        ] -> bool as "bool" {
            return ((IOptimizationProfile*) internal)->setExtraMemoryTarget(target);
        })
    }

    /// Get the extra memory target that has been defined for this profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#aa5339baa4f134993667bc2df94cb0c2e)
    pub fn get_extra_memory_target(&self) -> f32 {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> f32 as "float" {
            return ((const IOptimizationProfile*) internal)->getExtraMemoryTarget();
        })
    }

    /// Check whether the optimization profile is valid.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ae817a3cfb3f528a7b00173336521a187)
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
        let OptimizationProfile(internal, _) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let OptimizationProfile(internal, _) = *self;
        internal
    }
}

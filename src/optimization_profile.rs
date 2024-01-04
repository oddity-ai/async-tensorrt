use crate::ffi::sync::optimization_profile::OptimizationProfile as InnerOptimizationProfile;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Optimization profile for dynamic input dimensions and shape tensors.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html)
pub struct OptimizationProfile<'a> {
    inner: Option<&'a InnerOptimizationProfile>,
    inner_mut: Option<&'a mut InnerOptimizationProfile>,
}

impl<'a> OptimizationProfile<'a> {
    /// Create [`OptimizationProfile`] from its inner object.
    pub fn from_inner(inner: &'a InnerOptimizationProfile) -> OptimizationProfile<'a> {
        Self {
            inner: Some(inner),
            inner_mut: None,
        }
    }

    /// Create [`OptimizationProfile`] from its inner mutable object.
    pub fn from_inner_mut(inner: &'a mut InnerOptimizationProfile) -> OptimizationProfile<'a> {
        Self {
            inner: None,
            inner_mut: Some(inner),
        }
    }

    /// Set the minimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    #[inline(always)]
    pub fn set_min_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.inner_mut().set_min_dimensions(input_name, dims);
    }

    /// Set the optimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    #[inline(always)]
    pub fn set_opt_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.inner_mut().set_opt_dimensions(input_name, dims);
    }

    /// Set the maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ab723695382d6b03d4a0463b8cbe2b19f)
    #[inline(always)]
    pub fn set_max_dimensions(&mut self, input_name: &str, dims: &[i32]) -> bool {
        return self.inner_mut().set_max_dimensions(input_name, dims);
    }

    /// Get the minimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    #[inline(always)]
    pub fn get_min_dimensions(&'a self, input_name: &str) -> Option<Vec<i32>> {
        return self.inner().get_min_dimensions(input_name);
    }

    /// Get the optimum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    #[inline(always)]
    pub fn get_opt_dimensions(&'a self, input_name: &str) -> Option<Vec<i32>> {
        return self.inner().get_opt_dimensions(input_name);
    }

    /// Get the maximum dimensions for a dynamic input tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a495725c79864f3e4059055307a8cc59d)
    #[inline(always)]
    pub fn get_max_dimensions(&'a self, input_name: &str) -> Option<Vec<i32>> {
        return self.inner().get_max_dimensions(input_name);
    }

    /// Set the minimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d70193485)
    #[inline(always)]
    pub fn set_min_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.inner_mut().set_min_shape_values(input_name, values);
    }

    /// Set the optimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d70193485)
    #[inline(always)]
    pub fn set_opt_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.inner_mut().set_opt_shape_values(input_name, values);
    }

    /// Set the maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ad89508bb5e59d46d106cb74d70193485)
    #[inline(always)]
    pub fn set_max_shape_values(&mut self, input_name: &str, values: &[i32]) -> bool {
        return self.inner_mut().set_max_shape_values(input_name, values);
    }

    /// Get the minimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    #[inline(always)]
    pub fn get_min_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.inner().get_min_shape_values(input_name);
    }

    /// Get the optimum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    #[inline(always)]
    pub fn get_opt_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.inner().get_opt_shape_values(input_name);
    }

    /// Get the maximum values for an input shape tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#a0654f6beafd1e4004950d5cd45ecab2b)
    #[inline(always)]
    pub fn get_max_shape_values(&self, input_name: &str) -> Result<Option<Vec<i32>>> {
        return self.inner().get_max_shape_values(input_name);
    }

    /// Set a target for extra GPU memory that may be used by this profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ae817a3cfb3f528a7b00173336521a187)
    #[inline(always)]
    pub fn set_extra_memory_target(&mut self, target: f32) -> bool {
        self.inner_mut().set_extra_memory_target(target)
    }

    /// Get the extra memory target that has been defined for this profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#aa5339baa4f134993667bc2df94cb0c2e)
    #[inline(always)]
    pub fn get_extra_memory_target(&self) -> f32 {
        self.inner().get_extra_memory_target()
    }

    /// Check whether the optimization profile can be passed to an IBuilderConfig object.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html#ae817a3cfb3f528a7b00173336521a187)
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.inner().is_valid()
    }

    pub fn inner(&self) -> &InnerOptimizationProfile {
        self.inner_mut
            .as_ref()
            .map(|o| &**o)
            .or(self.inner)
            .unwrap()
    }

    fn inner_mut(&mut self) -> &mut InnerOptimizationProfile {
        *(self.inner_mut.as_mut().unwrap())
    }
}

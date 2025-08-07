use cpp::cpp;

/// Returns (Major, Minor, Patch, Build) version of tensorrt
pub fn get_tensorrt_version() -> (u32, u32, u32) {
    (
        cpp!(unsafe [] -> u32 as "uint32_t" {
            return NV_TENSORRT_MAJOR;
        }),
        cpp!(unsafe [] -> u32 as "uint32_t" {
            return NV_TENSORRT_MINOR;
        }),
        cpp!(unsafe [] -> u32 as "uint32_t" {
            return NV_TENSORRT_PATCH;
        }),
    )
}

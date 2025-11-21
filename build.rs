fn search_for_path(
    base_env: &str,
    default_base: Option<&str>,
    include_env: &str,
    lib_env: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let base_path = if let Some(default_base) = default_base {
        std::env::var(base_env)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from(default_base))
    } else {
        std::env::var(base_env)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| panic!("Missing environment variable `{base_env}`."))
    };

    let include_path = std::env::var(include_env)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| base_path.join("include"));

    let lib_path = std::env::var(lib_env)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            #[cfg(not(windows))]
            {
                base_path.join("lib64")
            }
            #[cfg(windows)]
            {
                base_path.join("lib").join("x64")
            }
        });

    (include_path, lib_path)
}

fn main() {
    #[cfg(not(windows))]
    let (cuda_include_path, cuda_lib_path) = search_for_path(
        "CUDA_PATH",
        Some("/usr/local/cuda"),
        "CUDA_INCLUDE_PATH",
        "CUDA_LIB_PATH",
    );

    #[cfg(windows)]
    let (cuda_include_path, cuda_lib_path) =
        search_for_path("CUDA_PATH", None, "CUDA_INCLUDE_PATH", "CUDA_LIB_PATH");

    #[cfg(not(windows))]
    let (tensorrt_include_path, tensorrt_lib_path) = search_for_path(
        "TENSORRT_PATH",
        Some("/usr/local/tensorrt"),
        "TENSORRT_INCLUDE_PATH",
        "TENSORRT_LIB_PATH",
    );

    #[cfg(windows)]
    let (tensorrt_include_path, tensorrt_lib_path) = search_for_path(
        "TENSORRT_PATH",
        None,
        "TENSORRT_INCLUDE_PATH",
        "TENSORRT_LIB_PATH",
    );

    let mut cpp_build_config = cpp_build::Config::new();
    cpp_build_config.include(cuda_include_path);
    cpp_build_config.include(tensorrt_include_path);
    cpp_build_config.build("src/lib.rs");

    println!("cargo:rustc-link-search={}", cuda_lib_path.display());
    println!("cargo:rustc-link-search={}", tensorrt_lib_path.display());

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
}

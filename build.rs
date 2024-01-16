fn main() {
    let cuda_path = std::env::var("CUDA_PATH").map(std::path::PathBuf::from);

    #[cfg(not(windows))]
    let cuda_path = cuda_path.unwrap_or_else(|_| std::path::PathBuf::from("/usr/local/cuda"));
    #[cfg(windows)]
    let cuda_path = cuda_path.expect("Missing environment variable `CUDA_PATH`.");

    let cuda_include_path = std::env::var("CUDA_INCLUDE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| cuda_path.join("include"));

    let cuda_lib_path = std::env::var("CUDA_LIB_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            #[cfg(not(windows))]
            {
                cuda_path.join("lib64")
            }
            #[cfg(windows)]
            {
                cuda_path.join("lib").join("x64")
            }
        });

    let mut cpp_build_config = cpp_build::Config::new();
    cpp_build_config.include(cuda_include_path);
    #[cfg(not(windows))]
    cpp_build_config.include("/usr/local/tensorrt/include");
    cpp_build_config.build("src/lib.rs");

    println!("cargo:rustc-link-search={}", cuda_lib_path.display());
    #[cfg(not(windows))]
    println!("cargo:rustc-link-search=/usr/local/tensorrt/lib64");

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
}

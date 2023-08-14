#[cfg(unix)]
fn main() {
    cpp_build::Config::new()
        .include("/usr/local/cuda/include")
        .build("src/lib.rs");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
}

#[cfg(windows)]
fn main() {
    let cuda_path = std::env::var("CUDA_PATH").expect("Missing environment variable `CUDA_PATH`.");
    let cuda_path = std::path::Path::new(&cuda_path);
    cpp_build::Config::new()
        .include(cuda_path.join("include"))
        .build("src/lib.rs");
    println!(
        "cargo:rustc-link-search={}",
        cuda_path.join("lib").join("x64").display()
    );
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
}

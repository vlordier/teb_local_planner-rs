use std::path::Path;

fn main() {
    let src = Path::new("src/cpp");
    let cpp_src = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("cpp")
        .join("compare_api.cpp");
    
    println!("cargo:rerun-if-changed={}", src.display());
    
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(&cpp_src)
        // Add include paths to the original C++ headers
        .include("/Users/vincent/Work/teb_local_planner-rs/include")
        // We DON'T need to link the full C++ teb library — instead,
        // we will compile the minimal C++ comparison functions directly.
        // See compare_api.cpp below.
        .compile("nav_compare_cpp");
    
    println!("cargo:rustc-link-lib=static=nav_compare_cpp");
}

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    nav_fuzz::fuzz_normalize_angle(data);
    nav_fuzz::fuzz_pose_se2(data);
});

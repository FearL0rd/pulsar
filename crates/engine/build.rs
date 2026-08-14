// Stamp the build with the git sha so --version identifies exactly which
// commit a bug report is against (issue #21: reports arrive with a pasted
// commit hash or nothing at all). Absent git or a tarball build degrades
// to "unknown" rather than failing the build.
use std::process::Command;

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.is_empty());
    println!(
        "cargo:rustc-env=PULSAR_GIT_SHA={sha}{}",
        if dirty { "-dirty" } else { "" }
    );
    // Re-stamp when HEAD moves; without this the sha sticks at whatever
    // the first build saw.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}

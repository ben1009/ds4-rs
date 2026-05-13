//! Integrity check for committed reference vectors.
//!
//! Hashes each file under `tests/vectors/` and compares against
//! `manifest.toml`. A vector file that changes without a corresponding
//! manifest bump will fail CI.
//!
//! See rfcs/0002-forward-pass.md §4.2.

use std::{fs, path::PathBuf};

use sha2::{Digest, Sha256};

#[test]
fn manifest_matches_committed_vectors() {
    let manifest_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/manifest.toml");
    let raw = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));

    // Tiny hand-rolled parser for the one-key-per-entry subset we use here.
    // Avoids pulling in a TOML crate for a 20-line file.
    let entries = parse_manifest(&raw);
    assert!(
        !entries.is_empty(),
        "manifest.toml has no [[vectors]] entries",
    );

    let vectors_dir = manifest_path.parent().unwrap().to_path_buf();
    let mut checked: Vec<String> = Vec::new();

    for entry in &entries {
        let path = vectors_dir.join(&entry.path);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut h = Sha256::new();
        h.update(&bytes);
        let got = hex(&h.finalize());
        assert_eq!(
            got, entry.sha256,
            "{}: sha256 mismatch (got {got}, expected {})",
            entry.path, entry.sha256,
        );
        checked.push(entry.path.clone());
    }

    // Also catch committed vectors that aren't in the manifest.
    for entry in fs::read_dir(&vectors_dir).unwrap() {
        let e = entry.unwrap();
        let name = e.file_name().to_string_lossy().into_owned();
        if name == "manifest.toml" {
            continue;
        }
        assert!(
            checked.contains(&name),
            "{name} present under tests/vectors/ but missing from manifest.toml",
        );
    }
}

struct Entry {
    path: String,
    sha256: String,
}

fn parse_manifest(raw: &str) -> Vec<Entry> {
    let mut entries = Vec::new();
    let mut path: Option<String> = None;
    let mut sha: Option<String> = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed == "[[vectors]]" {
            if path.is_some() || sha.is_some() {
                let p = path.take().expect("path before next [[vectors]]");
                let s = sha.take().expect("sha256 before next [[vectors]]");
                entries.push(Entry { path: p, sha256: s });
            }
            continue;
        }
        if let Some(v) = trimmed.strip_prefix("path = ") {
            path = Some(unquote(v).to_string());
        } else if let Some(v) = trimmed.strip_prefix("sha256 = ") {
            sha = Some(unquote(v).to_string());
        }
    }
    if let (Some(p), Some(s)) = (path, sha) {
        entries.push(Entry { path: p, sha256: s });
    }
    entries
}

fn unquote(s: &str) -> &str {
    s.trim().trim_matches('"')
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

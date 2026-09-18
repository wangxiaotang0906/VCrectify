"""Create a source-only research release; never include datasets or upstream code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/vcevo-0.1.0-source.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directories = ["src", "tests", "configs", "docs", "scripts", "examples", ".github"]
    names = ["README.md", "README.zh-CN.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
             "CONTRIBUTING.md", "CITATION.cff", "pyproject.toml", "requirements-tested.txt",
             ".gitignore", ".env.example", "MANIFEST.in"]
    paths = [root / name for name in names if (root / name).is_file()]
    paths += [p for directory in directories for p in (root / directory).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and not any(part.endswith(".egg-info") for part in p.parts)
              and p.suffix not in {".pyc", ".pyo"}]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            manifest[name] = hashlib.sha256(payload).hexdigest()
            entry = zipfile.ZipInfo("vcevo/" + name, date_time=(2026, 9, 18, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
        entry = zipfile.ZipInfo("vcevo/SOURCE_MANIFEST.json", date_time=(2026, 9, 18, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"archive": str(output), "files": len(manifest),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

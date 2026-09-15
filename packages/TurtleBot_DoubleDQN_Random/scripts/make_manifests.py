import posixpath
#!/usr/bin/env python3
"""Write RELEASE_MANIFEST.sha256 (every file) and SHARED_LAYER_MANIFEST.sha256
(the common-layer files listed in SHARED_LAYER_FILES.txt). Run from the
package root after any change; verify with scripts/verify_package.sh."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav.identity import (  # noqa: E402
    RELEASE_MANIFEST,
    SHARED_LAYER_MANIFEST,
    sha256_file,
    release_files,
    shared_layer_manifest_lines,
)

def main() -> int:
    shared = shared_layer_manifest_lines(ROOT)
    with open(os.path.join(ROOT, SHARED_LAYER_MANIFEST), "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(shared) + "\n")
    lines = [f"{sha256_file(os.path.join(ROOT, rel))}  {rel}" for rel in release_files(ROOT)]
    with open(os.path.join(ROOT, RELEASE_MANIFEST), "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"{SHARED_LAYER_MANIFEST}: {len(shared)} files; {RELEASE_MANIFEST}: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

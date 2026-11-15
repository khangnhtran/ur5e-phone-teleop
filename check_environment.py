"""Read installed versions without installing or modifying anything."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

expected=json.loads((Path(__file__).parent/"environment-versions.json").read_text())["simulation"]
ok=sys.version_info[:2]==(3,8)
print("Python:",sys.version.split()[0],"(tested:",expected["python"]+")")
for package,version in expected.items():
    if package=="python":
        continue
    try:
        actual=metadata.version(package)
    except metadata.PackageNotFoundError:
        actual="MISSING"
    match=actual==version
    ok=ok and match
    print(("OK" if match else "DIFF"),package,actual,"expected",version)
if not ok:
    print("Version mismatch: check the environment; no packages were changed.")
raise SystemExit(0 if ok else 1)

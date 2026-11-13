# SPDX-License-Identifier: Apache-2.0
"""Install the verified Task archive during the image build only (Linux amd64)."""
import hashlib
import io
from pathlib import Path
import tarfile
from urllib.request import urlopen

url = "https://github.com/go-task/task/releases/download/v3.54.0/task_linux_amd64.tar.gz"
data = urlopen(url, timeout=60).read()
if hashlib.sha256(data).hexdigest() != "680859dbb4d881a9c72d4d9a8f510825450849af8567deacd7302c01124416fb":
    raise SystemExit("Task archive checksum mismatch")
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    target = Path(".cache/docs-bin/task")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.extractfile("task").read())
    target.chmod(0o755)

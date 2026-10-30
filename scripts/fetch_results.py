"""Download the recorded results of the study from the GitHub release ``results-v1`` into outputs/.

    python scripts/fetch_results.py
    python scripts/fetch_results.py --archive ten-tenths-results.tar.gz   # an archive already downloaded

The archive (90 MB) holds what the figures, the animations and the web demo are made from, so that they can be rebuilt
without training: the recorded laps (outputs/trajectories), the minimum-time laps (outputs/reference), the evaluations
at the limit (outputs/evaluation/limit, balance_limit, edge), the training curves of the policies at the limit
(outputs/logs), the steady-state analysis of the balance index (outputs/analysis) and the three policies of seed 12
that the live demo runs (outputs/models). Its SHA-256 is checked before it is unpacked; existing files are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/antoniooodev/ten-tenths/releases/download/results-v1/ten-tenths-results.tar.gz"
SHA256 = "19a87218a8354a91a923e9a5dfdb37cd5a570473c1834c4176ed236146c7ea57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(archive: Path, root: Path) -> int:
    """Extract the members under outputs/ (and not the AppleDouble files of macOS) into ``root``."""
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.parts[0] != "outputs" or name.name.startswith("._") or not (member.isfile() or member.isdir()):
                continue
            tar.extract(member, root, filter="data")
            count += member.isfile()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="a downloaded copy of the archive (otherwise it is fetched)")
    parser.add_argument("--root", type=Path, default=Path("."), help="directory that receives outputs/")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / "results.tar.gz"
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive)  # noqa: S310
        digest = sha256(archive)
        if digest != SHA256:
            raise SystemExit(f"{archive}: SHA-256 {digest} is not the one of results-v1 ({SHA256})")
        print(f"unpacked {unpack(archive, args.root)} files into {args.root / 'outputs'}")


if __name__ == "__main__":
    main()

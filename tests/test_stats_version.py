"""The pre-registration lock records labcore.__version__ as the version of the statistics code.
That is only worth something if the version moves whenever the code does."""

from labtestkit import REPO

import labcore
from labcore.kaggle.bundle import tree_hash

STATS_DIGESTS = {
    "0.1.0": "sha256:453574feaba4be13c210e0b6828c52d34e5d78a5b911cdf3e5d89cf5bbb40180",
    "0.2.0": "sha256:c283857d86b79ac4218d5474de24f2e14067afa79070058d0b912fc95ccd0819",
}


def test_statistics_code_matches_the_version():
    digest = tree_hash(REPO / "labcore" / "stats")
    assert STATS_DIGESTS.get(labcore.__version__) == digest, (
        "labcore/stats changed. That can change a gated number: bump __version__ in labcore/__init__.py "
        "(this breaks the lock of every direction that is L1_RUNNING) and record the new digest here: "
        f"{digest}"
    )

"""The repository itself: what git must never be offered to commit."""

import shutil
import subprocess

import pytest

from omnia_llm.config import ROOT


@pytest.mark.parametrize("path", ["api-key.txt", "tokens.json", "tokens.tmp", "ngrok.env",
                                  "hf-cache/models--x/blob", "logs/gateway.log"])
def test_a_leftover_of_the_old_layout_can_never_be_committed(path):
    """The old flat layout kept its key, tokens, tunnel address, downloads and logs at the
    root. A checkout updated in place can still hold them there."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert ignored.returncode == 0, f"{path} is not ignored"

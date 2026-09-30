"""The presented token, and the first version's single key."""

from __future__ import annotations

import stat

import pytest

from omnia_llm.server import auth


class TestTheFirstVersionsKey:
    def test_a_new_install_has_none(self, tmp_path):
        """Nothing is generated: every token a new install accepts is one somebody issued."""
        path = tmp_path / "api-key.txt"

        assert auth.read_legacy_key(path) == ""
        assert not path.exists()

    def test_an_existing_one_is_read(self, tmp_path):
        path = tmp_path / "api-key.txt"
        path.write_text("omnia-alreadyhere\n")

        assert auth.read_legacy_key(path) == "omnia-alreadyhere"

    def test_a_loose_existing_file_is_tightened(self, tmp_path):
        """On a machine other people share, a world-readable key file is no key at all."""
        path = tmp_path / "api-key.txt"
        path.write_text("omnia-alreadyhere\n")
        path.chmod(0o644)

        auth.read_legacy_key(path)

        assert not path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)

    def test_a_blank_file_is_no_key(self, tmp_path):
        # A truncated file must not become an empty key that then matches an empty header.
        path = tmp_path / "api-key.txt"
        path.write_text("   \n")

        assert auth.read_legacy_key(path) == ""

    def test_retiring_it_deletes_the_file_once(self, tmp_path):
        path = tmp_path / "api-key.txt"
        path.write_text("omnia-alreadyhere\n")

        assert auth.retire_legacy_key(path) and not path.exists()
        assert not auth.retire_legacy_key(path)


class TestReadingTheHeader:
    @pytest.mark.parametrize(
        "header", ["Bearer abc123", "bearer abc123", "  Bearer   abc123  "]
    )
    def test_it_reads_the_openai_style_header(self, header):
        """What Omnia's `openai_compatible` provider already sends for its api_key.

        That is the whole reason this shape was chosen: authentication with no add-on changes.
        """
        assert auth.presented_key(header, None) == "abc123"

    def test_it_also_accepts_x_api_key_for_curl(self):
        assert auth.presented_key(None, "abc123") == "abc123"

    @pytest.mark.parametrize("header", [None, "", "Basic abc123", "Bearer", "Bearer   "])
    def test_anything_else_presents_nothing(self, header):
        assert auth.presented_key(header, None) == ""


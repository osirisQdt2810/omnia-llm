"""The API key: generated once, stored tight, and required before anything costs a GPU."""

from __future__ import annotations

import stat

import pytest

from gateway import auth


class TestTheKeyFile:
    def test_it_generates_one_on_first_use(self, tmp_path):
        key = auth.load_or_create(tmp_path / "api-key.txt")

        assert key.startswith("omnia-") and len(key) > 30

    def test_it_is_stable_across_restarts(self, tmp_path):
        path = tmp_path / "api-key.txt"

        assert auth.load_or_create(path) == auth.load_or_create(path)

    def test_two_gateways_do_not_share_a_key(self, tmp_path):
        """No default, no shipped value, nothing to forget to change."""
        a = auth.load_or_create(tmp_path / "a.txt")
        b = auth.load_or_create(tmp_path / "b.txt")

        assert a != b

    def test_it_is_not_readable_by_anyone_else(self, tmp_path):
        """On a machine eight people share, a world-readable key file is no key at all."""
        path = tmp_path / "api-key.txt"

        auth.load_or_create(path)

        assert not path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)

    def test_a_loose_existing_file_is_tightened(self, tmp_path):
        path = tmp_path / "api-key.txt"
        path.write_text("omnia-alreadyhere\n")
        path.chmod(0o644)

        auth.load_or_create(path)

        assert not path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)

    def test_an_empty_file_is_replaced_rather_than_trusted(self, tmp_path):
        # A truncated file must not become an empty key that then matches an empty header.
        path = tmp_path / "api-key.txt"
        path.write_text("   \n")

        assert auth.load_or_create(path).startswith("omnia-")


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


class TestComparing:
    def test_the_right_key_matches(self):
        assert auth.matches("omnia-abc", "omnia-abc")

    def test_a_wrong_key_does_not(self):
        assert not auth.matches("omnia-abd", "omnia-abc")

    @pytest.mark.parametrize("pair", [("", "omnia-abc"), ("omnia-abc", ""), ("", "")])
    def test_empty_never_matches(self, pair):
        """Including empty-against-empty.

        A gateway that somehow had no key would otherwise accept every request that sent no
        key — open to everyone, and silently.
        """
        assert not auth.matches(*pair)

    def test_a_prefix_of_the_key_does_not_match(self):
        assert not auth.matches("omnia-", "omnia-abc")

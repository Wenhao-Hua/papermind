"""Config secret masking — `config show` must never reveal a full API key."""

from __future__ import annotations


def test_mask_hides_secret_middle():
    from papermind.config import _mask

    key = "sk-1234abcdefghijklmnopqrstuvwxyz5678"
    masked = _mask(key)
    assert masked == "sk-1...5678"        # only first4 ... last4 shown
    assert key not in masked              # the full secret never appears
    assert "abcdefgh" not in masked       # the middle is hidden
    assert _mask("12345678") == "****"        # <= 8 chars: fully hidden
    assert _mask("123456789") == "1234...6789"  # > 8 chars: ends shown, middle hidden


def test_masked_view_masks_only_sensitive_fields(monkeypatch, tmp_path):
    import papermind.config as cfg_mod
    from papermind.config import Config

    monkeypatch.setenv("PAPERMIND_HOME", str(tmp_path))  # isolate config_path / cache_root
    raw = "sk-deadbeefdeadbeefdeadbeef0000"
    monkeypatch.setattr(cfg_mod, "load_config",
                        lambda: Config(deepseek_key=raw, default_model="deepseek/deepseek-chat"))

    view = cfg_mod.masked_view()
    assert raw not in str(view["deepseek_key"]) and "..." in str(view["deepseek_key"])  # secret masked
    assert view["default_model"] == "deepseek/deepseek-chat"  # non-sensitive field untouched
    assert all(raw not in str(v) for v in view.values())      # the raw key leaks nowhere in the view

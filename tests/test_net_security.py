"""Security regression tests for the public ``papermind serve`` attack surface:
SSRF guard, download caps, upload sanitization, and rate-limit IP trust.

No real network: ``socket.getaddrinfo`` is monkeypatched so URL host checks are
deterministic and offline.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from papermind import net
from papermind.errors import SourceError


# --------------------------------------------------------------------------- #
# IP classification (pure, no network)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ip", [
    "127.0.0.1", "::1",                 # loopback
    "10.0.0.5", "192.168.1.10", "172.16.0.1",  # private
    "169.254.169.254",                  # link-local / cloud metadata
    "0.0.0.0",                          # unspecified
    "::ffff:127.0.0.1",                 # IPv4-mapped loopback
    "not-an-ip",                        # unparseable -> refuse
])
def test_blocked_ips(ip):
    assert net._ip_is_blocked(ip) is True


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_ips_allowed(ip):
    assert net._ip_is_blocked(ip) is False


# --------------------------------------------------------------------------- #
# assert_public_url
# --------------------------------------------------------------------------- #
def _fake_resolve(ip):
    def _getaddrinfo(host, port, *a, **k):
        return [(0, 0, 0, "", (ip, port))]
    return _getaddrinfo


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "gopher://h/1"])
def test_rejects_non_http_scheme(url):
    with pytest.raises(net.BlockedURLError):
        net.assert_public_url(url)


def test_rejects_missing_host():
    with pytest.raises(net.BlockedURLError):
        net.assert_public_url("http:///nohost")


def test_blocks_host_resolving_to_metadata(monkeypatch):
    monkeypatch.setattr(net.socket, "getaddrinfo", _fake_resolve("169.254.169.254"))
    with pytest.raises(net.BlockedURLError):
        net.assert_public_url("http://evil.example.com/")


def test_blocks_host_resolving_to_private(monkeypatch):
    monkeypatch.setattr(net.socket, "getaddrinfo", _fake_resolve("10.1.2.3"))
    with pytest.raises(net.BlockedURLError):
        net.assert_public_url("https://internal.corp/")


def test_allows_public_host(monkeypatch):
    monkeypatch.setattr(net.socket, "getaddrinfo", _fake_resolve("93.184.216.34"))
    net.assert_public_url("https://example.com/paper.pdf")  # must not raise


# --------------------------------------------------------------------------- #
# Upload sanitization (web._resolve_upload)
# --------------------------------------------------------------------------- #
class _FakeUpload:
    def __init__(self, filename, data):
        self.filename = filename
        self.file = io.BytesIO(data)


def test_upload_ignores_client_path_and_is_random():
    from pathlib import Path

    from papermind import web

    up = _FakeUpload("../../../../evil.pdf", b"%PDF-1.4 hello world")
    path = web._resolve_upload("", up)
    name = Path(path).name
    assert ".." not in path                      # no traversal survived
    assert name.startswith("papermind_") and name.endswith(".pdf")
    assert Path(path).read_bytes().startswith(b"%PDF")
    Path(path).unlink(missing_ok=True)


def test_upload_same_name_gets_distinct_paths():
    from pathlib import Path

    from papermind import web

    a = web._resolve_upload("", _FakeUpload("paper.pdf", b"%PDF-1.4 A"))
    b = web._resolve_upload("", _FakeUpload("paper.pdf", b"%PDF-1.4 B"))
    try:
        assert a != b  # no cross-user collision on a shared filename
    finally:
        Path(a).unlink(missing_ok=True)
        Path(b).unlink(missing_ok=True)


def test_upload_rejects_non_pdf():
    from papermind import web

    with pytest.raises(SourceError):
        web._resolve_upload("", _FakeUpload("x.pdf", b"<html>not a pdf</html>"))


# --------------------------------------------------------------------------- #
# Rate-limit client IP — forged forwarding headers must not be trusted
# --------------------------------------------------------------------------- #
class _FakeClient:
    host = "203.0.113.9"


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.client = _FakeClient()


def test_client_ip_ignores_forged_headers_by_default(monkeypatch):
    from papermind import web

    monkeypatch.setattr(web, "_TRUST_PROXY", False)
    req = _FakeRequest({"x-forwarded-for": "1.2.3.4", "cf-connecting-ip": "9.9.9.9"})
    assert web._client_ip(req) == "203.0.113.9"  # the real socket peer, not a header


def test_client_ip_trusts_only_cf_when_enabled(monkeypatch):
    from papermind import web

    monkeypatch.setattr(web, "_TRUST_PROXY", True)
    # cf header is honored (Cloudflare sets it)...
    req = _FakeRequest({"cf-connecting-ip": "9.9.9.9", "x-forwarded-for": "1.2.3.4"})
    assert web._client_ip(req) == "9.9.9.9"
    # ...but a spoofable XFF alone is still ignored.
    req2 = _FakeRequest({"x-forwarded-for": "1.2.3.4"})
    assert web._client_ip(req2) == "203.0.113.9"


# --------------------------------------------------------------------------- #
# PDF download: atomic write, size cap, PDF-magic check (safe_stream stubbed)
# --------------------------------------------------------------------------- #
def _fake_stream(chunks, headers=None):
    hdrs = headers or {}

    @contextlib.contextmanager
    def cm(method, url, **kw):
        class _Resp:
            def __init__(self):
                self.headers = hdrs

            def raise_for_status(self):
                pass

            def iter_bytes(self, chunk_size=65536):
                yield from chunks

        yield _Resp()

    return cm


def test_download_pdf_success_writes_atomically(monkeypatch, tmp_path):
    from papermind.parser.arxiv import _download_pdf

    monkeypatch.setattr(net, "safe_stream", _fake_stream([b"%PDF-1.4 ", b"rest of pdf"]))
    dest = tmp_path / "paper.pdf"
    _download_pdf("https://example.com/y.pdf", dest)
    assert dest.read_bytes() == b"%PDF-1.4 rest of pdf"
    assert not (tmp_path / "paper.pdf.part").exists()  # temp renamed away, nothing left behind


def test_download_pdf_non_pdf_leaves_no_file(monkeypatch, tmp_path):
    # A truncated/garbage download must NOT leave a paper.pdf that the size>0 / %PDF
    # reuse checks would accept and feed to the parser.
    from papermind.parser.arxiv import _download_pdf

    monkeypatch.setattr(net, "safe_stream", _fake_stream([b"<html>not a pdf</html>"]))
    dest = tmp_path / "paper.pdf"
    with pytest.raises(SourceError):
        _download_pdf("https://example.com/y.pdf", dest)
    assert not dest.exists() and not (tmp_path / "paper.pdf.part").exists()


def test_download_pdf_oversize_rejected(monkeypatch, tmp_path):
    from papermind.parser.arxiv import _download_pdf

    big = b"%PDF-1.4 " + b"x" * (net.MAX_DOWNLOAD_BYTES + 100)
    monkeypatch.setattr(net, "safe_stream", _fake_stream([big]))
    dest = tmp_path / "paper.pdf"
    with pytest.raises(SourceError):
        _download_pdf("https://example.com/y.pdf", dest)
    assert not dest.exists()

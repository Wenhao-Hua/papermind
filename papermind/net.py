"""Network safety helpers shared by anything that fetches a user-supplied URL.

A public ``papermind serve`` lets anonymous users hand us arbitrary URLs (PDF
links, and image URLs returned by a model). Without guards, the server can be
tricked into fetching internal addresses (SSRF — e.g. the cloud metadata
endpoint ``169.254.169.254``) or downloading unbounded data. These helpers:

  * reject non-http(s) URLs and any host that resolves to a private / loopback /
    link-local / reserved IP, and re-check every redirect hop (so a public URL
    can't 30x-bounce to an internal one);
  * cap the download size.

Residual risk: there is a tiny TOCTOU window between the DNS check here and the
actual socket connect (DNS rebinding). Pinning the resolved IP into the
connection would close it; for this tool that residual risk is acceptable and
left as a known limitation.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
from urllib.parse import urlparse

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # papers are a few MB; this is generous slack


class BlockedURLError(ValueError):
    """A URL was refused (bad scheme, unresolvable host, or a private/internal IP)."""


def _ip_is_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable -> refuse
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:127.0.0.1 -> judge the embedded IPv4
        addr = mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local  # includes 169.254.169.254 cloud metadata
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def assert_public_url(url: str) -> None:
    """Raise :class:`BlockedURLError` unless ``url`` is http(s) and every address
    its host resolves to is a public, routable IP."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise BlockedURLError(f"只允许 http/https 链接：{url}")
    host = parsed.hostname
    if not host:
        raise BlockedURLError(f"URL 缺少主机名：{url}")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURLError(f"无法解析主机 {host}：{exc}") from exc
    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            raise BlockedURLError(f"拒绝访问内网/保留地址（{host} → {ip}）。")


@contextlib.contextmanager
def safe_stream(method: str, url: str, *, headers=None, timeout: float = 60.0, max_redirects: int = 5):
    """Yield an httpx streaming ``Response`` for ``url``, validating the host and
    every redirect hop with :func:`assert_public_url`. The caller iterates
    ``resp.iter_bytes(...)`` and is responsible for size limits.

    Redirects are followed manually (the client has ``follow_redirects=False``)
    so each new location is re-validated — auto-following would let a public URL
    redirect to an internal one unchecked.
    """
    import httpx

    headers = headers or {}
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        current = url
        for _ in range(max_redirects + 1):
            assert_public_url(current)
            resp = client.send(client.build_request(method, current, headers=headers), stream=True)
            if resp.is_redirect:
                location = resp.headers.get("location", "")
                base = resp.url
                resp.close()
                if not location:
                    raise BlockedURLError(f"重定向缺少目标：{current}")
                current = str(base.join(location))
                continue
            try:
                yield resp
            finally:
                resp.close()
            return
        raise BlockedURLError(f"重定向次数过多：{url}")

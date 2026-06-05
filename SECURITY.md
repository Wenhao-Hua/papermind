# Security Policy

PaperMind ships a public web service (`papermind serve --live`) that accepts
anonymous, attacker-controlled input — paper URLs, PDF uploads, and questions. Its
hardening is treated as a first-class concern: the server has SSRF guards (every
fetched URL and redirect hop is re-validated against private/reserved IP ranges),
upload sanitization, per-IP + global rate limiting, download size caps, and
SVG-injection stripping.

## Reporting a vulnerability

Please report security issues **privately**, not in a public issue or PR:

- open a [GitHub Security Advisory](https://github.com/Wenhao-Hua/papermind/security/advisories/new), or
- contact the maintainers via the repository profile.

Include reproduction steps and the affected version or commit. We aim to acknowledge
within a few days and will keep you posted on the fix and disclosure timeline.

## Scope

**In scope:** the `papermind serve` web service, the source-resolution and download
paths (SSRF / unbounded download), rate-limit handling, and API-key/secret handling.

**Out of scope:** issues that require a malicious local `~/.papermind/config.json`, a
self-hosted model endpoint you control, or running an untrusted `--local` model.

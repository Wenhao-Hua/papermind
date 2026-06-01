# Releasing PaperMind

Releases are published to [PyPI](https://pypi.org/project/paper-mind/) automatically by
the [`Release`](.github/workflows/release.yml) workflow when a `vX.Y.Z` tag is pushed.

## One-time setup

1. Create the project on PyPI (or reserve the name).
2. Configure **Trusted Publishing** (no API token needed): on PyPI →
   *Manage project* → *Publishing* → add a GitHub Actions publisher for
   `Wenhao-Hua/papermind`, workflow `release.yml`, environment `pypi`.
3. In the GitHub repo, create an Environment named `pypi`.

## Cutting a release

1. Bump the version in [`pyproject.toml`](pyproject.toml) and
   `papermind/__init__.py` (`__version__`). Keep them in sync.
2. Update any changelog notes, commit.
3. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. The `Release` workflow builds the sdist + wheel and publishes to PyPI.
5. Verify: `pip install paper-mind` in a clean environment.

## Pre-release checklist

- [ ] `pytest` passes locally and in CI.
- [ ] `pip install -e .` works from a fresh clone.
- [ ] `papermind --help`, `papermind config show` work.
- [ ] README install/usage commands are accurate.
- [ ] Version bumped in both `pyproject.toml` and `__init__.py`.

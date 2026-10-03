# Contributing to gpupool

> English. Tiếng Việt: [CONTRIBUTING.vi.md](CONTRIBUTING.vi.md)

Thanks for helping. This guide is short on purpose: match what is already there, prove your change works, keep the
docs in both languages in sync.

By contributing you agree that your work is released under the [MIT License](LICENSE). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md). For security problems, do not open a public issue: see
[SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/longduongbao29/multi-gpu-inference.git
cd multi-gpu-inference
uv sync                 # creates .venv with runtime and dev dependencies from uv.lock
uv run pytest -q        # about 810 tests, no GPU, no llama.cpp, no network needed
```

If `python` is not on your PATH, always go through `uv run python ...`.

### Tests that need real hardware

Tests marked `real` need llama.cpp binaries, a GGUF file and a GPU. They are skipped by default
(`addopts = "-m 'not real'"`). Run them with:

```bash
uv run pytest -m real
```

### End-to-end test with real images

`scripts/ci_e2e.py` starts a real coordinator and 3 CPU agents from the Docker images, serves a tiny model split over
RPC and (unless `--skip-convert`) converts a Hugging Face model. It needs Docker and, for the conversion stage, the
coordinator image built with `WITH_CONVERT=1` and access to huggingface.co. It is what CI runs; run it locally for
anything that touches images, llama.cpp or the multi-server path.

### UI work

The UI is plain HTML/CSS and [Alpine.js](https://alpinejs.dev/) (vendored under `src/gpupool/ui/vendor/`), with **no
build step**. Do not add a bundler, a framework or an npm dependency. For UI changes run the mock server:

```bash
uv run python scripts/ui_mock_server.py   # http://127.0.0.1:8090, admin key: dev
```

## Code style

- Match the surrounding code: naming, structure, typing, error handling. There is no formatter to run; read the
  neighbouring files first.
- **Comments are in English only.** Explain *why* for anything non-obvious (a threshold, a workaround, a flag order,
  an invariant), not what the line does. If a comment records a real failure or measurement, say so.
- Keep dependencies light. Do not add a heavy dependency (ML frameworks, large web stacks, build tooling) without
  discussing it in an issue first. Anything new that ships must also be listed in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (both languages).
- Do not swallow errors, write the source of truth non-atomically, or leave processes, files or ports behind on
  failure.

## Tests

- **Every bug fix comes with a regression test** that fails without the fix.
- New behaviour comes with tests. Tests must not depend on, or write into, the working directory (use `tmp_path`);
  the suite should pass from an empty directory.
- Anything that touches llama.cpp, the Dockerfiles, the agent's engine launch or the RPC path must also be run for
  real (a real binary, a real image, `scripts/ci_e2e.py`). Mocks prove the mocks work: say in the pull request what
  you ran against reality, and what you could not.
- `uv run pytest -q` must be green before you open a pull request.

## Documentation

Every document exists in English and Vietnamese: `X.md` and `X.vi.md` (or `X.en.md` and `X.vi.md` under `docs/`),
each linking to the other on line 3. Change one, change the other in the same pull request. Vietnamese should read
naturally and use full diacritics. The only exception is `LICENSE`, which is English only. Code comments stay English.

Update [CHANGELOG.md](CHANGELOG.md) (and `CHANGELOG.vi.md`) under *Unreleased* for any user-visible change.

## Commit messages

Say what changed and **why**. For a defect, record: what broke, under what conditions, and what now prevents it.

```
Create the data folder before opening the database

The conversion job manager opened <data>/coordinator.db without creating
<data>/, so create_app failed on a fresh checkout. The manager now creates
the folder like the store does; a regression test fails without the fix.
```

Small, focused commits are easier to review than one large one.

## Pull requests

1. Open an issue first for anything large or that changes behaviour, so we agree on the direction.
2. Branch from `master`, make the change, run `uv run pytest -q`, and (when relevant) the real runs above.
3. Fill in the pull request template: what and why, how you tested it, docs updated in both languages.
4. Keep the pull request to one concern. CI (unit tests, image builds, the end-to-end job) must pass.
5. Expect review comments; they are about the code, not about you.

## Where to ask

- Bugs and feature ideas: [GitHub Issues](https://github.com/longduongbao29/multi-gpu-inference/issues) using the
  templates.
- Questions: open an issue and label it `question`.
- Security: [SECURITY.md](SECURITY.md).

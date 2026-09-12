# Contributing

1. Create a clean environment and install `requirements.txt`.
2. Keep credentials and non-redistributable seed data out of the repository.
3. Run `python -m compileall -q .`, `python -m unittest discover -s tests -v`, and `ruff check .` before opening a pull request.
4. Preserve the paper's reverse-synthesis order and terminal dual-verifier requirement unless the paper and documentation are updated together.
5. Add tests for changes to validation, quota allocation, renderer restrictions, or output schemas.
6. By submitting a contribution, you agree to license that contribution under the [Apache License 2.0](LICENSE).

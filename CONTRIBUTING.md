# Contributing

Thanks for your interest in contributing to `eseg`!

## Development Environment
1. Fork & clone the repo
2. Create a virtual environment
3. Install in editable mode with extras:
```bash
pip install -e .[dev,viewer]
```

## Code Style
- Formatting: `black`
- Linting: `ruff`
- Type checking: `mypy` (not strict yet)

Run all:
```bash
ruff check .
black --check .
mypy src
pytest
```

## Pull Requests
- Use feature branches
- Add/Update docstrings for new public functions
- Include tests where possible

## Releasing
1. Bump version in `pyproject.toml`
2. Tag release: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
3. Build & upload:
```bash
python -m build
python -m twine upload dist/*
```

## License
By contributing you agree your code will be released under the MIT License.

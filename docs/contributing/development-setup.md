# Development Setup

Set up ChimeraLM for local development and contribution.

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/ChimeraLM.git
cd ChimeraLM

# 2. Install dependencies with uv
uv sync

# 3. Install pre-commit hooks (optional)
uv run pre-commit install

# 4. Verify installation
uv run chimeralm --version
```

## Prerequisites

- **Python**: 3.10 or 3.11
- **uv**: Fast Python package manager ([install](https://github.com/astral-sh/uv))
- **Git**: Version control
- **Optional**: CUDA-compatible GPU for testing GPU features

## Development Environment

### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/ylab-hi/ChimeraLM.git
cd ChimeraLM

# Install all dependencies (including dev dependencies)
uv sync

# Run commands with uv
uv run chimeralm --help
uv run pytest tests/
```

### Using pip + venv

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev,docs]"

# Verify
chimeralm --version
```

## Project Structure

```
ChimeraLM/
├── chimeralm/              # Main package
│   ├── __main__.py        # CLI entry point
│   ├── models/            # Model implementations
│   ├── data/              # Data loading
│   └── utils/             # Utilities
├── tests/                  # Test suite
│   ├── data/              # Test data
│   └── test_*.py          # Test files
├── docs/                   # Documentation (MkDocs)
├── src/                    # Rust source code
├── configs/                # Hydra configuration
├── pyproject.toml         # Project metadata
└── README.md              # Project README
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Edit code
vim chimeralm/models/new_model.py

# Run tests
uv run pytest tests/

# Run linting
uv run ruff check .
uv run ruff format .
```

### 3. Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_models.py

# Run with coverage
uv run pytest tests/ --cov=chimeralm --cov-report=html
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new model architecture"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Follow the PR template
```

## Code Quality

### Linting and Formatting

ChimeraLM uses **ruff** for linting and formatting:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
# Run mypy (if configured)
uv run mypy chimeralm/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# All tests
uv run pytest tests/

# Specific marker
uv run pytest tests/ -m "not slow"

# With coverage
uv run pytest tests/ --cov=chimeralm

# Verbose output
uv run pytest tests/ -v
```

### Writing Tests

```python
# tests/test_new_feature.py
import pytest
from chimeralm.models.lm import ChimeraLM

def test_model_loading():
    """Test model loads correctly."""
    model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
    assert model is not None

def test_prediction():
    """Test prediction works."""
    model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
    # Add test logic
```

## Building Documentation

```bash
# Build docs
uv run mkdocs build

# Serve locally
uv run mkdocs serve

# Open browser to http://127.0.0.1:8000
```

## Building Rust Extensions

```bash
# Build Rust extensions
maturin develop --release

# Run tests
cargo nextest run
```

## Troubleshooting

### uv sync fails

```bash
# Try with --reinstall
uv sync --reinstall

# Or clear cache
rm -rf .venv
uv sync
```

### Import errors

```bash
# Ensure package is installed in editable mode
uv pip install -e .
```

### GPU tests fail

```bash
# Skip GPU tests on CPU machines
uv run pytest tests/ -k "not gpu"
```

## See Also

- [Testing Guide](testing.md)
- [Adding Features](adding-features.md)
- [Code of Conduct](https://github.com/ylab-hi/ChimeraLM/blob/main/CODE_OF_CONDUCT.md)

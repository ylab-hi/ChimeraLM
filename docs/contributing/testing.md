# Testing Guide

ChimeraLM testing strategy, conventions, and best practices.

## Test Framework

ChimeraLM uses **pytest** for testing with these key features:

- **Fixtures**: Reusable test data and setups
- **Markers**: Categorize tests (slow, gpu, integration)
- **Coverage**: Minimum 40% coverage required
- **Parametrization**: Test multiple inputs efficiently

## Running Tests

### Basic Usage

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test function
uv run pytest tests/test_models.py::test_model_loading

# Verbose output
uv run pytest tests/ -v

# Stop on first failure
uv run pytest tests/ -x
```

### Test Markers

```bash
# Skip slow tests
uv run pytest tests/ -k "not slow"

# Run only GPU tests
uv run pytest tests/ -m gpu

# Run smoke tests only
uv run pytest tests/ -m smoke

# Run import tests
uv run pytest tests/ -m imports
```

### Coverage

```bash
# Run with coverage
uv run pytest tests/ --cov=chimeralm

# Generate HTML report
uv run pytest tests/ --cov=chimeralm --cov-report=html

# View report
open htmlcov/index.html
```

## Test Structure

### Directory Layout

```
tests/
├── data/                    # Test data files
│   └── mk1c_test.bam  # Sample BAM file
├── test_models.py          # Model tests
├── test_data.py            # Data loading tests
├── test_cli.py             # CLI tests
├── test_utils.py           # Utility tests
└── conftest.py             # Shared fixtures
```

### Test File Naming

- Prefix with `test_`
- Match module name: `chimeralm/models/lm.py` → `tests/test_models.py`
- One test file per module (generally)

### Test Function Naming

```python
# Good naming
def test_model_loads_pretrained_checkpoint():
    """Test that model loads from pretrained checkpoint."""
    pass

def test_prediction_returns_correct_shape():
    """Test prediction output shape."""
    pass

# Bad naming
def test_model():  # Too vague
    pass

def test1():  # Not descriptive
    pass
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from chimeralm.models.lm import ChimeraLM

def test_model_loading():
    """Test model loads successfully."""
    # Arrange
    model_path = "yangliz5/chimeralm"

    # Act
    model = ChimeraLM.from_pretrained(model_path)

    # Assert
    assert model is not None
    assert isinstance(model, ChimeraLM)
```

### Using Fixtures

```python
# conftest.py
import pytest
from chimeralm.models.lm import ChimeraLM

@pytest.fixture
def pretrained_model():
    """Load pretrained model for testing."""
    return ChimeraLM.from_pretrained("yangliz5/chimeralm")

# test_models.py
def test_model_inference(pretrained_model):
    """Test model inference."""
    import torch
    x = torch.randint(0, 5, (4, 1024))
    output = pretrained_model(x)
    assert output.shape == (4, 2)
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("batch_size,seq_len", [
    (4, 512),
    (8, 1024),
    (16, 2048),
])
def test_model_with_different_shapes(pretrained_model, batch_size, seq_len):
    """Test model with various input shapes."""
    import torch
    x = torch.randint(0, 5, (batch_size, seq_len))
    output = pretrained_model(x)
    assert output.shape == (batch_size, 2)
```

### Testing Exceptions

```python
import pytest

def test_invalid_input_raises_error():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError):
        # Code that should raise ValueError
        process_invalid_input()
```

## Test Categories

### Unit Tests

Test individual functions or classes in isolation:

```python
def test_tokenizer_encode():
    """Test DNA tokenizer encoding."""
    from chimeralm.data.tokenizer import DNATokenizer

    tokenizer = DNATokenizer()
    sequence = "ACGT"
    tokens = tokenizer.encode(sequence)

    assert tokens == [1, 2, 3, 4]
```

### Integration Tests

Test multiple components working together:

```python
@pytest.mark.slow
def test_end_to_end_prediction():
    """Test complete prediction pipeline."""
    from chimeralm.models.lm import ChimeraLM
    from chimeralm.data.bam import BamDataModule

    # Load model
    model = ChimeraLM.from_pretrained("yangliz5/chimeralm")

    # Load data
    data_module = BamDataModule(
        train_data_path="tests/data/mk1c_test.bam",
        batch_size=8
    )
    data_module.setup("predict")

    # Run prediction
    loader = data_module.predict_dataloader()
    batch = next(iter(loader))
    output = model(batch["input_ids"])

    assert output.shape[0] == 8
    assert output.shape[1] == 2
```

### GPU Tests

```python
import pytest
import torch

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_gpu():
    """Test model runs on GPU."""
    model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
    model = model.cuda()

    x = torch.randint(0, 5, (4, 1024)).cuda()
    output = model(x)

    assert output.device.type == "cuda"
```

## Coverage Requirements

- **Minimum**: 40% overall coverage
- **Target**: 60%+ for new code
- **Critical paths**: 80%+ coverage

### Checking Coverage

```bash
# Run with coverage
uv run pytest tests/ --cov=chimeralm --cov-report=term-missing

# Output shows uncovered lines
chimeralm/models/lm.py    85%   12, 45-47
chimeralm/data/bam.py     92%   67
```

### Improving Coverage

Focus on:
1. **Critical paths**: Model loading, prediction, training
2. **Error handling**: Exception cases
3. **Edge cases**: Empty inputs, max values, etc.

## Mocking and Fixtures

### Mocking External Dependencies

```python
import pytest
from unittest.mock import Mock, patch

def test_bam_file_loading(tmp_path):
    """Test BAM file loading with mock."""
    # Create temporary BAM file
    bam_file = tmp_path / "test.bam"
    bam_file.write_text("mock data")

    with patch("pysam.AlignmentFile") as mock_bam:
        mock_bam.return_value = Mock()
        # Test code
```

### Temporary Files

```python
def test_with_temp_file(tmp_path):
    """Test using temporary file."""
    # tmp_path is a pytest fixture
    test_file = tmp_path / "test.txt"
    test_file.write_text("test data")

    # Use test_file
    assert test_file.read_text() == "test data"
```

## Continuous Integration

Tests run automatically on:

- **Pull requests**: All tests must pass
- **Main branch**: After merge
- **Nightly**: Full test suite including slow tests

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: uv run pytest tests/ --cov=chimeralm
```

## Best Practices

### DO

- ✅ Write tests for new features
- ✅ Test both success and failure cases
- ✅ Use descriptive test names
- ✅ Keep tests independent (no shared state)
- ✅ Use fixtures for common setups
- ✅ Test edge cases

### DON'T

- ❌ Skip tests without good reason
- ❌ Test implementation details (test behavior)
- ❌ Write flaky tests (non-deterministic)
- ❌ Mock everything (test real code when possible)
- ❌ Ignore coverage warnings

## Debugging Tests

### Running Single Test with Debug Output

```bash
# Run with print statements visible
uv run pytest tests/test_models.py::test_model_loading -s

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Show local variables on failure
uv run pytest tests/ -l
```

### Using pytest's Debug Mode

```python
def test_with_debug():
    """Test with debugging."""
    import pdb; pdb.set_trace()  # Debugger breakpoint
    # Test code
```

## See Also

- [Development Setup](development-setup.md)
- [Adding Features](adding-features.md)
- [CI/CD Workflows](https://github.com/ylab-hi/ChimeraLM/tree/main/.github/workflows)

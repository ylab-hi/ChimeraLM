# Adding Features

Guide for contributing new features to ChimeraLM.

## Contribution Workflow

1. **Discuss** - Open an issue to discuss the feature
2. **Fork** - Fork the repository
3. **Branch** - Create a feature branch
4. **Implement** - Write code and tests
5. **Test** - Ensure tests pass
6. **Document** - Update documentation
7. **Submit** - Create pull request

## Feature Planning

### Before Starting

- [ ] Check existing issues/PRs for similar features
- [ ] Open an issue to discuss the feature
- [ ] Get feedback from maintainers
- [ ] Agree on design approach

### Design Considerations

- **Scope**: Keep features focused and small
- **API**: Maintain consistency with existing patterns
- **Performance**: Consider impact on speed/memory
- **Backward compatibility**: Don't break existing APIs

## Implementation Steps

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Write Tests First (TDD)

```python
# tests/test_new_feature.py
def test_new_feature():
    """Test new feature works as expected."""
    # Arrange
    ...
    # Act
    result = new_feature()
    # Assert
    assert result == expected
```

### 3. Implement Feature

```python
# chimeralm/new_module.py
def new_feature():
    """
    Brief description of what this function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
        >>> result = new_feature()
        >>> print(result)
    """
    pass
```

### 4. Update Documentation

```markdown
# docs/reference/new-feature.md

## New Feature

Description and usage examples...
```

### 5. Add to CHANGELOG

```markdown
## [Unreleased]

### Added
- New feature: description (#PR_NUMBER)
```

## Code Standards

### Python Style

- **Formatting**: Use ruff
- **Line length**: 120 characters
- **Docstrings**: Google style
- **Type hints**: Required for public APIs

**Example:**

```python
def predict_batch(
    sequences: List[str],
    model: torch.nn.Module,
    batch_size: int = 32
) -> List[int]:
    """
    Predict labels for a batch of sequences.

    Args:
        sequences: List of DNA sequences
        model: Trained model
        batch_size: Number of sequences per batch

    Returns:
        List of predicted labels (0 or 1)

    Example:
        >>> sequences = ["ACGT", "GGCC"]
        >>> labels = predict_batch(sequences, model)
        >>> print(labels)  # [0, 1]
    """
    pass
```

### Testing Requirements

- **Coverage**: New code must have 60%+ coverage
- **Test types**: Unit + integration tests
- **Edge cases**: Test boundary conditions
- **No flaky tests**: Tests must be deterministic

## Pull Request Checklist

Before submitting:

- [ ] Tests pass locally (`uv run pytest tests/`)
- [ ] Code is formatted (`uv run ruff format .`)
- [ ] No linting errors (`uv run ruff check .`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages are clear
- [ ] PR description explains changes

## Review Process

1. **Automated checks**: CI tests must pass
2. **Code review**: Maintainer reviews code
3. **Feedback**: Address review comments
4. **Approval**: Get approval from maintainer
5. **Merge**: Maintainer merges PR

### What Reviewers Look For

- Code quality and readability
- Test coverage and quality
- Documentation completeness
- Performance implications
- Breaking changes

## Common Patterns

### Adding a New Model

```python
# chimeralm/models/components/new_model.py
from torch import nn

class NewModel(nn.Module):
    """New model architecture."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Initialize layers

    def forward(self, x):
        """Forward pass."""
        # Implement forward pass
        return output
```

### Adding a New Data Module

```python
# chimeralm/data/new_format.py
import lightning as L

class NewFormatDataModule(L.LightningDataModule):
    """Data module for new format."""

    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: str):
        """Setup datasets."""
        pass

    def train_dataloader(self):
        """Return training dataloader."""
        pass
```

### Adding a New CLI Command

```python
# chimeralm/__main__.py
import typer

@app.command()
def new_command(
    input_file: str = typer.Argument(..., help="Input file"),
    output_dir: str = typer.Option("output/", help="Output directory")
):
    """
    New command description.
    """
    # Implement command logic
    pass
```

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproducible example
- **Feature requests**: Open an issue with use case

## See Also

- [Development Setup](development-setup.md)
- [Testing Guide](testing.md)
- [Code of Conduct](https://github.com/ylab-hi/ChimeraLM/blob/main/CODE_OF_CONDUCT.md)

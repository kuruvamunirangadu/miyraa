# Testing Quick Reference Card

## Quick Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing_expanded.py -v

# Run specific test class
pytest tests/test_preprocessing_expanded.py::TestNormalization -v

# Run specific test
pytest tests/test_preprocessing_expanded.py::TestNormalization::test_normalize_unicode -v

# Run tests matching pattern
pytest tests/ -k "normalize" -v

# Run with markers (if defined)
pytest tests/ -m "slow" -v
```

## Linting Commands

```bash
# Flake8 - all files
flake8 src/ scripts/ tests/

# Flake8 - strict (fail on errors)
flake8 src/ --select=E9,F63,F7,F82

# MyPy - all files
mypy src/ --config-file mypy.ini

# MyPy - specific module
mypy src/api/ --config-file mypy.ini

# Auto-fix formatting
autopep8 --in-place --aggressive --aggressive <file>
```

## Test Files Summary

| File | Tests | Focus |
|------|-------|-------|
| `test_preprocessing_expanded.py` | 32 | Text normalization, label mapping |
| `test_losses_expanded.py` | 22 | Loss functions, thresholds |
| `test_api_integration.py` | 21 | API endpoints, error handling |
| `test_onnx_integration.py` | 11 | ONNX loading, inference |
| `test_regression.py` | 14 | Accuracy monitoring |
| **TOTAL** | **78** | |

## CI/CD Jobs

| Job | When | Duration | Blocking |
|-----|------|----------|----------|
| test (lightweight) | Every commit | ~2 min | Yes |
| test (cpu-torch) | Every commit | ~5 min | Yes |
| regression | Pull requests | ~3 min | No |
| integration | Every commit | ~2 min | Yes |
| onnx-smoke | Every commit | ~2 min | Yes |

## Coverage Goals

- **Overall**: 85%+
- **API**: 95%+
- **Preprocessing**: 95%+
- **Training**: 80%+
- **Scripts**: 60%+

## Common Test Patterns

```python
# Basic assertion
def test_something():
    result = function()
    assert result == expected

# Test exception
def test_error():
    with pytest.raises(ValueError):
        function_that_raises()

# Parametrize
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_upper(input, expected):
    assert input.upper() == expected

# Skip test
@pytest.mark.skip(reason="Not implemented")
def test_future():
    pass

# Mock
from unittest.mock import patch

@patch('module.function')
def test_with_mock(mock_func):
    mock_func.return_value = 42
    result = code_using_function()
    assert result == 42
```

## Flake8 Error Codes

- **E**: PEP 8 style errors
- **F**: PyFlakes errors (undefined names, unused imports)
- **W**: Warnings
- **C90**: Cyclomatic complexity
- **E9**: Runtime errors (syntax)
- **F63**: Invalid print
- **F7**: Syntax error
- **F82**: Undefined name

## MyPy Common Issues

```python
# Missing type hints
def func(x: int) -> str:  # Add types
    return str(x)

# Ignore line
result = func()  # type: ignore

# Ignore file
# type: ignore  # At top of file

# Cast
from typing import cast
x = cast(int, some_value)
```

## Debugging Failed Tests

```bash
# Show full traceback
pytest tests/test_file.py -v --tb=long

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show print statements
pytest tests/ -s

# Verbose mode
pytest tests/ -vv
```

## Pre-Commit Checklist

- [ ] Run `pytest tests/ -v` - All tests pass
- [ ] Run `flake8 src/ scripts/` - No linting errors
- [ ] Run `mypy src/` - No type errors (or acceptable)
- [ ] Check coverage: `pytest --cov=src --cov-report=term`
- [ ] Update tests if adding new features
- [ ] Update docs if changing APIs
- [ ] Commit message follows convention

## Coverage Report

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report (open htmlcov/index.html)
pytest --cov=src --cov-report=html

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

## Useful Pytest Flags

- `-v` : Verbose output
- `-vv` : Extra verbose
- `-s` : Show print statements
- `-x` : Stop on first failure
- `--tb=short` : Short traceback
- `--tb=long` : Full traceback
- `--tb=no` : No traceback
- `-k "pattern"` : Run tests matching pattern
- `--lf` : Run last failed tests
- `--ff` : Run failed first, then others
- `--pdb` : Drop into debugger on failure
- `-n auto` : Parallel execution (requires pytest-xdist)

---

**Quick Start:**
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code quality
flake8 src/ && mypy src/

# Full CI simulation
pytest tests/ -v --cov=src && flake8 src/ && mypy src/
```

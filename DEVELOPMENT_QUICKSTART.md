# 🚀 FUSION Development Quick Start Guide

**New to FUSION development?** This guide gets you productive in 5 minutes!

## ✅ Prerequisites

- Python 3.11+ installed
- Git repository cloned
- Virtual environment activated

## 🎯 1-Minute Setup

```bash
# Install all development tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks (prevents bad commits)
make setup-hooks

# Verify installation
make help
```

## 📝 Daily Development Workflow

### Step 1: Format Your Code
```bash
# Auto-format all code before committing
make format
```
**What it does**: Fixes code style with black + isort automatically

### Step 2: Check for Issues
```bash
# Fast linting and type checking
make lint-new
```
**What it does**: Finds bugs, style issues, and type problems with ruff + mypy

### Step 3: Run Tests
```bash
# Run tests with coverage
make test-new
```
**What it does**: Runs pytest with coverage reporting

### Step 4: Full Quality Check
```bash
# Run everything before submitting PR
make check-all
```
**What it does**: Combines format + lint + test + analysis

## 🔧 Essential Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `make format` | Auto-fix code style | Before every commit |
| `make lint-new` | Check for issues | During development |
| `make test-new` | Run tests | After code changes |
| `make check-all` | Full quality check | Before PR submission |
| `make analyze` | Generate reports | Weekly/monthly analysis |
| `make profile` | Performance analysis | When optimizing |

## 🚨 Pre-commit Hooks (Automatic)

Pre-commit hooks run automatically when you commit:

- ✅ **Code formatting** (black, isort)
- ✅ **Linting** (ruff)
- ✅ **Type checking** (mypy)
- ✅ **Security scanning** (bandit)
- ✅ **Commit message validation** (conventional commits)

**If hooks fail**: Fix the issues and commit again.

## 📊 Understanding Tool Output

### Ruff Output
```
fusion/utils/config.py:42:89 E501 Line too long (94 > 88)
```
**Fix**: Shorten the line or use line breaks

### MyPy Output
```
fusion/utils/config.py:32: error: Returning Any from function
```
**Fix**: Add proper type annotations

### Test Coverage
```
TOTAL coverage: 85%
```
**Goal**: Maintain >80% coverage

## 📁 Generated Reports

After running `make check-all`, check these locations:

```bash
# View coverage in browser
open reports/coverage/htmlcov/index.html

# View dependency graphs
open reports/diagrams/architecture_overview.png

# View performance profile
snakeviz reports/profiling/profile.prof
```

## 🚫 Common Issues & Solutions

### "Command not found"
```bash
# Install missing tools
pip install -r requirements-dev.txt
```

### "Pre-commit hook failed"
```bash
# Fix the issue and retry
git commit --amend
```

### "Type checking errors"
```bash
# Add type hints or ignore specific lines
variable: str = "example"  # Add type hint
line_to_ignore  # type: ignore  # Ignore if needed
```

### "Tests failing"
```bash
# Run specific test for debugging
pytest tests/test_specific.py -v
```

## ⚡ Pro Tips

1. **Use `make format` frequently** - Prevents style issues
2. **Run `make lint-new` while coding** - Catches issues early
3. **Check `make test-new` after changes** - Ensures nothing breaks
4. **Use `make check-all` before PRs** - Comprehensive validation
5. **Review reports in `reports/`** - Understand code quality trends

## 🆘 Getting Help

- **Makefile commands**: `make help`
- **Detailed workflow**: See `DEVELOPMENT_WORKFLOW.md`
- **Coding standards**: See `CODING_STANDARDS.md`
- **Reports guide**: See `reports/README.md`
- **Tool docs**: Each tool has official documentation online

## 🎓 Learning More

### Advanced Usage
- **Custom analysis**: Edit scripts in `scripts/`
- **Configuration tuning**: Modify `pyproject.toml`, `mypy.ini`
- **CI/CD integration**: Check `.github/workflows/quality.yml`

### Understanding the Tools
- **Black**: Code formatter (opinionated, consistent)
- **Ruff**: Modern linter (fast, replaces many tools)
- **MyPy**: Type checker (catches type-related bugs)
- **Pytest**: Test runner (with coverage reporting)
- **Pre-commit**: Git hooks (automated quality checks)

---

**🎉 You're ready to contribute high-quality code to FUSION!**

Run `make check-all` and ensure everything passes before your first PR.

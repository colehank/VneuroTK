```markdown
# VneuroTK Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the VneuroTK Python codebase. You'll learn about file naming, import/export styles, commit message conventions, and how to write and organize tests. These guidelines ensure consistency and maintainability across the project.

## Coding Conventions

### File Naming
- Use **camelCase** for file names.
  - Example: `dataLoader.py`, `signalProcessor.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import preprocessData
    from .models import NeuralNetwork
    ```

### Export Style
- Use **named exports** (explicitly listing what is exported).
  - Example:
    ```python
    __all__ = ['preprocessData', 'NeuralNetwork']
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use the `docs` prefix for documentation changes.
  - Example:
    ```
    docs: update README with installation instructions
    ```

## Workflows

### Documentation Updates
**Trigger:** When making changes to documentation files.
**Command:** `/update-docs`

1. Edit the relevant documentation file(s).
2. Use a conventional commit message with the `docs` prefix.
   - Example: `docs: add usage examples to SKILL.md`
3. Push your changes to the repository.

## Testing Patterns

- Test files use the pattern `*.test.*` (e.g., `dataLoader.test.py`).
- The specific testing framework is **unknown**; check existing test files for structure.
- Place test files alongside the code they test or in a dedicated test directory.

**Example test file name:**
```
signalProcessor.test.py
```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /update-docs   | Standardize and commit documentation updates |
```

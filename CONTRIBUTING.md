# Contributing to Mycology Research Pipeline

First off, thank you for considering contributing to Mycology Research Pipeline! It's people like you that make this project such a great tool for the mycology research community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@mycologyresearch.com.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- A GitHub account
- Basic knowledge of Flask and SQLAlchemy

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork locally**
   ```bash
   git clone https://github.com/your-username/MycologyResearchPipeline.git
   cd MycologyResearchPipeline
   ```

3. **Add the upstream repository**
   ```bash
   git remote add upstream https://github.com/original/MycologyResearchPipeline.git
   ```

4. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment details (OS, Python version, etc.)

**Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python version: [e.g. 3.12.0]
 - Browser: [e.g. Chrome 120]
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Why this enhancement would be useful
- Possible implementation approach

### Your First Code Contribution

Unsure where to begin? Look for these tags in our issues:

- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

## Development Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.

# Run linting
flake8 .
black --check .
mypy .
```

### 4. Commit Your Changes

See [Commit Messages](#commit-messages) section for guidelines.

## Style Guidelines

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for function parameters and return values
- Docstrings for all public functions/classes (Google style)

**Example:**
```python
from typing import List, Optional

def analyze_sample(
    image_path: str,
    confidence_threshold: float = 0.7,
    databases: Optional[List[str]] = None
) -> dict:
    """Analyze a mushroom sample image.
    
    Args:
        image_path: Path to the image file.
        confidence_threshold: Minimum confidence for predictions.
        databases: List of databases to query. Defaults to all.
        
    Returns:
        Dictionary containing analysis results.
        
    Raises:
        ValueError: If image_path is invalid.
    """
    # Implementation here
    pass
```

### JavaScript Style Guide

- Use ES6+ features
- Semicolons required
- 2 spaces for indentation
- Use `const` and `let`, avoid `var`

### CSS/SCSS Style Guide

- Use BEM naming convention
- Mobile-first approach
- CSS variables for theming

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(auth): add two-factor authentication

Implement TOTP-based 2FA for user accounts. Users can now enable
2FA from their account settings.

Closes #123
```

```
fix(api): handle missing API key gracefully

Return proper error message when API key is missing instead of
throwing unhandled exception.

Fixes #456
```

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure all checks pass

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Updated documentation
   
   ## Screenshots (if applicable)
   
   ## Related Issues
   Closes #(issue number)
   ```

5. **Code Review**
   - Address reviewer feedback
   - Keep discussions focused and professional
   - Update PR based on feedback

## Testing Guidelines

### Writing Tests

- Test file naming: `test_<module_name>.py`
- Use pytest fixtures for common setup
- Aim for >80% code coverage
- Test edge cases and error conditions

**Example:**
```python
import pytest
from app import create_app
from models import User

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

def test_user_registration(client):
    """Test user registration endpoint."""
    response = client.post('/api/auth/register', json={
        'email': 'test@example.com',
        'password': 'SecurePass123!',
        'name': 'Test User'
    })
    
    assert response.status_code == 201
    assert 'access_token' in response.json
```

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Complex algorithms should have inline comments
- Update README.md for significant changes

### API Documentation

- Use OpenAPI/Swagger specifications
- Include request/response examples
- Document error responses

## Community

### Getting Help

- Discord: [Join our server](https://discord.gg/mycology)
- Discussions: Use GitHub Discussions for questions
- Email: dev@mycologyresearch.com

### Recognition

Contributors who make significant contributions will be:
- Added to our Contributors list
- Mentioned in release notes
- Invited to our private contributors channel

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Mycology Research Pipeline! 🍄 
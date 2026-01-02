# GitHub Actions CI/CD Workflows

This directory contains the CI/CD workflows for the Street Parking Analyzer project.

## Workflows

### 1. `ci.yml` - Main CI Pipeline
The main CI pipeline that orchestrates the entire testing process. It:
- Detects which parts of the codebase changed
- Triggers backend and/or frontend CI based on changes
- Runs integration checks
- Provides overall CI status

**Triggers:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### 2. `backend-ci.yml` - Backend CI
Tests and validates the Python backend (FastAPI + YOLO + MongoDB).

**Jobs:**
- **Lint and Test**: Runs on Python 3.10, 3.11, and 3.12
  - Code formatting check with Black
  - Import sorting check with isort
  - Linting with flake8
  - Unit tests with pytest
  - Coverage reporting
  - Import verification

- **Security Scan**:
  - Dependency vulnerability scanning with Safety
  - Security linting with Bandit

**Triggers:**
- Changes to `backend/**` directory
- Changes to `.github/workflows/backend-ci.yml`

### 3. `frontend-ci.yml` - Frontend CI
Tests and validates the React frontend.

**Jobs:**
- **Lint and Build**: Runs on Node.js 18, 20, and 21
  - ESLint code quality checks
  - Prettier formatting checks
  - Unit tests (if configured)
  - Production build
  - Build artifact upload

- **Type Check**:
  - TypeScript type checking (if configured)

- **Security Scan**:
  - npm audit for vulnerabilities
  - Outdated dependency checks

**Triggers:**
- Changes to `frontend/**` directory
- Changes to `.github/workflows/frontend-ci.yml`

## Running Workflows Locally

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run formatting
black app/
isort app/

# Run linting
flake8 app/

# Run tests
pytest tests/ -v --cov=app

# Security checks
safety check --file=requirements.txt
bandit -r app/
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run linting
npm run lint

# Check formatting
npx prettier --check "src/**/*.{js,jsx,ts,tsx,json,css,scss,md}"

# Run tests
npm test

# Build
npm run build

# Security checks
npm audit
```

## Configuration Files

### Backend
- `backend/pyproject.toml` - Configuration for Black, isort, pytest, and coverage
- `backend/.flake8` - Flake8 linting rules
- `backend/requirements.txt` - Python dependencies (includes dev dependencies)

### Frontend
- `frontend/.prettierrc` - Prettier formatting rules
- `frontend/.prettierignore` - Files to ignore for Prettier
- `frontend/package.json` - Contains lint script configuration

## CI Status Badges

Add these badges to your README.md:

```markdown
![CI Pipeline](https://github.com/GuiSchet/street-parking-analyzer/workflows/CI%20Pipeline/badge.svg)
![Backend CI](https://github.com/GuiSchet/street-parking-analyzer/workflows/Backend%20CI/badge.svg)
![Frontend CI](https://github.com/GuiSchet/street-parking-analyzer/workflows/Frontend%20CI/badge.svg)
```

## Troubleshooting

### Common Issues

1. **Python import errors**: Make sure all `__init__.py` files exist
2. **Node.js build failures**: Clear `node_modules` and reinstall
3. **Linting failures**: Run formatters locally before pushing
4. **Test failures**: Ensure MongoDB is not required for unit tests (use mocking)

### Skipping CI

To skip CI on a commit, add `[skip ci]` to the commit message:

```bash
git commit -m "docs: update README [skip ci]"
```

## Future Enhancements

- [ ] Add integration tests with MongoDB
- [ ] Add E2E tests for frontend
- [ ] Docker image building and pushing
- [ ] Deployment workflows
- [ ] Performance testing
- [ ] Code coverage requirements

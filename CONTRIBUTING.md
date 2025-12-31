# Contributing to KV-Cache Inference

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/your-username/kv-cache-inference.git
   cd kv-cache-inference
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # or .venv/Scripts/activate on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file
   cp .env.example .env
   # Edit .env and add your HF_TOKEN
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 style guide for Python code
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

### Testing

Before submitting a PR, ensure all tests pass:

```bash
# Run benchmark tests
python benchmark/clean_prefix_test.py
python benchmark/test_eviction.py
python benchmark/test_concurrent_kv_cache.py

# Test Docker setup
docker-compose up -d
python test_docker.py
docker-compose down
```

### Commit Messages

Write clear, descriptive commit messages:

```
feat: add semantic caching support
fix: resolve memory leak in cache eviction
docs: update Docker deployment guide
test: add benchmark for concurrent requests
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

## 🎯 What to Contribute

### Ideas for Contributions

- **Features**
  - Semantic caching (similar prefix matching)
  - Distributed cache (Redis backend)
  - Additional eviction policies (LFU, ARC)
  - Batch inference optimization
  - Streaming response support
  
- **Improvements**
  - Performance optimizations
  - Better error handling
  - Enhanced monitoring/metrics
  - Documentation improvements
  
- **Bug Fixes**
  - Memory leaks
  - Edge cases
  - Platform-specific issues

### Areas Needing Help

- 📊 Benchmarking on different hardware (GPU, CPU)
- 🐛 Testing on Windows and Linux
- 📚 Documentation improvements
- 🎨 Adding visualization tools for cache statistics
- 🔧 Adding configuration management

## 📋 Pull Request Process

1. **Update documentation**
   - Add docstrings to new functions
   - Update README.md if adding features
   - Add examples if applicable

2. **Test your changes**
   - Run existing tests
   - Add new tests for new features
   - Verify Docker build works

3. **Submit PR**
   - Push to your fork
   - Create PR with clear description
   - Link any related issues

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] All existing tests pass
   - [ ] Added new tests for changes
   - [ ] Tested on multiple platforms
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes
   ```

## 🐛 Reporting Issues

When reporting bugs, include:

1. **Environment**
   - OS (Mac, Linux, Windows)
   - Python version
   - PyTorch version
   - Device (CPU, CUDA, MPS)

2. **Steps to reproduce**
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Logs**
   - Error messages
   - Stack traces
   - Relevant logs

## 💡 Feature Requests

For feature requests:

1. Check existing issues first
2. Describe the use case
3. Explain why it's valuable
4. Suggest implementation approach (optional)

## 📞 Questions?

- Open a GitHub Issue for bugs or features
- Start a Discussion for questions
- Check existing documentation first

## 🎉 Recognition

Contributors will be:
- Listed in release notes
- Mentioned in README acknowledgments
- Added to CONTRIBUTORS file (if created)

Thank you for making this project better! 🚀

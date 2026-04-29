# RAG Tester - AI Assistant Guide

## Project Overview

**Purpose:** Testing and evaluating Retrieval-Augmented Generation (RAG) systems  
**Tech Stack:** Python 3.8+  
**Structure:** Standard Python package with setuptools/pyproject.toml

## Key Commands

```bash
# Setup
pip install -r requirements.txt
pip install -e .

# Development
pip install -e ".[dev]"
pytest                    # Run tests
black .                   # Format code
flake8 .                  # Lint code
mypy .                    # Type checking
```

## Project Structure

```
rag-tester/
├── rag_tester/          # Main package
│   └── __init__.py
├── tests/               # Test files (to be created)
├── README.md            # User documentation
├── pyproject.toml       # Project configuration
├── requirements.txt     # Dependencies
└── CLAUDE.md           # This file
```

## Essential Conventions

- **Code Style:** Black formatter (100 char line length)
- **Type Hints:** Required (enforced by mypy)
- **Python Version:** 3.8+ compatibility
- **Package Name:** `rag_tester` (underscore, not hyphen)

## Documentation Index

*No additional documentation files yet. Create `.agent_docs/` directory and topic-specific files as the project grows.*

## Current Status

- Initial project structure created
- Git repository initialized
- Ready for first commit and GitHub push

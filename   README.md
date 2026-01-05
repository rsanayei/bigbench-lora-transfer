# bblt

**bblt** is a small project for training and evaluating **LoRA adapters** on selected **BIG-bench** tasks.

The project fine-tunes separate adapters on:
- `metaphor_boolean`
- `implicatures`

and evaluates them (along with the base model) on:
- `identify_odd_metaphor`

The goal is to analyze cross-task generalization and performance differences between adapters.

---

## Setup

This project requires **Python 3.10+**.

```bash
pip install -e .
````

For development tools (formatting, linting, tests):

```bash
pip install -e ".[dev]"
```

---

## Project Structure (WIP)

```text
.
├── src/            # Training and evaluation code
├── configs/        # YAML experiment configs
├── scripts/        # Entry-point scripts
├── tests/          # Unit tests
├── pyproject.toml
└── README.md
```

---

## Status

This repository is under active development.
Initial focus is on reproducible LoRA training and evaluation pipelines.

---
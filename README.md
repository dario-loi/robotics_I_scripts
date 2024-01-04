# Robotics I Utilities

This repository contains a collection of utilities for the Robotics I exam at the AI & Robotics master's degree at Sapienza University of Rome.

## Table of Contents

- [Robotics I Utilities](#robotics-i-utilities)
  - [Table of Contents](#table-of-contents)
  - [What's inside](#whats-inside)
  - [Installation](#installation)
  - [Usage](#usage)
    - [`robots`](#robots)
    - [`pyrobots`](#pyrobots)
  - [Tests](#tests)
  - [Contributing](#contributing)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## What's inside

- `robots.py` (also aliased as `robots`), a CLI tool for solving common kinematic problems. It is a wrapper around the `pyrobots` package that allows you to quickly solve problems without having to write any code. Most of the commands accept math expressions as input, so you can use it as a calculator as well. (e.g. `./robots rotation-direct "2 * pi + pi/4" "x"` performs a rotation of 2π + π/4 radians around the x axis)
- `pyrobots` - A Python package containing the utilities as functions and classes. The functions perform direct and inverse derivation for common kinematic problems that come up during the exam. Can't find the right command in `robots`? import this package in a notebook and write your own code!

## Installation

The project uses `pdm` as a build backend and package manager. Check out the [pdm documentation](https://pdm.fming.dev/) for more information on how to install it.

Once you have `pdm` installed, you can clone this repository and install the dependencies with:

```bash
git clone
cd robotics_I_scripts
pdm install && pdm run pre-commit install
```

This project is distributed through this repository, so remember to `git pull` once in a while to get the latest version.

## Usage

### `robots`

The `robots` command makes use of the powerful `typer` library to provide a CLI interface. You can get a comprehensive list of all the available commands and their options by running `./robots --help`.

### `pyrobots`

Every function and class in the `pyrobots` package is documented with docstrings. You can use the `help` function in Python to get more information on how to use them, your IDE should also be able to provide you with autocompletion and documentation.

## Tests

Having a library fail you during an exam is not fun, so I try to keep the code as bug-free as possible. The `tests` directory contains a collection of tests that check the correctness of the functions in the `pyrobots` package. You can run them with:

```bash
pytest -vv --mypy test_*.py
```

It should be all green!

If you want to be of help to this project but you don't want to touch the application itself, you can still write tests for the functions in `pyrobots` and open a pull request! This will help everyone else in the future.

## Contributing

This project is Open Source Software and is coded by students for students.

If you use the script, I only ask you to please report any bugs on the repository's [issue tracker](https://github.com/dario-loi/robotics_I_scripts/issues), if you want to contribute to the project, do not hesitate to open a pull request!

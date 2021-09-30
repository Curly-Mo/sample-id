# Sample ID

[![Build Status](https://app.travis-ci.com/Curly-Mo/sample-id.svg?branch=master)](https://app.travis-ci.com/Curly-Mo/sample-id)
[![Coverage](https://coveralls.io/repos/github/Curly-Mo/sample-id/badge.svg)](https://coveralls.io/github/Curly-Mo/sample-id)
[![Documentation](https://readthedocs.org/projects/sample-id/badge/?version=latest)](https://sample-id.readthedocs.org/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/sample-id.svg)](https://pypi.python.org/pypi/sample-id)
[![PyPI Pythons](https://img.shields.io/pypi/pyversions/sample-id.svg)](https://pypi.python.org/pypi/sample-id)
[![License](https://img.shields.io/pypi/l/sample-id.svg)](https://github.com/Curly-Mo/sample-id/blob/master/LICENSE)

Acoustic fingerprinting for Sample Identification

## Features

* What it do?

## Usage

* TODO

## Install

```console
pip install sample-id
```

## Documentation
See https://sample-id.readthedocs.org/en/latest/

## Development
```console
pip install poetry
cd sample-id
poetry install
```
### Run
To run cli entrypoint:
```console
poetry run sample_id --help
```

### Tests
```console
poetry run tox
```

### Docker
To run with docker
```console
docker build -t sample_id .
docker run sample_id:latest sample_id --help
```

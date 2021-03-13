"""
test_cli
----------------------------------
Tests for `cli` module.
"""
import pytest
from click.testing import CliRunner

from sample_id import cli

# pylint doesn't like pytest-fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture()
def runner():
    return CliRunner()


def test_cli(runner):
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0

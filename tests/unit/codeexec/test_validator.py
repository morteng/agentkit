import pytest

from agentkit.codeexec.errors import CodeValidationError
from agentkit.codeexec.validator import validate_source


def test_allows_plain_code():
    validate_source("x = 1\nfor i in range(3):\n    print(i)\nreturn x")


def test_rejects_import():
    with pytest.raises(CodeValidationError, match="import"):
        validate_source("import os")


def test_rejects_from_import():
    with pytest.raises(CodeValidationError, match="import"):
        validate_source("from os import system")


@pytest.mark.parametrize(
    "name", ["eval", "exec", "compile", "__import__", "open", "getattr", "globals"]
)
def test_rejects_forbidden_names(name):
    with pytest.raises(CodeValidationError):
        validate_source(f"{name}('x')")


def test_rejects_dunder_attribute_escape():
    with pytest.raises(CodeValidationError, match="dunder"):
        validate_source("().__class__.__bases__[0].__subclasses__()")


def test_rejects_syntax_error_as_validation_error():
    with pytest.raises(CodeValidationError, match="syntax"):
        validate_source("def (:")

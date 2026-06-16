import pytest

from agentkit.codeexec.errors import CodeValidationError
from agentkit.codeexec.namespace import SAFE_BUILTIN_NAMES
from agentkit.codeexec.validator import FORBIDDEN_NAMES, validate_source


def test_denylist_and_namespace_allowlist_are_disjoint():
    """The two gates must agree: a name the runtime namespace exposes must not
    be one the parse-time validator forbids (that mismatch silently breaks every
    script that uses it, e.g. `type` was denied while being a normal builtin)."""
    overlap = set(SAFE_BUILTIN_NAMES) & FORBIDDEN_NAMES
    assert not overlap, f"names exposed by namespace but forbidden by validator: {sorted(overlap)}"


def test_type_now_allowed_but_dunder_traversal_still_blocked():
    validate_source("t = type(1)\nreturn t is int")
    with pytest.raises(CodeValidationError, match="dunder"):
        validate_source("return type(1).__bases__")


def test_allows_plain_code():
    validate_source("x = 1\nfor i in range(3):\n    print(i)\nreturn x")


def test_rejects_import():
    with pytest.raises(CodeValidationError, match="import"):
        validate_source("import os")


def test_rejects_from_import():
    with pytest.raises(CodeValidationError, match="import"):
        validate_source("from os import system")


def test_import_error_message_guides_toward_prebound_modules():
    # The message must steer a model away from retrying `import` and toward
    # using host-provided modules directly, so a failed first attempt
    # self-corrects instead of falling back to manual computation.
    with pytest.raises(CodeValidationError, match="without import"):
        validate_source("import math")


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


def test_rejects_bare_builtins_name():
    with pytest.raises(CodeValidationError, match="dunder name"):
        validate_source("__builtins__['__import__']('os').system('id')")


def test_rejects_bare_dunder_names():
    for name in ("__builtins__", "__loader__", "__spec__"):
        with pytest.raises(CodeValidationError, match="dunder name"):
            validate_source(f"x = {name}")


def test_allows_dunder_free_script():
    # A benign script with no dunders should validate cleanly.
    validate_source("x = pikkolo.list_content()\nprint(x)")

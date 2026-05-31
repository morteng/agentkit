import types

from agentkit.codeexec.namespace import SAFE_MODULES, StdoutBuffer, build_safe_builtins


def test_safe_builtins_has_common_safe_names():
    b = build_safe_builtins(StdoutBuffer(max_bytes=1024))
    for name in ["len", "range", "sorted", "sum", "list", "dict", "str", "int", "print"]:
        assert name in b


def test_safe_builtins_excludes_dangerous_names():
    b = build_safe_builtins(StdoutBuffer(max_bytes=1024))
    for name in ["eval", "exec", "open", "__import__", "getattr", "globals"]:
        assert name not in b


def test_print_writes_to_buffer():
    buf = StdoutBuffer(max_bytes=1024)
    b = build_safe_builtins(buf)
    b["print"]("hello", "world")
    assert buf.getvalue() == "hello world\n"


def test_buffer_truncates_at_limit():
    buf = StdoutBuffer(max_bytes=10)
    b = build_safe_builtins(buf)
    b["print"]("x" * 100)
    assert len(buf.getvalue()) <= 64  # truncation marker allowed
    assert "truncated" in buf.getvalue()


def test_safe_modules_includes_pure_compute_stdlib():
    for name in [
        "math",
        "statistics",
        "datetime",
        "json",
        "decimal",
        "itertools",
        "collections",
        "re",
    ]:
        assert name in SAFE_MODULES, f"{name} should be a pre-bound safe module"


def test_safe_modules_excludes_io_process_and_introspection():
    for name in ["os", "sys", "subprocess", "importlib", "pathlib", "socket", "builtins"]:
        assert name not in SAFE_MODULES, f"{name} must not be exposed to scripts"


def test_safe_modules_values_are_modules_named_after_their_key():
    for name, mod in SAFE_MODULES.items():
        assert isinstance(mod, types.ModuleType)
        assert mod.__name__ == name

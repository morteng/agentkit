from agentkit.codeexec.namespace import StdoutBuffer, build_safe_builtins


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

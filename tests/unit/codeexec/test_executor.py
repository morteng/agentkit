import asyncio

from agentkit.codeexec.executor import execute
from agentkit.codeexec.limits import ExecLimits


async def test_runs_plain_script_and_captures_print():
    res = await execute({}, "print('hi')\nx = 2 + 3\nprint(x)")
    assert res.stdout == "hi\n5\n"
    assert res.error is None


async def test_returns_value():
    res = await execute({}, "return 41 + 1")
    assert res.return_value == 42


async def test_awaits_injected_async_function():
    async def fetch():
        return [1, 2, 3]

    res = await execute(
        {"fetch": fetch}, "rows = await fetch()\nprint(len(rows))\nreturn sum(rows)"
    )
    assert res.stdout == "3\n"
    assert res.return_value == 6


async def test_validation_error_surfaces():
    res = await execute({}, "import os")
    assert res.error is not None
    assert res.error_type == "CodeValidationError"


async def test_runtime_error_is_captured_not_raised():
    res = await execute({}, "x = 1 / 0")
    assert res.error_type == "ZeroDivisionError"
    assert res.return_value is None


async def test_timeout():
    async def slow():
        await asyncio.sleep(5)

    res = await execute({"slow": slow}, "await slow()", ExecLimits(wall_clock_s=0.1))
    assert res.error_type == "CodeTimeoutError"

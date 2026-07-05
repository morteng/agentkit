"""scrub_request deep-copy non-mutation across all text-bearing surfaces."""

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._messages import MessageRole
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import PiiPolicy
from agentkit.providers.base import ProviderRequest, SystemBlock

from .conftest import FakeDetector, FakeTokenMap, make_msg


def _fw(detector: FakeDetector) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy())


def _req() -> ProviderRequest:
    return ProviderRequest(
        model="openai/gpt-5.5",
        system=[SystemBlock(text="System note about Kari Nordmann")],
        messages=[
            make_msg(MessageRole.USER, TextBlock(text="email kari@example.no please")),
            make_msg(
                MessageRole.ASSISTANT,
                ToolUseBlock(
                    id="t1",
                    name="lookup",
                    arguments={"q": "Ola Nordmann", "n": 3, "nested": {"e": "c@d.no"}},
                ),
            ),
            make_msg(
                MessageRole.TOOL,
                ToolResultBlock(
                    tool_use_id="t1",
                    content=[TextBlock(text="found kari@example.no")],
                ),
            ),
        ],
    )


def test_scrub_request_covers_all_surfaces(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    req = _req()
    out = fw.scrub_request(req, tmap)

    # System.
    assert "Kari Nordmann" not in out.system[0].text
    # User message text.
    assert "kari@example.no" not in out.messages[0].content[0].text  # type: ignore[union-attr]
    # Tool-use args (string values, nested).
    tu = out.messages[1].content[0]
    assert isinstance(tu, ToolUseBlock)
    assert "Ola Nordmann" not in tu.arguments["q"]
    assert tu.arguments["n"] == 3  # non-string untouched
    assert "c@d.no" not in tu.arguments["nested"]["e"]
    # Tool-result inner text.
    tr = out.messages[2].content[0]
    assert isinstance(tr, ToolResultBlock)
    assert "kari@example.no" not in tr.content[0].text  # type: ignore[union-attr]


def test_scrub_request_never_mutates_input(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    req = _req()
    _ = fw.scrub_request(req, tmap)
    # Original request untouched.
    assert req.system[0].text == "System note about Kari Nordmann"
    assert req.messages[0].content[0].text == "email kari@example.no please"  # type: ignore[union-attr]
    tu = req.messages[1].content[0]
    assert isinstance(tu, ToolUseBlock)
    assert tu.arguments["q"] == "Ola Nordmann"
    assert tu.arguments["nested"]["e"] == "c@d.no"


def test_scrub_request_leaves_tools_untouched(detector: FakeDetector, tmap: FakeTokenMap):
    from agentkit.providers.base import ToolDefinition

    fw = _fw(detector)
    req = _req()
    req.tools = [
        ToolDefinition(
            name="lookup",
            description="find Kari Nordmann records",
            parameters={"type": "object"},
        )
    ]
    out = fw.scrub_request(req, tmap)
    # Static tool schemas are NOT scrubbed.
    assert out.tools[0].description == "find Kari Nordmann records"

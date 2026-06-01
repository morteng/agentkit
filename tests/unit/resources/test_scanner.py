from agentkit.resources import ApprovalScanner, OpRegistry, OpSpec, Reversibility


def _classify_status(static_kwargs, dynamic_args):
    if "status" in dynamic_args:
        return Reversibility.GATED
    if static_kwargs.get("status") in {"published", "archived"}:
        return Reversibility.GATED
    return Reversibility.REVERSIBLE


def _classify_delete(static_kwargs, dynamic_args):
    return Reversibility.GATED


def _reg():
    async def _apply(ctx, **kw):
        return {}

    reg = OpRegistry()
    reg.register(OpSpec(name="content.search", apply=_apply, is_read=True))
    reg.register(OpSpec(name="content.patch", apply=_apply, classify=_classify_status))
    reg.register(OpSpec(name="content.delete", apply=_apply, classify=_classify_delete))
    return reg


def _scan(src):
    return ApprovalScanner(client_var="pikkolo").scan(src, _reg())


def test_reads_only_no_approval():
    src = "rows = await pikkolo.content.search('x')\nprint(len(rows))"
    c = _scan(src)
    assert c.findings == []
    assert c.requires_approval is False


def test_reversible_patch_no_approval():
    src = "await pikkolo.content.patch(cid, tags=['needs-photo'])"
    c = _scan(src)
    assert c.worst is Reversibility.REVERSIBLE
    assert c.requires_approval is False


def test_static_publish_requires_approval():
    src = "await pikkolo.content.patch(cid, status='published')"
    c = _scan(src)
    assert c.worst is Reversibility.GATED
    assert c.requires_approval is True


def test_dynamic_status_requires_approval():
    src = "await pikkolo.content.patch(cid, status=target)"
    c = _scan(src)
    assert c.worst is Reversibility.GATED


def test_delete_requires_approval():
    src = "await pikkolo.content.delete(cid)"
    c = _scan(src)
    assert c.requires_approval is True


def test_loop_with_mixed_calls_worst_wins():
    src = (
        "for c in drafts:\n"
        "    await pikkolo.content.patch(c.id, tags=['x'])\n"
        "await pikkolo.content.delete(bad_id)\n"
    )
    c = _scan(src)
    assert len(c.findings) == 2
    assert c.requires_approval is True


def test_ignores_non_client_calls():
    src = "x = len([1, 2, 3])\nawait other.thing.patch(1, status='published')"
    c = _scan(src)
    assert c.findings == []

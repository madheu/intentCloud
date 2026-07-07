from core.models import (
    ConsciousFrame,
    Constraint,
    ErrorCode,
    FrameModality,
    FrameType,
    ImmutableKernel,
    Identity,
    IntentLayer,
    IntentNode,
    SAFE_FALLBACK_BLUEPRINT,
    SafetyVerdict,
    SilenceState,
)


def test_error_code_values():
    assert ErrorCode.SAFE_FALLBACK == "SAFE_FALLBACK"
    assert ErrorCode.KERNEL_IMMUTABLE == "KERNEL_IMMUTABLE"


def test_conscious_frame_silence():
    frame = ConsciousFrame.silence(
        reason="test silence",
        reason_code=ErrorCode.SAFE_FALLBACK,
        tick_id=7,
    )
    assert frame.modality == FrameModality.SILENCE
    assert frame.frame_type == FrameType.SILENCE
    assert frame.tick_id == 7
    assert isinstance(frame.data, SilenceState)
    assert frame.data.reason_code == ErrorCode.SAFE_FALLBACK


def test_conscious_frame_fallback():
    frame = ConsciousFrame.fallback(
        ErrorCode.MODULE_EXCEPTION,
        payload={"func": "foo"},
        tick_id=3,
    )
    assert frame.frame_type == FrameType.FALLBACK
    assert frame.data.reason_code == ErrorCode.MODULE_EXCEPTION


def test_intent_node_corroboration_guard():
    node = IntentNode(id="i1", text="test")
    assert node.trust == 0.5
    # 直接设为 0.7 以上不会触发保护，但通过 corroborate 提升会检查次数
    node.trust = 0.75
    node.corroborate(delta=0.1)
    # 仅 1 次印证，不允许继续提升
    assert node.trust == 0.75
    node.corroborate(delta=0.1)
    # 第 2 次印证后才允许提升
    assert node.trust == 0.85


def test_intent_node_decay():
    node = IntentNode(id="i2", text="test", strength=1.0)
    node.decay(lambda_=0.1, dt=1.0)
    assert node.strength == 0.9


def test_immutable_kernel_mutation_verdict():
    kernel = ImmutableKernel(
        identity=Identity(name="IntCog-R", role="assistant", immutable=True),
        safety_constraints=[Constraint(id="c1", text="do no harm")],
    )
    verdict = kernel.with_attempted_mutation()
    assert verdict.blocked is True
    assert ErrorCode.KERNEL_IMMUTABLE in verdict.reason_codes


def test_safety_verdict_block_chain():
    v = SafetyVerdict()
    v2 = v.block("a", ErrorCode.INPUT_BLOCKED).block("b", ErrorCode.INPUT_TOO_LONG)
    assert not v2.pass_
    assert v2.reason_codes == [ErrorCode.INPUT_BLOCKED, ErrorCode.INPUT_TOO_LONG]


def test_safe_fallback_blueprint_immutable_fields():
    fb = SAFE_FALLBACK_BLUEPRINT
    assert fb.reason_code == ErrorCode.SAFE_FALLBACK
    assert fb.type == "fallback"

from pathlib import Path
import pytest

from strata_control.gguf_dialects import (
    PRISM_DIALECTS,
    TURBOQUANT_DIALECTS,
    DialectType,
    compute_tensor_bytes,
    get_dialect_descriptor,
)
from strata_control.profiles import SPIN8_MODES
from strata_control.session_identity import SessionIdentity, validate_session_resume


def test_spin8_modes_and_session_invalidation():
    """Verify all Spin8 triality modes exist and invalidate cache when modified."""
    expected_modes = {"off", "codec_only", "vector", "spinor_plus", "spinor_minus", "consensus"}
    assert set(SPIN8_MODES) == expected_modes

    sess1 = SessionIdentity(
        model_set_id="bundle-1",
        backend_id="strata-native",
        kv_cache_dialect="standard",
        spin8_mode="consensus",
        steering_hash="",
        geometry_hash="",
    )

    sess2 = SessionIdentity(
        model_set_id="bundle-1",
        backend_id="strata-native",
        kv_cache_dialect="standard",
        spin8_mode="vector",  # changed mode
        steering_hash="",
        geometry_hash="",
    )

    with pytest.raises(ValueError, match="Session identity mismatch"):
        validate_session_resume(sess1, sess2)


def test_turboquant_dialects_preserved():
    """Verify TurboQuant dialects TQ4_1S and TQ3_0S are registered with checked byte computation."""
    assert 2001 in TURBOQUANT_DIALECTS
    assert 2002 in TURBOQUANT_DIALECTS

    tq4 = get_dialect_descriptor(2001)
    assert tq4.name == "TQ4_1S"
    assert tq4.dialect == DialectType.TURBOQUANT
    assert tq4.block_size == 32
    assert tq4.type_size == 20

    tq3 = get_dialect_descriptor(2002)
    assert tq3.name == "TQ3_0S"
    assert tq3.dialect == DialectType.TURBOQUANT
    assert tq3.block_size == 32
    assert tq3.type_size == 15

    # 128 elements of TQ4_1S = 4 blocks * 20 bytes = 80 bytes
    bytes_128 = compute_tensor_bytes([128, 1], tq4.block_size, tq4.type_size)
    assert bytes_128 == 80


def test_query_v3_bonsai_dialects_preserved():
    """Verify Bonsai/Prism PQ2_0 and PTQ1_0 are preserved with correct element ratios."""
    assert 142 in PRISM_DIALECTS
    assert 143 in PRISM_DIALECTS

    pq2 = get_dialect_descriptor(142)
    assert pq2.name == "PQ2_0"
    assert pq2.dialect == DialectType.PRISM
    assert pq2.block_size == 128
    assert pq2.type_size == 34

    ptq1 = get_dialect_descriptor(143)
    assert ptq1.name == "PTQ1_0"
    assert ptq1.dialect == DialectType.PRISM
    assert ptq1.block_size == 128
    # Recorded Prism block: 26 bytes of trits plus the two-byte scale.
    assert ptq1.type_size == 28

    # 256 elements of PQ2_0 = 2 blocks * 34 bytes = 68 bytes
    assert compute_tensor_bytes([256, 1], pq2.block_size, pq2.type_size) == 68

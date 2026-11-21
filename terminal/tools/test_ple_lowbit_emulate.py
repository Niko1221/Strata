"""pytest: tools/ple_lowbit_emulate.py. The last tests read the real IQ4_NL table (skipped when it is not on this machine)."""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ple_lowbit_emulate as E  # noqa: E402

GGUF_PY = HERE.parents[1] / "engine" / "third_party" / "strata-gguf-py"
REAL = pathlib.Path(r"D:\exstrata-work\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf")
DOC_NMSE = {"c4": 0.0361, "c3": 0.0281, "c2s": 0.0227}   # ple-representation.md section 4


def gguf_py():
    if not (GGUF_PY / "gguf").is_dir():
        pytest.skip("strata-gguf-py not in this checkout")
    sys.path.insert(0, str(GGUF_PY))
    import gguf
    return gguf


def synthetic_rows(n, seed=1):
    """n IQ4_NL rows of 90 B: random codes, fp16 scales of about the real size and either sign."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, (n, 5, 18), dtype=np.uint8)
    d = (rng.uniform(5e-5, 3e-4, (n, 5)) * rng.choice([-1, 1], (n, 5))).astype("<f2")
    b[:, :, :2] = d.view(np.uint8).reshape(n, 5, 2)
    return b.reshape(n, 90)


def nmse(y, x):
    return float(((y.astype(np.float64) - x) ** 2).sum() / (x.astype(np.float64) ** 2).sum())


def test_iq4nl_decode_matches_ggml_reference():
    rows = synthetic_rows(50)
    got = E.iq4nl_decode(rows)
    kv = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    for r in range(rows.shape[0]):
        for blk in range(5):
            raw = rows[r, blk * 18:(blk + 1) * 18]
            d = np.float32(raw[:2].view("<f2")[0])
            for j in range(16):   # dequantize_row_iq4_nl: y[j] = d*kvalues[qs[j] & 15], y[j+16] = d*kvalues[qs[j] >> 4]
                assert got[r, blk * 32 + j] == d * np.float32(kv[raw[2 + j] & 15])
                assert got[r, blk * 32 + j + 16] == d * np.float32(kv[raw[2 + j] >> 4])


def test_q8_0_roundtrip_within_bound_and_matches_ggml_rounding():
    x = np.random.default_rng(2).standard_normal((300, 160)).astype(np.float32) * 0.03
    x[0] = 0                                                       # an all-zero row must survive
    q = E.q8_0_encode(x)
    assert q.shape == (300, 170)
    y = E.q8_0_decode(q)
    d = np.abs(x.reshape(300, 5, 32)).max(2) / 127
    bound = (0.5 + 127 * 2.0 ** -11) * d                           # half a step, plus fp16 rounding of d times |q| <= 127
    assert (np.abs((y - x).reshape(300, 5, 32)) <= bound[:, :, None] * 1.0001 + 1e-12).all()
    assert (y[0] == 0).all()
    blk = x[1, :32]                                                # ggml quantize_row_q8_0_ref by hand: roundf = half away from zero
    d1 = np.float32(np.abs(blk).max() / np.float32(127))
    ref = [int(np.sign(v) * np.floor(abs(v * np.float32(1) / d1) + 0.5)) for v in blk]
    assert list(q[1, 2:34].view(np.int8)) == ref and q[1, :2].view("<f2")[0] == np.float16(d1)


def test_e4m3_round():
    v = np.array([0, 1, 1.0625, 1.1875, 448, 449, 1000, -0.3, 2 ** -9, 2 ** -10 * 1.1, 17], np.float32)
    got = E.e4m3_round(v)
    assert list(got[:7]) == [0, 1, 1.0, 1.25, 448, 448, 448]      # ties to even: 1.0625 -> 1.0, 1.1875 -> 1.25
    assert got[7] == np.float32(-0.3125) and got[8] == 2 ** -9 and got[9] == 2 ** -9 and got[10] == 16
    x = np.random.default_rng(3).standard_normal(5000).astype(np.float32) * 50
    assert (E.e4m3_round(E.e4m3_round(x)) == E.e4m3_round(x)).all()
    assert len(np.unique(np.abs(E.e4m3_round(np.linspace(0, 460, 400000, dtype=np.float32))))) == 127   # 126 finite positives + 0


def test_c4_gaussian_rows_are_62_bytes_and_sane():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4096, 160)).astype(np.float32) * 0.1
    Bn = (X / np.abs(X).max(1, keepdims=True)).ravel()
    grid = E.lloyd(Bn.astype(np.float64), 8).astype(np.float32)
    deq, codes, scales, _ = E.quantize(X, grid, 160, "f16", 1.0, 7)
    assert codes.shape == (4096, 160) and codes.max() < 8        # 3 bits a code
    assert scales.shape == (4096, 1)                              # one fp16 scale a row
    assert E.row_bytes("c4") == 160 * 3 // 8 + 2 == 62
    assert E.row_bytes("c3") == 65 and E.row_bytes("c2s") == 70
    assert (scales == scales.astype(np.float16)).all()
    assert np.array_equal(deq, (scales[:, :, None] * grid[codes.reshape(4096, 1, 160)]).reshape(4096, 160))
    assert 0.025 < nmse(deq, X) < 0.045                          # the doc's 0.0361 on near-Gaussian data


def test_pipeline_on_synthetic_table_and_workers_agree(tmp_path, monkeypatch):
    gguf = gguf_py()
    n = 3000
    rows = synthetic_rows(n, seed=5)
    src = tmp_path / "src.gguf"
    w = gguf.GGUFWriter(str(src), "strata-ple")
    w.add_tensor("per_layer_token_embd.weight", rows, raw_shape=rows.shape, raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    outs = []
    for workers in (1, 2):
        out = tmp_path / ("o%d.gguf" % workers)
        monkeypatch.setattr(sys, "argv", ["x", "--format", "c3", "--src", str(src), "--out", str(out), "--rows", "100:2900",
                                          "--chunk-rows", "700", "--workers", str(workers)])
        E.main()
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert (tmp_path / "o1.gguf.codebook.json").exists()
    r = gguf.GGUFReader(str(tmp_path / "o1.gguf"))
    assert len(r.tensors) == 1
    t = r.tensors[0]
    assert t.name == "per_layer_token_embd.weight" and list(t.shape) == [160, 2800]
    assert t.tensor_type == gguf.GGMLQuantizationType.Q8_0 and t.n_bytes == 2800 * 170
    assert r.data_offset % 32 == 0 and r.data_offset + t.n_bytes == len(outs[0])   # ngram.cpp:430-446: the tensor fills the file
    X = E.iq4nl_decode(rows[100:2900])
    assert 0.015 < nmse(E.q8_0_decode(np.asarray(t.data).reshape(2800, 170)), X) < 0.06


@pytest.mark.skipif(not REAL.exists(), reason="the real IQ4_NL table is not on this machine")
@pytest.mark.parametrize("fmt", ["c4", "c3", "c2s"])
def test_real_table_slice_nmse_matches_doc(fmt, tmp_path, monkeypatch, capsys):
    gguf = gguf_py()
    out = tmp_path / (fmt + ".gguf")
    monkeypatch.setattr(sys, "argv", ["x", "--format", fmt, "--src", str(REAL), "--out", str(out), "--rows", "0:200000",
                                      "--idle-priority"])
    E.main()
    text = capsys.readouterr().out
    with capsys.disabled():
        print("\n" + "\n".join(l for l in text.splitlines() if l.startswith(("NMSE", "E4M3", "codebook", "  grid", "  Q8_0", "format", "[0-9]"))))
    val = float([l for l in text.splitlines() if "after Q8_0" in l][0].split(":")[1])
    assert abs(val / DOC_NMSE[fmt] - 1) < 0.06
    t = gguf.GGUFReader(str(out)).tensors[0]
    assert t.name == "per_layer_token_embd.weight" and list(t.shape) == [160, 200000] and t.tensor_type == gguf.GGMLQuantizationType.Q8_0

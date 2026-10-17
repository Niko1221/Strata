#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII (п. 11): извлечение обязательных счётчиков движка из среза рабочего лога.

Счётчики, которые обязан фиксировать каждый прогон стенда (§8.7 и §9.2 patch-recommendations.md):
  * «… MB of expert reads» и «the OS read … MB (major faults)» — отличают «меньше PCIe» от
    «больше стрима»;
  * «expert cache … slots» и «the prompt path borrows …» — цена VRAM-затрат патча;
  * «decode expert cache hit rate»;
  * строка «layer split auto:» и выбранное разбиение «layer split: layers …»;
  * «profiled pairs» / «pre-filled N of M slots» — заполненность кэша;
  * «prompt N tokens = … reused + … read …» — привязка счётчиков к конкретному запросу.

usage: ab-extract-log.py --log FILE --out DIR [--from-byte N] [--to-byte M]
Создаёт: engine-request-counters.csv, engine-init-lines.txt, engine-cuda-errors.txt,
         engine-log-slice.log (исходный срез — оригинальные логи хранятся отдельно от таблиц)
"""
import argparse
import csv
import pathlib
import re

CUDA_ERR = re.compile(r"no kernel image is available|invalid device function|CUDA error|cudaError|"
                      r"out of memory|CUBLAS_STATUS|illegal memory access|device-side assert", re.I)

REQ = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                 r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)")
TIERS = re.compile(r"expert tiers: GPU (\d+) hits this request; since the start RAM (\d+) blobs, "
                   r"files (\d+) blobs ([\d.]+) MB read")
FILEIO = re.compile(r"file tier I/O this request: the OS read ([\d.]+) MB from storage \((\d+) major faults\) "
                    r"for ([\d.]+) MB of expert reads")
INIT_PAT = re.compile(r"expert cache \d+ slots|expert cache auto:|prompt path borrows|layer split auto:|"
                      r"layer split: layers |profiled pairs|pre-filled \d+ of \d+ slots|"
                      r"experts resident|prompt chunk auto|GiB free|token graph hit path|1020 MiB of VRAM free|"
                      r"VRAM free with everything loaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-byte", type=int, default=0)
    ap.add_argument("--to-byte", type=int, default=-1)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.log)
    with p.open("rb") as f:
        f.seek(a.from_byte)
        data = f.read() if a.to_byte < 0 else f.read(a.to_byte - a.from_byte)
    text = data.decode("utf-8", "replace")
    (out / "engine-log-slice.log").write_text(text, encoding="utf-8")

    lines = text.splitlines()
    init_lines = [ln for ln in lines if INIT_PAT.search(ln)]
    (out / "engine-init-lines.txt").write_text("\n".join(init_lines) + "\n", encoding="utf-8")
    errs = [ln for ln in lines if CUDA_ERR.search(ln)]
    (out / "engine-cuda-errors.txt").write_text("\n".join(errs) + "\n", encoding="utf-8")

    rows, cur = [], {}

    def flush():
        if cur:
            rows.append(dict(cur))
            cur.clear()

    for ln in lines:
        m = REQ.search(ln)
        if m:
            flush()
            cur.update(prompt_n=int(m.group(1)), reused=int(m.group(2)), read_tokens=int(m.group(3)),
                       prompt_ms=int(m.group(4)), prompt_tok_s=float(m.group(5)),
                       gen_n=int(m.group(6)), gen_ms=int(m.group(7)), gen_tok_s=float(m.group(8)),
                       drafts_ok=m.group(9) or "", drafts_offered=m.group(10) or "")
            continue
        m = HIT.search(ln)
        if m:
            cur.update(hit_rate_pct=float(m.group(1)), hits=int(m.group(2)), lookups=int(m.group(3)))
            continue
        m = TIERS.search(ln)
        if m:
            cur.update(gpu_hits=int(m.group(1)), ram_blobs=int(m.group(2)), file_blobs=int(m.group(3)),
                       file_mb_since_start=float(m.group(4)))
            continue
        m = FILEIO.search(ln)
        if m:
            cur.update(os_read_mb=float(m.group(1)), major_faults=int(m.group(2)), expert_reads_mb=float(m.group(3)))
            flush()
    flush()

    cols = ["prompt_n", "reused", "read_tokens", "prompt_ms", "prompt_tok_s", "gen_n", "gen_ms", "gen_tok_s",
            "drafts_ok", "drafts_offered", "hit_rate_pct", "hits", "lookups", "gpu_hits", "ram_blobs",
            "file_blobs", "file_mb_since_start", "os_read_mb", "major_faults", "expert_reads_mb"]
    with (out / "engine-request-counters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    tot = sum(r.get("expert_reads_mb", 0) for r in rows)
    print(f"строк запросов: {len(rows)}, суммарно expert reads: {tot:.1f} МБ, "
          f"OS read: {sum(r.get('os_read_mb', 0) for r in rows):.1f} МБ, "
          f"major faults: {sum(r.get('major_faults', 0) for r in rows)}, "
          f"строк CUDA-ошибок: {len(errs)}, инициализационных строк: {len(init_lines)}")


if __name__ == "__main__":
    main()

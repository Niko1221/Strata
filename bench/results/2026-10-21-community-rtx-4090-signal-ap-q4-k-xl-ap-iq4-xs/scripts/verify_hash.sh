#!/bin/bash
# SHA-256 of both Signal files against setup.py's pins, with the bcachefs checksum-error counters before and after
out=<BENCH>/hash-check.txt
cnt() { for d in /sys/fs/bcachefs/*/dev-0; do printf '%s=%s ' "$(basename $(dirname $d))" "$(awk -F: '/checksum/ {gsub(/ /,"",$2); print $2; exit}' $d/io_errors)"; done; }
{ echo "start: $(date -Is)"; echo "bcachefs checksum counters before: $(cnt)"
  for f in <MODEL_DATA>/ap-q4kxl-gguf/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf <MODEL_DATA>/ap-iq4xs/models/signal-AP-IQ4_XS/Signal-3.8-Flash-Next-AP-IQ4_XS.gguf; do
    t0=$(date +%s); h=$(sha256sum "$f" | cut -d' ' -f1); echo "$(basename $f) $h ($(( $(date +%s) - t0 )) s)"; done
  echo "bcachefs checksum counters after: $(cnt)"; echo "end: $(date -Is)"; } > "$out" 2>&1
cat "$out"

#!/usr/bin/env python3
"""Recreate the synthetic code/Russian prompts; no private conversation data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="Hybrid GGUF shard 1")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "tools"))
    from strata_tokenizer import Tokenizer
    tk = Tokenizer.from_gguf(args.model)
    nl = chr(10)
    prefix = nl.join(["<|im_start|>user", "Review this synthetic Python module for correctness, edge cases and maintainability. After the source, give a detailed technical review with concrete examples.", "```python", ""])
    suffix = nl.join(["", "```", "Provide the detailed review now, including examples and suggested tests.", "<|im_end|>", "<|im_start|>assistant", ""])
    body = "".join(nl.join([
        f"def transform_{i:05d}(values, limit={i%97+3}):",
        f'    """Normalize batch {i} while preserving order."""',
        "    result = []", "    for index, value in enumerate(values):",
        "        if value is None:", "            continue",
        f"        adjusted = (value * {i%19+1} + index) % limit",
        f"        if adjusted >= {i%7}:", "            result.append((index, adjusted))",
        "    return result", "", ""]) for i in range(4500))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    def save(name, n, before, text, after):
        a = tk.encode(before, parse_special=True)
        b = tk.encode(text, parse_special=False)
        c = tk.encode(after, parse_special=True)
        ids = a + b[:n-len(a)-len(c)] + c
        assert len(ids) == n
        encoded = json.dumps(ids)
        (args.output / (name + ".ids.json")).write_text(encoded + nl)
        manifest[name] = {"tokens": n, "ids_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    save("code4k", 4096, prefix, body, suffix)
    save("code64k", 65536, prefix, body, suffix)
    prefix = "<|im_start|>user"+nl+"Ниже дан синтетический журнал испытаний очереди задач. Проанализируй ограничения, противоречия и риски реализации. Ответь подробно на русском языке, предложи проверки и конкретные исправления."+nl
    suffix = nl+"Конец журнала. Дай техническое заключение и план проверки отказоустойчивости. Не пересказывай журнал целиком."+nl+"<|im_end|>"+nl+"<|im_start|>assistant"+nl
    body = "".join(f"Испытание {i:05d}. Очередь разделена на {i%17+1} разделов. Запись подтверждается после сохранения на диске. Рабочий процесс получил пакет из {i%53+1} задач, обработал {i%29} и потерял соединение. Повторная доставка использует тот же идентификатор, но таблица дедупликации очищается каждые {i%37+1} минут. Счётчик попыток хранится отдельно от результата. Необходимо проверить атомарность подтверждения, порядок сообщений и восстановление после перезапуска."+nl for i in range(2000))
    save("ru64k", 65536, prefix, body, suffix)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + nl)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()

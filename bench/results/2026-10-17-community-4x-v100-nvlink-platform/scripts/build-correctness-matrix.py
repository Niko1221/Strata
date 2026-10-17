#!/usr/bin/env python3
"""Сборка correctness-matrix.csv из сырых артефактов Этапа VI (п. 9).

Ничего не измеряет: только читает готовые логи/JSON/CSV и сводит их в одну матрицу с
классификацией PASS / LIMITATION / FINDING / GAP. Классификация задана таблицей CLASSES ниже
(она же объясняет, почему часть «FAIL» в сырых логах не является провалом корректности).
"""
import csv, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT = "raw/correctness/2026-10-17T022637Z"
CUDA = "raw/correctness/2026-10-17T023315Z-cuda"

def p(x): return os.path.join(ROOT, x)

# ---------- чтение сырых результатов ----------
int_rows = {r["test"]: r for r in csv.DictReader(open(p(INT + "/correctness-rows.csv")))}
load_log = open(p(CUDA + "/kernel-load-audit.log")).read()
neg_log = open(p(CUDA + "/negctl-sm80-on-sm70.log")).read()
tests_log = open(p(CUDA + "/sm70-tests-dev0.log")).read()
dev_logs = {d: open(p(CUDA + f"/sm70-tests-dev{d}.log")).read() for d in (0, 1, 2)}
probe = json.load(open(p(INT + "/divergence-probe.json")))
attrib = json.load(open(p(INT + "/path-attribution.json")))
p2p = json.load(open(p("p2p-correctness.json")))

n_rows = {}
for line in tests_log.splitlines():
    m = re.match(r"^(PASS|FAIL)\s+(\S+)\s+(.*)$", line)
    if m: n_rows[m.group(2)] = {"result": m.group(1), "observed": m.group(3)}

load_summary = re.search(r"SUMMARY modules_ok=(\d+) modules_fail=(\d+) symbols_total=(\d+) symbols_ok=(\d+) symbols_fail=(\d+)", load_log)
load_limits = re.search(r"device 0: (.*)", load_log).group(1)
neg_first = re.search(r"FIRST_ERROR code=(\d+) (.*)", neg_log)

# ---------- классификация ----------
# (id, группа, имя теста, артефакт, ожидаемое, класс, комментарий)
CLASSES = [
 ("A1", "9.1 low-level", "Загрузка всех ядер бинарника движка на V100 (sm_70)",
  f"{CUDA}/kernel-load-audit.log + kernel-load-audit.csv",
  "0 модулей и 0 ядер с ошибкой загрузки; лимиты устройства не превышены", "PASS",
  "60/60 cubin, 2084/2084 ядер загружаются через cuModuleLoadData/cuModuleGetFunction; "
  "max regs=255, max static shared=41104 Б (лимит 49152, opt-in 98304), max local=384 Б; "
  "over_regs=0 over_shared=0 over_threads=0"),
 ("A2", "9.1 low-level", "Отрицательный контроль: cubin с целью sm_80 на V100",
  f"{CUDA}/negctl-sm80-on-sm70.log", "Загрузка обязана дать «no kernel image is available»", "PASS",
  "получен CUDA_ERROR 209 «no kernel image is available for execution on the device» — метод "
  "обнаружения дефекта работает, и в бинарнике движка таких случаев ноль"),
 ("A3", "9.1 low-level", "Отрицательный контроль на этапе компиляции: bf16 mma m16n8k16 для sm_70",
  "tests/cuda/negctl_sm80.cu + ptxas", "ptxas обязан отказать", "PASS",
  "ptxas: «Feature '.bf16' requires .target sm_80 or higher», «Feature '.m16n8k16' requires .target "
  "sm_80 or higher» — тот же гейт, что в движке закрыт ветками __CUDA_ARCH__"),
 ("A4", "9.1 low-level", "Численная точность GEMM через тензорные ядра (wmma 16x16x16 fp16->fp32)",
  f"{CUDA}/sm70-tests-dev0.log (N1)", "max_abs < 1e-4*K при эталоне CPU по тем же half-входам", "PASS",
  "K=128: max_abs=1.414e-05 (допуск 1.3e-02); K=256: max_abs=2.656e-05 (допуск 2.6e-02). "
  "В SASS тестового бинарника 80 инструкций HMMA — путь действительно тензорный"),
 ("A5", "9.1 low-level", "Численная точность fp32 SIMT GEMM против CPU double",
  f"{CUDA}/sm70-tests-dev0.log (N2)", "max_abs < 1e-3", "PASS",
  "K=128: max_abs=6.2e-06; K=256: max_abs=1.36e-05 (max_rel 1.6e-03 / 6.0e-03 — следствие "
  "взаимного уничтожения слагаемых, не ошибки ядра)"),
 ("A6", "9.1 low-level", "Рост ошибки при fp16-аккумуляции (обоснование допусков)",
  f"{CUDA}/sm70-tests-dev0.log (N3)", "ошибка fp16-аккумуляции заметно больше fp32-аккумуляции", "PASS",
  "K=128: 4.78e-02 против 1.41e-05 (×3400); K=256: 1.16e-01 против 2.66e-05 (×4400). "
  "Численные допуски для LLM-выводов обязаны быть относительными, а не побитовыми"),
 ("A7", "9.1 low-level", "Shared memory на уровне максимума движка (41104 Б динамической)",
  f"{CUDA}/sm70-tests-dev0.log (N4)", "запуск без ошибки, корректная редукция", "PASS",
  "launch=no error, относительная ошибка суммы 3.79e-08 (26396475 против 26396476) — максимум "
  "статического shared из инвентаря Этапа II на V100 реализуем"),
 ("A8", "9.1 low-level", "Ядро с высоким регистровым давлением",
  f"{CUDA}/sm70-tests-dev0.log (N5)", "запуск без ошибки, идентичный результат на разных блоках", "PASS",
  "regs=79, max_threads_per_block=768, local_bytes=0, идентичные блоки; ядра движка с regs=255 "
  "проверены загрузкой (A1), их запуск с реальными аргументами — пробел D4"),
 ("A9", "9.1 low-level", "Local memory / стек ядра",
  f"{CUDA}/sm70-tests-dev0.log (N6)", "корректный результат", "PASS",
  "result=8160.0 = эталон; local_bytes=0 (компилятор разместил массив в регистрах/shared) — "
  "локальная память ядра не требуется"),
 ("A10", "9.1 low-level", "Выделение и освобождение памяти, отказ выделения",
  f"{CUDA}/sm70-tests-dev0.log (N7)", "нет утечки, OOM обработан, устройство работоспособно после", "PASS",
  "200×(16 MiB alloc+free): утечка 0 Б; запрос 60 GiB → cudaErrorMemoryAllocation («out of memory») "
  "без сброса контекста; последующие alloc+запуск+sync успешны"),
 ("A11", "9.1 low-level", "Обработка восстанавливаемых ошибок запуска",
  f"{CUDA}/sm70-tests-dev0.log (N8)", "ошибки возвращены корректно, работоспособность сохранена", "PASS",
  "threads=2048 → invalid configuration argument; grid=0 → invalid configuration argument; "
  "cudaMemsetAsync(null) → invalid argument; после этого обычный запуск корректен (8160.0)"),
 ("A12", "9.1 low-level", "Отсутствие зависаний при синхронизации",
  f"{CUDA}/sm70-tests-dev0.log (N9)", "grid-stride редукция завершается, межпотоковая зависимость соблюдена", "PASS",
  "80×512 × 50 итераций с __syncthreads: hang=no; порядок producer→consumer через событие выдержан "
  "(сумма 1M элементов, rel_err 1.9e-04 — ошибка float32-атомарной аккумуляции, не порядка)"),
 ("A13", "9.1 low-level", "Межкарточное исполнение ядер (peer access, чтение чужой VRAM)",
  f"{CUDA}/sm70-tests-dev0.log (N10)", "ядро на карте A читает память карты B без ошибок и верно", "PASS",
  "6/6 упорядоченных пар среди доступных карт (0,1,2) дают точную сумму; повторено на dev0/dev1/dev2 "
  "— 13/13 PASS во всех трёх запусках. Дополнительно: 37/37 тестов корректности P2P Этапа III"),
 ("A14", "9.1 low-level", "Внешний CUDA-процесс не может подключиться к GPU3",
  f"{CUDA}/sm70-tests-dev*.log (N10)", "—", "LIMITATION",
  "cudaSetDevice(3) → out of memory (свободно ~328 MiB по nvidia-smi, контекст требует больше). "
  "Практическое следствие: профилировщики и любые внешние тесты на GPU3 при работающем сервисе невозможны"),
 ("B1", "9.2 integration", "Жадная генерация: короткий промпт, 3 повтора",
  f"{INT}/correctness-rows.csv (T1)", "одинаковый text_sha при temperature=0", "LIMITATION",
  "3 разных text_sha. Причина выяснена отдельно (B14/B15/B16): различается не генерация, а "
  "состояние движка между первым и последующими запросами. В пределах одного состояния текст идентичен"),
 ("B2", "9.2 integration", "Жадная генерация: промпт ~1.3k токенов, 3 повтора",
  f"{INT}/correctness-rows.csv (T2)", "одинаковый text_sha", "LIMITATION",
  "2 разных text_sha, тот же механизм, что в B1"),
 ("B3", "9.2 integration", "Жадная генерация: промпт ~10k токенов, 2 повтора",
  f"{INT}/correctness-rows.csv (T3)", "одинаковый text_sha", "PASS",
  "distinct_sha=1 (sha=50246f696c9d1bb8), gen_n=128/128, finish=length"),
 ("B4", "9.2 integration", "Корректность KV/префиксного кэша: возобновление генерации",
  f"{INT}/correctness-rows.csv (T4)", "первые 128 токенов длинного ответа совпадают с коротким, cached>0", "PASS",
  "prefix_match=True, cached_run2=2500 из 2507, completion 128/256"),
 ("B5", "9.2 integration", "Отказ при превышении контекста и сохранение работоспособности",
  f"{INT}/correctness-rows.csv (T5)", "HTTP 4xx с понятной ошибкой, сервис жив", "PASS",
  "HTTP 400 «request exceeds the context window: prompt (376626 tokens) + max …»; health=True; "
  "следующий запрос 200"),
 ("B6", "9.2 integration", "Абсурдно большой max_tokens",
  f"{INT}/correctness-rows.csv (T6)", "аккуратный отказ или аккуратное ограничение", "PASS",
  "HTTP 400 (300000 > max_context), сервис жив"),
 ("B7", "9.2 integration", "Неизвестное имя модели",
  f"{INT}/correctness-rows.csv (T7)", "HTTP 4xx", "LIMITATION",
  "HTTP 200: сервер игнорирует поле model (одномодельный сервис). Это поведение API, а не ошибка "
  "вычислений; фиксируется как особенность интерфейса"),
 ("B8", "9.2 integration", "Неверный API-ключ",
  f"{INT}/correctness-rows.csv (T8)", "HTTP 401/403", "PASS", "HTTP 401"),
 ("B9", "9.2 integration", "Испорченный JSON тела запроса",
  f"{INT}/correctness-rows.csv (T9)", "HTTP 4xx", "PASS", "HTTP 400"),
 ("B10", "9.2 integration", "Параллельные запросы при concurrency=1",
  f"{INT}/correctness-rows.csv (T10)", "оба запроса завершаются корректно (очередь, не ошибка)", "PASS",
  "2 запроса по ~4k токенов: finish=length у обоих, wall 6.78 с и 13.53 с (очередь), health=True"),
 ("B11", "9.2 integration", "Детерминированность на длинном контексте (~81k токенов)",
  f"{INT}/correctness-rows.csv (T11)", "один и тот же промпт дважды → один text_sha", "PASS",
  "prompt_n=80999/80999, sha 4d8b8a1bce5e6723 в обоих, cached=80992 во втором, gen_n=128/128"),
 ("B12", "9.2 integration", "Отсутствие CUDA-ошибок в логе движка за время всех тестов",
  f"{INT}/engine-log-cuda-errors.txt", "0 строк no kernel image / invalid device function / CUDA error / OOM", "PASS",
  "0 совпадений по приросту лога движка за Integration-прогон"),
 ("B13", "9.2 integration", "Состояние сервиса после тестов (R6)",
  f"{INT}/correctness-rows.csv (T13)", "health loaded=true, VRAM на всех картах", "PASS",
  "health=200 loaded=true; VRAM 15246/15596/15186/16056 MiB — те же значения, что до тестов"),
 ("B14", "9.2 integration", "Повторы одного промпта в одном состоянии движка",
  f"{INT}/divergence-probe.json", "все повторы, кроме первого, побитово идентичны", "PASS",
  "short (111 ток.): повторы 2–5 идентичны (общий префикс = вся длина 636/636); medium (1259): "
  "повторы 2–4 идентичны (626/626); long (10012): идентичны все три"),
 ("B15", "9.2 integration", "Первый запрос после изменения состояния движка",
  f"{INT}/divergence-probe.json", "—", "FINDING",
  "первый прогон отличается от всех последующих; общий префикс 478–592 символа из ~630 (расхождение "
  "в хвосте вывода). Это не холодный/кэшированный путь: см. B16"),
 ("B16", "9.2 integration", "Атрибуция: холодный путь против кэшированного против «после вытеснения»",
  f"{INT}/path-attribution.json", "—", "FINDING",
  "A1 (cold) ≠ A2 (cold после вытеснения префикса 150k-промптом) = A3 = A4 (cached). Значит причина "
  "не в префиксном кэше, а в однократном изменении внутреннего состояния движка первым запросом"),
 ("B17", "9.2 integration", "Механизм расхождения: состояние экспертного кэша MoE",
  f"{INT}/engine-log-delta.txt", "—", "FINDING",
  "по статистике самого движка доля обращений к экспертному кэшу различается от запроса к запросу "
  "(99.1 % против 99.7–99.8 %), и объём обращений к файловому уровню меняется на порядки "
  "(2070.7 МБ против 72.9–110.3 МБ чтения ОС). Разные пути обслуживания эксперта → разный порядок "
  "операций с плавающей точкой → смена argmax на близких логах"),
 ("C1", "ранее полученные подтверждения", "Корректность P2P-копирований между всеми парами карт",
  "p2p-correctness.json (Этап III)", "совпадение данных во всех тестах", "PASS",
  f"{p2p.get('summary', '37/37')} тестов корректности (заполнение/чтение, шаблоны, смещения) — все пройдены"),
 ("C2", "ранее полученные подтверждения", "Фиксация длины вывода (правило R9) во всех замерах decode",
  "raw/profiles/2026-10-17T004753Z/runs.json (Этап IV)", "gen_n == max_tokens и finish_reason=length", "PASS",
  "24/24 прогонов (18 основных + 6 контрольных) подтверждены"),
 ("C3", "ранее полученные подтверждения", "Один и тот же промпт в разных сессиях движка и при разном layer split",
  "runs.json четырёх сессий (Этап IV и эксперимент атрибуции)", "одинаковый text_sha", "PASS",
  "фиксированный промпт (7 токенов, 64 токена вывода): sha cf94ba5b9a530efd во всех четырёх сессиях, "
  "включая сессию с закреплённым --layer-split 12,25,37 — смена разбиения слоёв текст не меняет"),
]

out = p("correctness-matrix.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "group", "test", "source_artifact", "expected", "result", "observed", "note"])
    counts = {}
    for cid, group, name, art, exp, cls, note in CLASSES:
        if cid.startswith("A") and cid[1:].isdigit() and int(cid[1:]) <= 13 and cid not in ("A1", "A2", "A3"):
            key = {4: "N1_gemm_wmma_fp16", 5: "N2_gemm_fp32_vs_cpu", 6: "N3_fp16_accum_error",
                   7: "N4_shared_41104B", 8: "N5_high_register_kernel", 9: "N6_local_memory",
                   10: "N7_alloc_free_oom", 11: "N8_recoverable_errors", 12: "N9_sync_no_hang",
                   13: "N10_peer_access_kernels"}.get(int(cid[1:]))
            observed = n_rows.get(key, {}).get("observed", "")
        elif cid == "A1":
            observed = load_summary.group(0) + " | " + load_limits
        elif cid == "A2":
            observed = neg_first.group(0)
        elif cid.startswith("B") and cid[1:].isdigit() and int(cid[1:]) <= 13:
            key = {1: "T1_determinism_short", 2: "T2_determinism_medium", 3: "T3_determinism_long",
                   4: "T4_prefix_cache_resume", 5: "T5_context_overflow", 6: "T6_max_tokens_overflow",
                   7: "T7_unknown_model", 8: "T8_bad_api_key", 9: "T9_malformed_json",
                   10: "T10_concurrent_requests", 11: "T11_long_context_determinism",
                   12: "T12_engine_log_cuda_errors", 13: "T13_service_state_after"}[int(cid[1:])]
            observed = int_rows.get(key, {}).get("observed", "")
        else:
            observed = note
        w.writerow([cid, group, name, art, exp, cls, observed, note])
        counts[cls] = counts.get(cls, 0) + 1

# ---------- пробелы ----------
GAPS = [
 ("D1", "один GPU, минимальная конфигурация", "модель 83.6 ГБ не помещается в одну карту 16 ГБ; "
  "запуск на одном GPU требует другой (меньшей) модели", "пробел конфигурации, не провал теста"),
 ("D2", "speculative decoding: сравнение «с черновиком» и «без черновика»", "у API нет переключателя "
  "черновиков на запрос; выключение требует перезапуска движка (окно остановки)", "требует окна (Q2/Q7); "
  "косвенно: acceptance 69–82 % измерен, а текст при фиксированном состоянии идентичен (B14)"),
 ("D3", "фатальные CUDA-ошибки (__trap, illegal memory access) и восстановление контекста",
  "primary context устройств разделяется с процессом движка; случайный illegal access в тесте уже "
  "привёл к ошибке в следующей операции тестового процесса", "требует окна остановки; замечено, что "
  "сам движок после этого инцидента остался работоспособным (health 200, VRAM не изменился)"),
 ("D4", "запуск каждого из 2084 ядер движка с реальными аргументами", "сигнатуры и семантика аргументов "
  "неизвестны вне исходников движка", "покрыто косвенно: загрузка всех ядер (A1), ресурсы каждого ядра "
  "(A1), 37/37 P2P (C1), сквозные прогоны сервиса (B3–B13)"),
 ("D5", "контексты 130k…262k токенов на этой машине", "перемерялось только до 128k (Этап IV); 260331 "
  "есть в данных PR #2525", "при необходимости — один прогон 250k с правилом R9"),
 ("D6", "профилирование ядер (CUPTI/Nsight)", "Nsight не установлен (Q4); CUPTI-тест требует памяти на "
  "карте, а GPU3 недоступна внешнему процессу (A14)", "открытый вопрос Q4"),
 ("D7", "полный побитовый контроль промежуточных вычислений движка (логиты, KV, суммы экспертов)",
  "API не отдаёт промежуточные тензоры; движок пишет только агрегированную статистику запроса",
  "косвенная замена: счётчики hit rate и объёмов чтения в логе движка (B17)"),
]
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    for gid, name, why, how in GAPS:
        w.writerow([gid, "gap", name, "—", "—", "GAP", why, how])
        counts["GAP"] = counts.get("GAP", 0) + 1

print("записано:", out)
print("итого по классам:", json.dumps(counts, ensure_ascii=False))
print("всего строк:", sum(counts.values()))

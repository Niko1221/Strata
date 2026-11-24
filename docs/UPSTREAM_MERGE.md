# Безопасный мерж upstream

Инструмент: `tools/upstream_merge/upstream_merge.py`.

## Принцип

Мерж выполняется только по матрице решений. Инструмент не принимает решения за человека и не пушит изменения без явного `--yes --push`.

## Команды

```text
python tools/upstream_merge/upstream_merge.py preflight --repo . --ours main --upstream upstream/main
python tools/upstream_merge/upstream_merge.py preflight --repo . --fetch
python tools/upstream_merge/upstream_merge.py matrix --repo . --profile tools/upstream_merge/merge_profile.example.json --out merge-matrix.json
python tools/upstream_merge/upstream_merge.py prepare --repo . --branch sync/upstream-0.2.21 --yes
python tools/upstream_merge/upstream_merge.py experiment --repo . --branch opt/example --yes
python tools/upstream_merge/upstream_merge.py status --repo .
python tools/upstream_merge/upstream_merge.py check --repo . --plan merge-plan.json --format both
python tools/upstream_merge/upstream_merge.py promote --repo . --branch opt/example --plan merge-plan.json --report-out merge-promote.json --yes
python tools/upstream_merge/upstream_merge.py promote --repo . --branch opt/example --plan merge-plan.json --report-out merge-promote.json --yes --push --pr --delete-local
python tools/upstream_merge/upstream_merge.py abort --repo . --yes
python tools/upstream_merge/upstream_merge.py selftest
```

## Безопасность

- `preflight` без `--fetch` только читает refs и дерево.
- `matrix` строит черновик: наличие якоря не определяет `full/partial/none`; нужны явный статус, evidence и ручной review.
- `prepare` без `--yes` ничего не меняет.
- `prepare --yes` создаёт локальный backup tag, sync-ветку и запускает `merge --no-ff --no-commit`.
- `check` и `promote` сначала валидируют `tools/upstream_merge/merge-plan.schema.json`, затем семантику `merge_gate`: refs, ancestry, dirty tree, anchors, evidence, warnings и независимое review.
- Современный план обязан содержать все 14 категорий evidence; `anchorMethod`/`anchorEvidence` обязательны, `anchorStatus=missing` блокируется.
- Каждый `implementationCombinations` run имеет собственные `status=passed`, `revision` и непустой `evidence`; `anchorStatus=removed` допускается только для `drop` с полной upstream-заменой.
- `abort --yes` выполняет только `git merge --abort`.
- Push требует явного `--yes --push`; PR дополнительно требует `--pr` и `gh`.

## Жизненный цикл эксперимента

- `experiment --yes` создаёт отдельную ветку от `main`.
- Эксперимент коммитится в своей ветке.
- Успешный эксперимент проходит `check --run-checks`.
- `promote --yes` требует `--report-out`, включает полный gate-отчёт, создаёт backup tag и переводит `main` через `--ff-only`.
- До изменения refs записывается `promote-started`; финальный отчёт включает gate, cleanup errors и recovery metadata.
- При сбое tag/switch/merge инструмент откатывает созданные tag/ветку и возвращает recovery-отчёт.
- `--push` отправляет `main` и экспериментальную ветку атомарно через `--atomic`.
- `--pr` создаёт PR через `gh`; без `gh` команда отклоняется до изменения main.
- `--delete-local` удаляет только локальную ветку и только после успешного push; remote-ветка остаётся.

## Владелец сборок и прогонов

Проектный hook `.qoder/settings.json` вызывает `tools/upstream_merge/owner_guard.ps1`. Он получает `session_id` из PreToolUse и разрешает build/test команды только сеансу `443fddf1-c6a1-4de1-a762-0b61e5ca941c`. Остальные сеансы получают блокирующий отказ. Эти два файла — локальная обвязка сеанса и не включаются в upstream PR.

## Порядок

1. Preflight и матрица «наш патч vs апстрим».
2. Ручное сравнение алгоритмов, контрактов, маршрута, ресурсов и совместимости.
3. Заполнение `merge-plan.json`.
4. `prepare --yes` в отдельной sync-ветке.
5. Разрешение конфликтов по матрице.
6. Сборка, тесты, correctness/quality и контрольные точки.
7. `check --run-checks`.
8. Переход `main` через `--ff-only` только после зелёного гейта.

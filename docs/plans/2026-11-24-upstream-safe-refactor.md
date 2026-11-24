# План безопасного рефакторинга «на месте апстрима»

## Цель

Уменьшить размер форкового ядра, не превращая синхронизацию с upstream в постоянное переписывание. Общая инфраструктура, кэш, lifecycle и dispatch остаются на месте upstream; наши головы, ядра и оптимизации подключаются через узкие адаптеры.

## Границы

- Не менять алгоритмы, численные результаты, дефолты и публичные контракты в рефакторинге.
- Не переносить upstream-файлы целиком ради уменьшения diff.
- Не выносить CUDA kernels и templates из header-only `.cuh` без доказанной необходимости.
- Не трогать `init`, placement и lifetime буферов до завершения inventory ownership.
- Каждый этап — отдельный коммит, сборка и тесты.

## Карта

| Область | База | Наша часть | Целевая граница |
|---|---|---|---|
| Cache и lifecycle | upstream engine/serve | expert cache, MAP/D1 | adapter policy, без замены lifecycle |
| Dispatch и graphs | upstream graph/scheduler | q36 route, FA/FDEC, P4 | route descriptor + fallback |
| Heads и kernels | upstream interfaces | q36 heads, fused down, int8 | kernel interface с теми же допусками |
| KV/session | upstream session layout | quantized KV, FDEC, copy_session | явный layout adapter, без скрытых смещений |
| Serve | upstream request/serve | batch/MTP/read windows | тонкая orchestration обвязка |

## Этапы

1. Зафиксировать публичные интерфейсы и маркеры наших патчей.
2. Выделить leaf helpers и verification без изменения lifetime.
3. Разделить `Serve` на control/slots/schedule/request/run.
4. Выделить rows/session copy с явным KV ownership.
5. Подключить upstream cache/lifecycle через adapter policy.
6. Подключить upstream dispatch/graphs через route descriptor.
7. Перенести MoE/forward/heads за стабильные kernel interfaces.
8. Только после inventory ownership — `init` и placement.

## Критерии приёмки

- Сборка всех затронутых backend’ов.
- Bitwise/численные parity-тесты для KV, session copy, rows, MTP и forward.
- Сохранение fallback, graph capture и CPU/GPU hybrid.
- Контроль pp/tg на одинаковой модели, окне и флагах.
- Отсутствие утечек и двойного освобождения.
- Минимальный diff в upstream-файлах и понятные adapter boundaries.

## Красные флаги

- Копирование upstream-функции целиком.
- Зависимость от внутренних смещений чужого буфера.
- Скрытые глобальные состояния и fork-условия по чужому коду.
- Изменение layout, ownership или stream ordering без отдельной приёмки.
- Рост конфликтной поверхности при следующем upstream merge.

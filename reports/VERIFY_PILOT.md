# Verification report (pilot)

Generated: 2026-10-06 10:21:37 | runtime: 6505.6s

## Totals

- input rows: **7186** | passed: **4251** | failed: **2935** | pass rate: **59.2%**
- inputs: `data/raw_trajectories/gen_local.jsonl`, `data/raw_trajectories/gen_openai.jsonl`

## Per-stage failures

| stage | pass | fail | skipped |
|---|---|---|---|
| structural | 6917 | 269 | 0 |
| schema | 7186 | 0 | 0 |
| exec | 6846 | 340 | 0 |
| secrets | 7176 | 10 | 0 |
| language | 7164 | 22 | 0 |
| judge | 4461 | 2725 | 0 |
| mmr | 7175 | 11 | 0 |

## Breakdown by mode / language / teacher

### Mode
| group | total | passed | pass rate |
|---|---|---|---|
| error_recovery | 269 | 121 | 45% |
| no_tool | 1103 | 985 | 89% |
| tool_use | 5814 | 3145 | 54% |

### Language
| group | total | passed | pass rate |
|---|---|---|---|
| de | 1805 | 1002 | 56% |
| en | 469 | 256 | 55% |
| es | 1437 | 856 | 60% |
| fr | 1755 | 1108 | 63% |
| pl | 1720 | 1029 | 60% |

### Teacher
| group | total | passed | pass rate |
|---|---|---|---|
| gpt-5.4-mini | 106 | 36 | 34% |
| gpt-5.4-nano | 86 | 39 | 45% |
| local/gpt-oss-20b | 600 | 257 | 43% |
| local/qwen3.5-9b | 6394 | 3919 | 61% |

## Judge scores

- `local/qwen3.5-9b` (n=7186): alignment=7.59, grounding=7.37, naturalness=8.15, language=9.93, avg=8.26 | histogram(avg): 1:2 2:13 3:100 4:630 5:584 6:471 7:513 8:552 9:2722 10:1599
- `local/gpt-oss-20b` (n=7186): alignment=7.53, grounding=8.5, naturalness=8.56, language=9.87, avg=8.61 | histogram(avg): 1:47 2:14 3:82 4:175 5:268 6:395 7:1019 8:769 9:2295 10:2122

## Exec replay

- unique recorded calls: 17343 (cache hits: 14731, live: 2612)
- recorded-ok calls: 22570 replayed ok, 156 failed on replay
- recorded-fail calls that succeed now (recovered): 0

## Language-consistency catches (expected ~3 French mislabels)

- `tool_use-fd80b9999cac`: heuristic: a user turn looks en (declared pl); judge confirms en != declared pl
- `tool_use-b7d1e317f80a`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-aa60e1757754`: heuristic: a user turn looks en (declared de); judge confirms en != declared de
- `tool_use-09c73ade45f5`: heuristic: a user turn looks en (declared de); judge confirms en != declared de
- `no_tool-bd9536222a57`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `no_tool-500e7c10cbc3`: heuristic: a user turn looks es (declared de); judge confirms es != declared de
- `tool_use-73d91b143675`: heuristic: a user turn looks en (declared es); judge confirms en != declared es
- `tool_use-9de13bb26503`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-c9eaf5b46e90`: heuristic: a user turn looks en (declared de); judge confirms en != declared de
- `no_tool-59392b355e65`: heuristic: a user turn looks en (declared de); judge confirms en != declared de
- `no_tool-56f2db6ad663`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-e05cfea1d9d8`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-80f6f60ba685`: heuristic: a user turn looks en (declared es); judge confirms en != declared es
- `no_tool-bc2816bfbe0b`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `no_tool-7ada22526fb2`: heuristic: a user turn looks en (declared es); judge confirms en != declared es
- `tool_use-4268034a255e`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-81e0d720ebb3`: heuristic: a user turn looks en (declared es); judge confirms en != declared es
- `tool_use-2200259ce843`: heuristic: a user turn looks en (declared de); judge confirms en != declared de
- `no_tool-d1dd503ab60c`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-b66abe4215ad`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `no_tool-41c18f0dbe36`: heuristic: a user turn looks en (declared fr); judge confirms en != declared fr
- `tool_use-fd672eaf3fd1`: heuristic: a user turn looks en (declared de); judge confirms en != declared de

## Rejection reasons histogram

- 1204x judge: dimension scored < 5 (min=2)
- 568x judge: dimension scored < 5 (min=1)
- 478x judge: dimension scored < 5 (min=3)
- 449x judge: dimension scored < 5 (min=4)
- 274x exec: unknown tool in exec_log: 'tomtom-waypoint-routing'
- 92x structural: msg[6] tool_call name not in registry: 'tomtom-waypoint-routing'
- 65x exec: replay failed: tomtom-routing (MCPError: [tomtom] tool tomtom-routing errored: {'content': [{'type': 'text', '
- 42x exec: replay failed: tomtom-fuzzy-search (MCPError: [tomtom] tool tomtom-fuzzy-search errored: {'content': [{'type':
- 37x structural: msg[5] tool_call name not in registry: 'tomtom-waypoint-routing'
- 37x structural: msg[9] tool_call name not in registry: 'tomtom-waypoint-routing'
- 30x judge: judge avg < 7.0 (local/qwen3.5-9b=4.2, local/gpt-oss-20b=7.5)
- 26x structural: msg[7] tool_call name not in registry: 'tomtom-waypoint-routing'
- 24x structural: msg[11] tool_call name not in registry: 'tomtom-waypoint-routing'
- 23x structural: msg[2] tool_call name not in registry: 'tomtom-waypoint-routing'
- 21x judge: judge avg < 7.0 (local/qwen3.5-9b=6.2, local/gpt-oss-20b=9.0)
- 19x judge: judge avg < 7.0 (local/qwen3.5-9b=5.8, local/gpt-oss-20b=9.2)
- 18x judge: judge avg < 7.0 (local/qwen3.5-9b=5.8, local/gpt-oss-20b=7.5)
- 18x exec: replay failed: tomtom-poi-search (MCPError: [tomtom] tool tomtom-poi-search errored: {'content': [{'type': 'te
- 17x judge: judge avg < 7.0 (local/qwen3.5-9b=5.8, local/gpt-oss-20b=9.5)
- 17x judge: judge avg < 7.0 (local/qwen3.5-9b=4.2, local/gpt-oss-20b=7.0)
- 17x judge: judge avg < 7.0 (local/qwen3.5-9b=6.2, local/gpt-oss-20b=9.2)
- 16x judge: judge avg < 7.0 (local/qwen3.5-9b=4.2, local/gpt-oss-20b=6.2)
- 16x judge: judge avg < 7.0 (local/qwen3.5-9b=5.8, local/gpt-oss-20b=9.0)
- 16x judge: judge avg < 7.0 (local/qwen3.5-9b=4.8, local/gpt-oss-20b=7.8)
- 15x judge: judge avg < 7.0 (local/qwen3.5-9b=5.8, local/gpt-oss-20b=10.0)

## Example rejected dialogs

- `tool_use-41eefd64c71a` (gen_local.jsonl:3, tool_use/pl) failed=['structural', 'exec', 'judge']: "Planuję podróż z Wrocławia do Gdańska. Chcę uniknąć odcinków płatnych, zrobić jedną 30‑min"
- `error_recovery-ea165a369d4a` (gen_local.jsonl:6, error_recovery/fr) failed=['judge']: "Je veux faire le trajet de Wuhan à Hengyang en suivant la route la plus pittoresque, ça fa"
- `tool_use-2f21db4517d2` (gen_local.jsonl:10, tool_use/pl) failed=['judge']: "Zaczynam od Krakowa do Zakopanego, chcę omijać autostrady i szukam malowniczej trasy, bo c"
- `no_tool-a3262974890f` (gen_local.jsonl:12, no_tool/de) failed=['judge']: "Hey, hör zu. Eilige Frage für die Notizen: Erinnere mich bitte an 14 Uhr, dass ich die veg"
- `tool_use-3c80e8b5a396` (gen_local.jsonl:14, tool_use/es) failed=['structural', 'exec']: "Hola, me has dicho que viaje desde Valencia hasta San Sebastián sin peajes, con parada par"
- `tool_use-7559eeec3da0` (gen_local.jsonl:15, tool_use/de) failed=['judge']: "Hey, kannst du mir bitte die Route von Köln nach Wien planen? Wir wollen in Nürnberg zum T"
- `tool_use-ce3396ca84c9` (gen_local.jsonl:16, tool_use/fr) failed=['structural', 'exec']: "Bonjour, je vais de Bordeaux à Limoges en prenant la nationale sans péage. Pourrais-tu pla"
- `tool_use-bd4ebdde968f` (gen_local.jsonl:17, tool_use/de) failed=['judge']: "Ich plane eine Fahrt von Sydney nach Melbourne und muss dabei durch Canberra und Albury fa"

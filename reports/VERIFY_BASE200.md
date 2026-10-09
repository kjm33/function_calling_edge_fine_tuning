# Verification report (pilot)

Generated: 2026-10-03 12:38:46 | runtime: 294.0s

## Totals

- input rows: **200** | passed: **93** | failed: **107** | pass rate: **46.5%**
- inputs: `/tmp/opencode/gen_batch1_first200.jsonl`

## Per-stage failures

| stage | pass | fail | skipped |
|---|---|---|---|
| structural | 200 | 0 | 0 |
| schema | 200 | 0 | 0 |
| exec | 200 | 0 | 0 |
| secrets | 197 | 3 | 0 |
| language | 200 | 0 | 0 |
| judge | 95 | 105 | 0 |
| mmr | 200 | 0 | 0 |

## Breakdown by mode / language / teacher

### Mode
| group | total | passed | pass rate |
|---|---|---|---|
| error_recovery | 6 | 5 | 83% |
| no_tool | 27 | 27 | 100% |
| tool_use | 167 | 61 | 37% |

### Language
| group | total | passed | pass rate |
|---|---|---|---|
| de | 55 | 25 | 45% |
| en | 6 | 4 | 67% |
| es | 36 | 12 | 33% |
| fr | 46 | 27 | 59% |
| pl | 57 | 25 | 44% |

### Teacher
| group | total | passed | pass rate |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 21 | 8 | 38% |
| google/gemma-4-26b-a4b-it | 28 | 9 | 32% |
| gpt-4.1-mini | 18 | 10 | 56% |
| gpt-4o-mini | 34 | 21 | 62% |
| gpt-5.4-mini | 35 | 12 | 34% |
| gpt-5.4-nano | 23 | 11 | 48% |
| qwen/qwen3.5-9b | 10 | 5 | 50% |
| z-ai/glm-5.3-flash | 31 | 17 | 55% |

## Judge scores

- `gpt-5.4-mini` (n=200): alignment=6.03, grounding=5.97, naturalness=7.63, language=9.78, avg=7.35 | histogram(avg): 1:2 2:1 3:2 4:11 5:30 6:35 7:35 8:38 9:22 10:24
- `qwen/qwen3.5-9b` (n=200): alignment=8.24, grounding=7.8, naturalness=8.55, language=10.0, avg=8.65 | histogram(avg): 3:3 4:14 5:12 6:6 7:12 8:13 9:76 10:64

## Exec replay

- unique recorded calls: 726 (cache hits: 685, live: 41)
- recorded-ok calls: 744 replayed ok, 0 failed on replay
- recorded-fail calls that succeed now (recovered): 0

## Language-consistency catches (expected ~3 French mislabels)

- none found

## Rejection reasons histogram

- 37x judge: dimension scored < 5 (min=2)
- 30x judge: dimension scored < 5 (min=4)
- 20x judge: dimension scored < 5 (min=3)
- 18x judge: dimension scored < 5 (min=1)
- 3x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=9.2)
- 3x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=9.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.8, qwen/qwen3.5-9b=9.2)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=8.5)
- 2x secrets: secret leak x7 (scrubbed in output copy)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=4.5, qwen/qwen3.5-9b=4.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=4.2, qwen/qwen3.5-9b=5.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=5.5, qwen/qwen3.5-9b=9.2)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.8, qwen/qwen3.5-9b=9.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.0, qwen/qwen3.5-9b=9.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=9.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.8, qwen/qwen3.5-9b=10.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.8, qwen/qwen3.5-9b=9.5)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=4.0, qwen/qwen3.5-9b=8.2)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=4.2, qwen/qwen3.5-9b=6.5)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=5.2, qwen/qwen3.5-9b=5.5)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=6.0, qwen/qwen3.5-9b=7.5)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=8.0, qwen/qwen3.5-9b=5.8)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=5.2, qwen/qwen3.5-9b=10.0)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=6.8, qwen/qwen3.5-9b=7.5)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=5.2, qwen/qwen3.5-9b=9.8)

## Example rejected dialogs

- `tool_use-3ac878b08543` (gen_batch1_first200.jsonl:1, tool_use/fr) failed=['judge']: "Je pars de Besançon pour aller à Genève, en évitant le tunnel du Mont-Blanc à cause du bro"
- `tool_use-c5ce79c6b381` (gen_batch1_first200.jsonl:4, tool_use/de) failed=['judge']: "Ich möchte von Sydney nach Melbourne fahren und dabei über Canberra und Albury fahren. Kan"
- `tool_use-f4c8cb442b23` (gen_batch1_first200.jsonl:10, tool_use/es) failed=['judge']: "Vale, necesito ir desde Valencia hasta San Sebastián. Quiero evitar peajes, y quiero hacer"
- `tool_use-d4f1838f0164` (gen_batch1_first200.jsonl:12, tool_use/es) failed=['judge']: "Quiero ir de Lyon a Marsella por la ruta costera más pintoresca, evitando peajes porque ya"
- `tool_use-f78331e0b9ac` (gen_batch1_first200.jsonl:15, tool_use/pl) failed=['judge']: "Jasne — jadę z Gdańska do Helu i chcę unikać płatnych odcinków na S6. Trasa ma uwzględnić "
- `tool_use-9e3d6bc4f99d` (gen_batch1_first200.jsonl:17, tool_use/pl) failed=['judge']: "Cześć, chciałbym pojechać z Warszawy do Zakopanego, ale proszę wyznaczyć trasę przez malow"
- `tool_use-33e0a2902c94` (gen_batch1_first200.jsonl:18, tool_use/fr) failed=['judge']: "Je veux un itinéraire le plus rapide de Ba Vì à Ban Mo, et j’aimerais aussi savoir quelles"
- `tool_use-21ba7dcc3429` (gen_batch1_first200.jsonl:19, tool_use/es) failed=['judge']: "Quiero ir de Shijiazhuang a Dingxi, que son aproximadamente 1220 km. Me gustaría hacer una"

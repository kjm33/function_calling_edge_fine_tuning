# Verification report (pilot)

Generated: 2026-10-03 12:38:56 | runtime: 303.4s

## Totals

- input rows: **219** | passed: **111** | failed: **108** | pass rate: **50.7%**
- inputs: `data/raw_trajectories/ab_test.jsonl`

## Per-stage failures

| stage | pass | fail | skipped |
|---|---|---|---|
| structural | 219 | 0 | 0 |
| schema | 219 | 0 | 0 |
| exec | 217 | 2 | 0 |
| secrets | 217 | 2 | 0 |
| language | 219 | 0 | 0 |
| judge | 113 | 106 | 0 |
| mmr | 218 | 1 | 0 |

## Breakdown by mode / language / teacher

### Mode
| group | total | passed | pass rate |
|---|---|---|---|
| error_recovery | 9 | 7 | 78% |
| no_tool | 26 | 26 | 100% |
| tool_use | 184 | 78 | 42% |

### Language
| group | total | passed | pass rate |
|---|---|---|---|
| de | 61 | 30 | 49% |
| en | 10 | 7 | 70% |
| es | 53 | 28 | 53% |
| fr | 47 | 25 | 53% |
| pl | 48 | 21 | 44% |

### Teacher
| group | total | passed | pass rate |
|---|---|---|---|
| google/gemma-4-26b-a4b-it | 43 | 29 | 67% |
| gpt-4.1-mini | 42 | 23 | 55% |
| gpt-4o-mini | 43 | 12 | 28% |
| gpt-5.4-mini | 32 | 17 | 53% |
| gpt-5.4-nano | 35 | 22 | 63% |
| qwen/qwen3.5-9b | 24 | 8 | 33% |

## Judge scores

- `gpt-5.4-mini` (n=219): alignment=6.22, grounding=6.76, naturalness=7.78, language=9.98, avg=7.68 | histogram(avg): 4:5 5:34 6:30 7:46 8:41 9:41 10:22
- `qwen/qwen3.5-9b` (n=219): alignment=7.72, grounding=7.21, naturalness=8.46, language=10.0, avg=8.34 | histogram(avg): 3:2 4:15 5:22 6:15 7:14 8:20 9:71 10:60

## Exec replay

- unique recorded calls: 698 (cache hits: 576, live: 122)
- recorded-ok calls: 730 replayed ok, 2 failed on replay
- recorded-fail calls that succeed now (recovered): 0

## Language-consistency catches (expected ~3 French mislabels)

- none found

## Rejection reasons histogram

- 47x judge: dimension scored < 5 (min=2)
- 25x judge: dimension scored < 5 (min=4)
- 18x judge: dimension scored < 5 (min=1)
- 14x judge: dimension scored < 5 (min=3)
- 4x judge: judge avg < 7.0 (gpt-5.4-mini=7.0, qwen/qwen3.5-9b=5.8)
- 3x judge: judge avg < 7.0 (gpt-5.4-mini=5.2, qwen/qwen3.5-9b=9.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=5.0, qwen/qwen3.5-9b=7.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=5.8, qwen/qwen3.5-9b=9.2)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=7.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.2, qwen/qwen3.5-9b=9.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=7.5, qwen/qwen3.5-9b=4.8)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=7.0, qwen/qwen3.5-9b=6.2)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=7.0, qwen/qwen3.5-9b=4.5)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=5.2)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=5.5, qwen/qwen3.5-9b=8.0)
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=5.5, qwen/qwen3.5-9b=9.0)
- 2x exec: replay failed: tomtom-poi-search (MCPError: [tomtom] tool tomtom-poi-search errored: {'content': [{'type': 'te
- 2x judge: judge avg < 7.0 (gpt-5.4-mini=6.2, qwen/qwen3.5-9b=6.2)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=9.2, qwen/qwen3.5-9b=4.8)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=5.8, qwen/qwen3.5-9b=9.0)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=6.0, qwen/qwen3.5-9b=9.2)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=6.5, qwen/qwen3.5-9b=7.2)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=8.8, qwen/qwen3.5-9b=3.8)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=8.5, qwen/qwen3.5-9b=6.2)
- 1x judge: judge avg < 7.0 (gpt-5.4-mini=5.5, qwen/qwen3.5-9b=6.2)

## Example rejected dialogs

- `tool_use-9581a3eadc6d` (ab_test.jsonl:1, tool_use/es) failed=['judge']: "Necesito ir a visitar a un cliente esta noche, con lluvia ligera y tráfico complicado por "
- `tool_use-605a00285c94` (ab_test.jsonl:4, tool_use/de) failed=['judge']: "Ich fahre von Sydney nach Melbourne und soll über Canberra und Albury routen. Kannst du mi"
- `tool_use-59bd3564eac1` (ab_test.jsonl:5, tool_use/pl) failed=['judge']: "Chciałbym pojechać z Warszawy do Zakopanego, ale proszę wyznaczyć mi trasę przez jakieś ma"
- `tool_use-99ee443bbe9d` (ab_test.jsonl:7, tool_use/fr) failed=['judge']: "Je veux aller de Bordeaux à Limoges en passant par la nationale, sans péage, et je veux fa"
- `tool_use-6109291c447c` (ab_test.jsonl:8, tool_use/de) failed=['judge']: "Ich möchte von Zaragoza nach Ciempozuelos fahren und dabei die schnellste Route wählen. Es"
- `tool_use-3402f9b212ac` (ab_test.jsonl:13, tool_use/es) failed=['judge']: "Quiero ir de Madrid a Lisboa por la ruta más rápida, pero sin peajes. También me gustaría "
- `tool_use-70d72d8ebb9b` (ab_test.jsonl:15, tool_use/de) failed=['judge']: "Ich muss morgen früh von München nach Hamburg fahren und brauche eine schnelle Route mit g"
- `tool_use-f9e3a1d1d71e` (ab_test.jsonl:16, tool_use/es) failed=['judge']: "Quiero ir de Shijiazhuang a Dingxi conduciendo, recorriendo aproximadamente 1220 km, y nec"

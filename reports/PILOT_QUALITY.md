# Pilot Quality Report — dataset_gen

Date: 2026-10-02 · Corpus: **42 accepted dialogs** (36 generated with final code)
Files: `pilot2c.jsonl` (6) + `pilot30.jsonl` (30) + `pilot_errrec.jsonl` (2) + `pilot_langfix.jsonl` (4), all append-mode.
Earlier partial runs (`pilot.jsonl`, `pilot2*.jsonl`) were intermediate-code validation batches, kept on disk but excluded from headline stats.

## Results

- **Pilot-6**: 6/6 OK (`pilot2c`). **Pilot-30**: 30/30 OK (`pilot30`), 7 weak attempts rejected by the quality gate and re-rolled, 0 job failures. **error_recovery**: 2/2 (`pilot_errrec`). **Lang-fix validation**: 4/4 (`pilot_langfix`).
- No OpenRouter model had to be dropped (no persistent 429s). All 8 teachers contributed.

### Counts (42 dialogs)

| mode | n | | lang | n | | teacher | n |
|---|---|---|---|---|---|---|---|
| tool_use | 35 | | pl | 11 | | gpt-4o-mini | 13 |
| no_tool | 5 | | de | 11 | | gpt-5.4-mini / nano | 7 / 7 |
| error_recovery | 2 | | es | 10 | | deepseek-v4-flash / glm-5.3-flash | 4 / 4 |
| | | | en | 6 | | gemma-4-26b / qwen3.5-9b | 3 / 3 |
| | | | fr | 4 | | gpt-4.1-mini | 1 |

### Tool-call stats

- **190 emitted calls, avg 4.5/dialog** (final-code corpus: 148 calls, avg 4.1, max 7).
- Executions: 190 total → 183 ok / 38 err pre-fix; final-code corpus 131 ok / 17 err (errors are genuine recovery fodder: HERE 400s, budget-exceeded).
- Most used: `here_geocode` (52), `here_directions` (28), `tomtom-geocode` (21), `nominatim_geocode` (20), `here_search_places` (12), `tomtom-waypoint-routing` (11), `osrm_route` (9).
- Executions by provider family: HERE 101 · TomTom 60 · Nominatim 20 · OSRM 9.

## Quality checks (criteria a–f)

- **(a) Language**: verified per-turn with a phrase heuristic + manual read. After the seed-language fix (below), 4/4 validation dialogs are clean; pl/de/es/fr user turns are native and assistant answers match. ⚠ `pilot30.jsonl` still contains **3 mixed-language dialogs** generated before the fix (fr turns in es/pl/en dialogs) — filter at the verify stage or regenerate.
- **(b) Real tool_calls**: gate-enforced; every tool_use/error_recovery dialog has actual `tool_calls` arrays with parseable JSON arguments (0 violations in 42). Prose-only teachers are rejected and re-rolled.
- **(c) Real registry JSON**: every `role:"tool"` message parses as JSON; content matches `exec_log` entries (verified programmatically; spot-checked against cache/registry).
- **(d) Final answer**: gate requires a non-empty closing assistant message; 0 violations. A post-loop guarantee closes dialogs that would otherwise end on a tool message.
- **(e) no_tool purity**: 5/5 no_tool dialogs contain **zero** tool_calls despite 14 tools being offered.
- **(f) Error recovery**: both errrec dialogs show a failed call (e.g. `here_directions` 400) followed by a provider switch (`tomtom-waypoint-routing`) and a grounded final answer.

## Example excerpt (pilot2c, de, gpt-5.4-mini)

> **User:** "Ich brauche bitte eine Route von Lyon nach Grenoble, möglichst ohne Maut, und mit zwei Pflicht-Stopps …"
> **Assistant:** → `osrm_route("4.8357,45.7640;5.7280,45.1885")` ∥ `here_search_places("Aire de repos", at≈45.5,5.0)` ×2 → results `{"distance_km": 106.9, "duration_min": 83}`, rest-area lists; 2nd hop `tomtom-traffic(bbox=…)` → incidents ("Gesperrt" D7).
> **Assistant (final):** "Die Strecke **Lyon → Grenoble** ist ungefähr **106,9 km** … ich schlage **Aire de Solaize** und **Aire de Service de Bron** vor; aktuell mehrere Verkehrsereignisse bei Lyon …"
> (numbers in the answer trace back to tool outputs)

## Issues found & fixes applied

1. **`llm_client.py`**: gpt-5.x/o-series reject `max_tokens` and `temperature≠1` → proactive rename to `max_completion_tokens` + temperature drop for `^(gpt-5|o[0-9])`, plus reactive 400-compat (retry without `temperature` / with renamed param on complaint). OpenRouter payloads untouched. Hard 400s no longer burn 5 backoff retries.
2. **User-sim quirks** (`dataset_gen.py`): role prefixes ("Driver:", "Użytkownik:") and wrapping quotes stripped; `DONE` detected after cleaning; turn-0 `DONE`/empty now retried (×3) with an explicit "open with the task" instruction.
3. **No-tool answers in tool_use mode**: stronger system prompt ("MUST call tools … answering from memory is forbidden") + acceptance gate with per-job re-roll (≤3 attempts).
4. **Degenerate dialogs saved**: acceptance gate — ≥1 user turn, final answer present, tool_use needs ≥1 call + ≥1 ok exec (and ≤4 failed execs → catches thrashing loops like a 20-call deepseek dialog), no_tool needs zero calls, errrec needs an error or clarification + recovery.
5. **Tool-budget spiral**: after budget exhaustion the tool loop now stops and the dialog closes in prose; system prompt forbids identical-call retries.
6. **Mixed-language dialogs**: non-`en` seeds now force their own language; phrase-based foreign-language check added to the gate.
7. **Junky seeds**: 25% of instruction seeds were malformed JSON blobs (doubled quotes, truncations) → repaired (origin/destination → "route from X to Y"), unusable ones skipped.
8. **Keyless provider nudge bug**: `preferred=keyless` only matched `osrm_*`; now includes `nominatim_*`/`open_meteo_*`.

## Remaining issues / follow-ups

- **⚠ Secret leakage in existing raw files**: HERE API errors embed the request URL incl. `apikey=…` → 6 occurrences in `pilot30.jsonl` (4) and `pilot_errrec.jsonl` (2) written **before** the redaction fix. Dataset_gen now redacts (`_redact`), cache is clean, but the two files must be scrubbed during verification (files are append-only here): `sed -E 's/(apikey|key|token)=[A-Za-z0-9_.-]+/\1=***/g'`.
- deepseek/qwen as **user-sim** sometimes return `DONE` at turn 0 (~1 in 6 attempts); gate + re-roll handle it, but they are cheaper as assistant-only teachers.
- `here_directions` 400s with some arg shapes (e.g. `via` list) — teacher recovers via TomTom; worth a `_normalize_args` case if it recurs often.
- error_recovery acceptance is strict: ~40% of attempts rejected because fuzzy geocoding silently "fixes" misspellings (no hard error) — consider seeding real API failures or ambiguous-place quirks with weaker geocoders.

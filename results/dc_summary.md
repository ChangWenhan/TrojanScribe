# D_c results (100 fixed questions)

## xlam-2-8b

clean correct: 100/100, clean ASR: 0/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **96/100** | 93 | 3 | 4 | 0 | 0 | 93/100 |
| PoisonedRAG | **65/100** | 64 | 1 | 34 | 1 | 0 | 64/100 |
| AgentPoison | **25/100** | 22 | 3 | 75 | 0 | 0 | 22/100 |
| CorruptRAG-AS | **79/100** | 79 | 0 | 21 | 0 | 0 | 79/100 |
| CorruptRAG-AK | **37/100** | 37 | 0 | 63 | 0 | 0 | 37/100 |

## qwen3-4b-instruct

clean correct: 99/100, clean ASR: 1/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **77/100** | 77 | 0 | 22 | 0 | 1 | 78/100 |
| PoisonedRAG | **32/100** | 32 | 0 | 67 | 0 | 1 | 33/100 |
| AgentPoison | **7/100** | 6 | 1 | 92 | 0 | 1 | 7/100 |
| CorruptRAG-AS | **34/100** | 33 | 1 | 65 | 0 | 1 | 34/100 |
| CorruptRAG-AK | **18/100** | 18 | 0 | 81 | 0 | 1 | 18/100 |

## gpt-oss-20b

clean correct: 99/100, clean ASR: 0/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **88/100** | 87 | 1 | 11 | 0 | 1 | 88/100 |
| PoisonedRAG | **27/100** | 27 | 0 | 72 | 0 | 1 | 27/100 |
| AgentPoison | **9/100** | 8 | 1 | 90 | 0 | 1 | 8/100 |
| CorruptRAG-AS | **52/100** | 51 | 1 | 47 | 0 | 1 | 52/100 |
| CorruptRAG-AK | **32/100** | 32 | 0 | 67 | 0 | 1 | 32/100 |

## olmo-3-7b

clean correct: 88/100, clean ASR: 3/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **75/100** | 75 | 0 | 12 | 1 | 12 | 86/100 |
| PoisonedRAG | **62/100** | 62 | 0 | 25 | 1 | 12 | 72/100 |
| AgentPoison | **34/100** | 34 | 0 | 53 | 1 | 12 | 42/100 |
| CorruptRAG-AS | **48/100** | 48 | 0 | 40 | 0 | 12 | 53/100 |
| CorruptRAG-AK | **20/100** | 19 | 1 | 68 | 0 | 12 | 25/100 |


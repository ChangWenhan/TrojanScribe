# D_c results (100 fixed questions)

## xlam-2-8b

clean correct: 92/100, clean ASR: 2/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **84/100** | 84 | 0 | 7 | 1 | 8 | 80/100 |
| PoisonedRAG | **46/100** | 45 | 1 | 46 | 0 | 8 | 45/100 |
| AgentPoison | **22/100** | 21 | 1 | 70 | 0 | 8 | 22/100 |
| CorruptRAG-AS | **62/100** | 62 | 0 | 30 | 0 | 8 | 64/100 |
| CorruptRAG-AK | **56/100** | 56 | 0 | 36 | 0 | 8 | 58/100 |

## qwen3-4b-instruct

clean correct: 98/100, clean ASR: 0/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **73/100** | 73 | 0 | 24 | 1 | 2 | 60/100 |
| PoisonedRAG | **34/100** | 33 | 1 | 64 | 0 | 2 | 30/100 |
| AgentPoison | **17/100** | 17 | 0 | 81 | 0 | 2 | 14/100 |
| CorruptRAG-AS | **40/100** | 40 | 0 | 58 | 0 | 2 | 36/100 |
| CorruptRAG-AK | **41/100** | 41 | 0 | 57 | 0 | 2 | 39/100 |

## gpt-oss-20b

clean correct: 96/100, clean ASR: 0/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **78/100** | 77 | 1 | 18 | 0 | 4 | 72/100 |
| PoisonedRAG | **29/100** | 29 | 0 | 67 | 0 | 4 | 26/100 |
| AgentPoison | **12/100** | 11 | 1 | 84 | 0 | 4 | 11/100 |
| CorruptRAG-AS | **56/100** | 56 | 0 | 39 | 1 | 4 | 55/100 |
| CorruptRAG-AK | **55/100** | 55 | 0 | 40 | 1 | 4 | 52/100 |

## olmo-3-7b

clean correct: 61/100, clean ASR: 1/100

every cell counts the same 100 questions. attack success = flip + silent. `still ok` = clean-correct and still correct; `pre-wrong` = already wrong before the attack (the victim's own baseline error).

| method | attack success | =flip | +silent | still ok | unclear | pre-wrong | ASR |
|---|---|---|---|---|---|---|---|
| ours | **50/100** | 50 | 0 | 4 | 7 | 39 | 71/100 |
| PoisonedRAG | **34/100** | 33 | 1 | 25 | 2 | 39 | 58/100 |
| AgentPoison | **19/100** | 19 | 0 | 38 | 4 | 39 | 28/100 |
| CorruptRAG-AS | **20/100** | 20 | 0 | 36 | 5 | 39 | 27/100 |
| CorruptRAG-AK | **15/100** | 15 | 0 | 39 | 7 | 39 | 24/100 |


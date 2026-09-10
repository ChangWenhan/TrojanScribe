#!/bin/bash
# run_corruptrag_eval.sh — CorruptRAG evaluation, dual-GPU but inject-serialized.
#
# CorruptRAG has 4 poison sets: {as, ak} x {hotpot, musique}. Each set is
# injected into its KB (marker-guarded, single writer), then the 4 victims are
# evaluated CONCURRENTLY on two GPUs:
#   local GPU  (localhost)        : xlam-2-8b, gpt-oss-20b
#   141 GPU    (REMOTE_URL)       : qwen3-8b, llama-3.1-8b
# KB + code live on the local machine; 141 is a pure LLM endpoint.
#
# Usage: bash experiments/run_corruptrag_eval.sh
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
LOGDIR=$ROOT/results/logs
mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8

LOCAL_URL=http://localhost:8000/v1
REMOTE_URL=http://192.168.31.141:8000/v1
# 141 password comes from the SSHPASS env var (sshpass standard) — never hardcode.
REMOTE_USER=hp
REMOTE_HOST=192.168.31.141

REC_HP="$ROOT/results/baseline_corruptrag_targets_hp.json"
REC_MUS="$ROOT/results/baseline_corruptrag_targets_mus.json"

[ -f "$REC_HP" ] || ( cd "$ROOT/experiments" && "$PY" -c "
import json
t = json.load(open('../data/targets/hotpotqa.json'))
recs = {q: {'question': t[q]['question'], 'gold': t[q]['correct answer'], 'wrong': t[q]['incorrect answer']} for q in t}
json.dump(recs, open('$REC_HP','w'), ensure_ascii=False)
print('hp records', len(recs))
" )
[ -f "$REC_MUS" ] || ( cd "$ROOT/experiments" && "$PY" -c "
import json
t = json.load(open('../research/frozen/musique_targets_59.json'))
recs = {q: {'question': t[q]['question'], 'gold': t[q]['gold'], 'wrong': t[q]['wrong']} for q in t}
json.dump(recs, open('$REC_MUS','w'), ensure_ascii=False)
print('mus records', len(recs))
" )

declare -A LOCAL_SPECS=(
  [xlam-2-8b]="/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  [gpt-oss-20b]="/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
)

serve_local () {  # $1 victim
  local vpath vname vparser vextra
  IFS='|' read -r vpath vname vparser vextra <<< "${LOCAL_SPECS[$1]}"
  pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  # shellcheck disable=SC2086
  nohup "$PY" "$VLLM" serve "$vpath" --port 8000 \
      --gpu-memory-utilization 0.85 \
      --served-model-name "$vname" --enable-auto-tool-choice --tool-call-parser "$vparser" $vextra \
      > "$LOGDIR/vllm_$vname.log" 2>&1 &
  for _ in $(seq 1 150); do
    curl -s --max-time 3 "$LOCAL_URL/models" 2>/dev/null | grep -q "\"id\": *\"$vname\"" && return 0
    sleep 5
  done
  echo "FAILED local serve $vname" >&2; return 1
}

serve_remote () {  # $1 victim (qwen3-8b|llama-3.1-8b)
  sshpass -e ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
      "bash -s $1" < "$ROOT/scripts/serve_141.sh" > "$LOGDIR/serve141_$1.log" 2>&1
  for _ in $(seq 1 30); do
    curl -s --max-time 3 "$REMOTE_URL/models" 2>/dev/null | grep -q "\"id\": *\"$1\"" && return 0
    sleep 5
  done
  echo "FAILED remote serve $1" >&2; return 1
}

inject_set () {  # $1 poison_json  $2 kb_dir  $3 kb_collection  $4 marker
  local marker="$4"
  if [ ! -f "$marker" ]; then
    ( cd "$ROOT/experiments" && "$PY" -c "
import chromadb
c = chromadb.PersistentClient(path='$2')
coll = c.get_collection('$3')
ids = coll.get(where={'is_poison': 1})['ids']
if ids: coll.delete(ids=ids)
" )
    ( cd "$ROOT/experiments" && AGENTIC_RAG_RUN_ID="corruptrag_$(basename $1 .json)" \
        "$PY" longtail_attack.py --phase inject-from --inject-from "$1" \
        --kb-dir "$2" --kb-collection "$3" \
        --results-name "corruptrag_inj_$(basename $1 .json)" 2>&1 | tail -1 )
    touch "$marker"
  fi
}

# run_eval_one: $1 victim  $2 kb_dir  $3 kb_collection  $4 target_rec  $5 results_name  $6 url
run_eval_one () {
  local out="$ROOT/results/$5.json"
  if [ -s "$out" ]; then echo "SKIP $5"; return; fi
  if [ "$1" = "qwen3-8b" ]; then export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}'
  else unset AGENTIC_RAG_CHAT_KWARGS; fi
  ( cd "$ROOT/experiments" && AGENTIC_RAG_RUN_ID="$5" AGENTIC_RAG_BASE_URL="$6" \
      "$PY" longtail_attack.py --phase eval-after --model "$1" \
      --target-records "$4" --kb-dir "$2" --kb-collection "$3" \
      --results-name "$5" 2>&1 | tail -1 )
  echo "DONE $5"
}

# evaluate one set on one GPU instance: local -> xlam + gptoss ; remote -> qwen3 + llama
# $1: set_name  $2 kb_dir  $3 kb_collection  $4 target_rec  $5 which(local|remote)
eval_side () {
  local setname=$1 kbd=$2 kbc=$3 rec=$4 side=$5
  if [ "$side" = local ]; then
    serve_local xlam-2-8b || return 1
    run_eval_one xlam-2-8b "$kbd" "$kbc" "$rec" "baseline_corruptrag_${setname}_eval_xlam-2-8b" "$LOCAL_URL"
    serve_local gpt-oss-20b || return 1
    run_eval_one gpt-oss-20b "$kbd" "$kbc" "$rec" "baseline_corruptrag_${setname}_eval_gpt-oss-20b" "$LOCAL_URL"
  else
    serve_remote qwen3-8b || return 1
    run_eval_one qwen3-8b "$kbd" "$kbc" "$rec" "baseline_corruptrag_${setname}_eval_qwen3-8b" "$REMOTE_URL"
    serve_remote llama-3.1-8b || return 1
    run_eval_one llama-3.1-8b "$kbd" "$kbc" "$rec" "baseline_corruptrag_${setname}_eval_llama-3.1-8b" "$REMOTE_URL"
  fi
}

# poison sets
declare -A POISON=(
  [as_hp]="$ROOT/results/baseline_corruptrag_as_60.json"
  [ak_hp]="$ROOT/results/baseline_corruptrag_ak_60.json"
  [as_mus]="$ROOT/results/baseline_corruptrag_as_musique.json"
  [ak_mus]="$ROOT/results/baseline_corruptrag_ak_musique.json"
)

for setname in as_hp ak_hp as_mus ak_mus; do
  PJ="${POISON[$setname]}"
  case "$setname" in
    *_hp) KBD="$ROOT/data/chroma"; KBC=hotpot_kb; REC="$REC_HP" ;;
    *_mus) KBD="$ROOT/data/chroma_musique"; KBC=musique_kb; REC="$REC_MUS" ;;
  esac
  echo "=== set $setname: inject + eval (dual GPU) ==="
  inject_set "$PJ" "$KBD" "$KBC" "$ROOT/results/corruptrag_${setname}_inj.flag"
  # run both sides concurrently; both read the same KB (already injected)
  eval_side "$setname" "$KBD" "$KBC" "$REC" local  > "$LOGDIR/cr_${setname}_local.log" 2>&1 &
  LPID=$!
  eval_side "$setname" "$KBD" "$KBC" "$REC" remote > "$LOGDIR/cr_${setname}_remote.log" 2>&1 &
  RPID=$!
  wait "$LPID"; wait "$RPID"
  echo "=== set $setname done ==="
done

# restore local default server
pkill -f "vllm serve" 2>/dev/null; sleep 8
nohup "$PY" "$VLLM" serve /mnt/disk/cwh/LLMs/xlam-2-8b-fc-r --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --served-model-name xlam-2-8b --enable-auto-tool-choice --tool-call-parser xlam --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja \
    > "$LOGDIR/vllm_xlam-2-8b_restore.log" 2>&1 &
echo "=== corruptrag eval all done ==="
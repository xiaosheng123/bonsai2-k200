#!/bin/bash
# orn6.sh — 经 8090 依次跑 6 条任务型提问, 记录原文与耗时
OUT=${OUT:-/home/caden/orn_engine/orn6.log}
: > "$OUT"
ask () {
  local i="$1" q="$2"
  local t0=$(date +%s.%N)
  local r=$(curl -s -m 1800 http://127.0.0.1:8090/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d "{\"model\":\"ornith-1.5-9b-k200\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64,\"temperature\":0}")
  local t1=$(date +%s.%N)
  echo "########## Q$i: $q" >> "$OUT"
  echo "$r" >> "$OUT"
  echo "[wall $(echo "$t1 - $t0" | bc)s]" >> "$OUT"
  echo "" >> "$OUT"
  echo "Q$i done ($(echo "$t1 - $t0" | bc)s)"
}
ask 1 "你好"
ask 2 "1+1等于几"
ask 3 "用三句话解释什么是光合作用"
ask 4 "把“今天天气不错，我们出去走走吧”翻译成英文"
ask 5 "写一个Python函数输入整数列表返回最大值索引(只给代码)"
ask 6 "我明天要交一份季度税务报告，列5条检查清单"
echo "ALL6_DONE" >> "$OUT"
echo ALL6_DONE

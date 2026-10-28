#!/bin/bash
cd "$HOME/momoe" || exit 1
# Start a long generation in background
( curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"Write a 300-word essay about Linux kernel scheduling. Be detailed."}],"max_tokens":512,"reasoning_effort":"none"}' > /tmp/long_resp.json ) &
curlbg=$!
peak_vram=0
for i in $(seq 1 60); do
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
  vram=${line%%,*}
  iv=${vram%%.*}
  if [ "$iv" -gt "$peak_vram" ] 2>/dev/null; then peak_vram=$vram; fi
  sleep 0.5
done
wait $curlbg
echo "Peak VRAM during generation: ${peak_vram} MiB / 12288 MiB"
python3 -c "import json;d=json.load(open('/tmp/long_resp.json'));print('completion tokens:',d['usage']['completion_tokens'])" 2>&1
echo "--- CPU snapshot ---"
top -bn1 | head -6
echo "--- free mem ---"
free -h | head -2

#!/bin/bash
# Watch worker startup: pass/fail detection with early error exit.
export PATH=$PATH:/usr/local/bin:/usr/bin
PF_POD=""
DEC_POD=""
for i in $(seq 1 120); do
  ts=$(date -u +%H:%M:%S)
  PF_POD=$(kubectl get pods -n glm53 -l glm53.nvidia.com/role=prefill -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  DEC_POD=$(kubectl get pods -n glm53 -l glm53.nvidia.com/role=decode -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  pf_st=$(kubectl get pod "$PF_POD" -n glm53 -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null | head -c 60)
  dec_st=$(kubectl get pod "$DEC_POD" -n glm53 -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null | head -c 60)
  # latest significant log lines
  pf_log=$(kubectl logs "$PF_POD" -n glm53 --tail 2>/dev/null | grep -oE 'Loading model|it/s|GiB/s|Capturing.*graphs|ValueError|Error.*' | tail -1)
  echo "[$ts] prefill: ${pf_st:0:40} ${pf_log:0:40} | decode: ${dec_st:0:40}"
  # failure detection
  err=$(kubectl logs "$PF_POD" -n glm53 --tail 50 2>/dev/null | grep -E 'ValueError|RuntimeError: Engine core' | tail -1)
  err2=$(kubectl logs "$DEC_POD" -n glm53 --tail 50 2>/dev/null | grep -E 'ValueError|RuntimeError: Engine core' | tail -1)
  if [ -n "$err" ]; then echo "PREFILL FAIL: $err"; break; fi
  if [ -n "$err2" ]; then echo "DECODE FAIL: $err2"; break; fi
  # success detection: both workers report Ready
  pf_ready=$(kubectl get pod "$PF_POD" -n glm53 -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  dec_ready=$(kubectl get pod "$DEC_POD" -n glm53 -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  if [ "$pf_ready" = "true" ] && [ "$dec_ready" = "true" ]; then echo "WORKERS READY"; break; fi
  sleep 120
done

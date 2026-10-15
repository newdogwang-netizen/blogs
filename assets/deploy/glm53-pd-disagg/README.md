# GLM-5.3-Flash PD-Disaggregation on 2x8 H100 — deployment artifacts

Runnable artifacts for the blog post
[_posts/2026-10-14-ai-dynamo-glm53-flash-pd-disaggregation-h100.md]({{ site.baseurl }}/cloud-native-ai-infra/)
(ai-Dynamo 1.5.0 + vLLM, Nebius mk8s, InfiniBand fabric-4).

Files are numbered in the order they are applied:

| # | File | What it does |
|---|------|--------------|
| 10 | `10-dynamo-values.yaml` | Helm values: operator only + etcd (`global.etcd.install=true`), grove/nats disabled |
| 20 | `20-storage-and-download.yaml` | Namespace, two 500Gi RWO PVCs (one per GPU node), two model-download Jobs pinned via anti-affinity |
| 30 | `30-dgd-deploy-h100.yaml` | The `DynamoGraphDeployment`: frontend + TP8 prefill + TP8 decode, NixlConnector producer/consumer, MTP7, privileged + hostPath `/dev/infiniband` |
| 40 | `40-watch-worker-startup.sh` | Startup watcher: polls pod state, greps logs for early ValueError, reports WORKERS READY |
| 50 | `50-verify-kv-transfer-pod.yaml` | Throwaway pod mounting a worker output PVC — used to inspect NCCL logs / smoke-test results |

## Apply order

```bash
# 1. Platform: operator + etcd (chart from ai-dynamo/dynamo v1.5.0, deploy/helm/charts/platform)
helm dependency update
helm upgrade --install dynamo-platform . -n dynamo --create-namespace \
    -f 10-dynamo-values.yaml --set global.etcd.install=true

# 2. Storage + model download (306 GB FP8 checkpoint, ~23 min per copy)
kubectl apply -f 20-storage-and-download.yaml

# 3. The deployment itself
kubectl apply -f 30-dgd-deploy-h100.yaml

# 4. Watch startup (weights + CUDA graph capture take tens of minutes)
bash 40-watch-worker-startup.sh

# 5. Smoke test
kubectl port-forward -n glm53 svc/glm53-flash-disagg-frontend 8000:8000
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

## Environment assumptions baked into these files

- Nebius mk8s, driverful GPU nodes (host driver 580.173.02, no GPU operator,
  no `rdma/ib` resource plugin). IB access is privileged + hostPath
  `/dev/infiniband` passthrough — adjust for your cluster.
- Node names are hardcoded in `30-dgd-deploy-h100.yaml`
  (`kubernetes.io/hostname` selectors). Change them.
- Per-node RWO model caches because Nebius network SSD has no RWX support.
- H100 80GB memory budget: `max-model-len 65536`, mem-util 0.90/0.93. Do NOT
  carry `--attention-backend FLASHMLA_SPARSE` over from the upstream H200
  recipe — it is not valid on H100 (see blog post, section 6).

## Known pitfall: operator rolling-update deadlock on GPU-exclusive nodes

Updating the DGD while old workers still hold all 8 GPUs per node deadlocks:
new pods stay Pending (no free GPUs), the operator never scales down the old
component until the new one is Ready. The clean fix is deleting the old
`DynamoComponentDeployment` objects directly — deleting the ReplicaSet does
nothing (the DCD controller recreates it from the old template).

```bash
kubectl get dynamocomponentdeployments -n glm53          # old + new coexist
kubectl delete dynamocomponentdeployment <old-hash> -n glm53
```

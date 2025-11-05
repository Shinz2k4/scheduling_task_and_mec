import os, time, itertools, statistic, argparse
import httpx, yaml
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

with open("./../configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
POOLS = {k.lower(): [f"http://{u}" for u in v] for k, v in cfg.get("services", {}).items()}
ITER = {k: itertools.cycle(v) for k, v in POOLS.items() if v}

METRICS = {"count": 0, "latencies": [], "start": time.time()}

@app.post("/infer/{model}")
async def route(model: str, file: UploadFile = File(...), port: str = Form(...)):
    model = model.lower()
    if model not in POOLS or not POOLS[model]:
        return JSONResponse({"error": f"No backends for {model}"}, status_code=503)
    tem_url = next(ITER[model])
    url = tem_url.split(":")[0]+":"+tem_url.split(":")[1]+":"+str(port)+ "/infer"
    print(f"do inference for {model} {url} {model} {port}")
    data = await file.read()
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=240.0) as client:
        r = await client.post(
                                url,
                                files={
                                    "file": (file.filename, data, file.content_type or "application/octet-stream"),
                                    "model": (None, model)  # gửi model như một form field
                                }
                            )
    dt = (time.perf_counter() - t0) * 1000

    METRICS["latencies"].append(dt)
    METRICS["count"] += 1

    return JSONResponse({"backend": url, "ms": dt, "payload": r.json()})

@app.get("/metrics")
async def metrics():
    dur = max(1e-6, time.time() - METRICS["start"])
    count = METRICS["count"]
    tput = count / dur
    lat = METRICS["latencies"]
    if lat:
        avg = sum(lat)/len(lat)
        p50 = statistics.median(lat)
        p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)
        p99 = statistics.quantiles(lat, n=100)[98] if len(lat) >= 100 else max(lat)
    else:
        avg = p50 = p95 = p99 = 0.0
    return {
        "requests": count,
        "throughput_rps": tput,
        "latency_ms": {"avg": avg, "p50": p50, "p95": p95, "p99": p99},
        "backends": {k: len(v) for k, v in POOLS.items()},
    }

@app.get("/health")
async def health():
    return {"status": "ok", "backends": {k: len(v) for k, v in POOLS.items()}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=str, default="10000")    
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=int(args.port))
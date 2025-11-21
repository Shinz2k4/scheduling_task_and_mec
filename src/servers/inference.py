
import argparse, asyncio, glob, os, random, statistics, time
import httpx
import yaml
from PIL import Image
import io
SHARED_PATH = "/shared" 

async def inference(id_picture:int, model:str, port_base:int = 10000):
    with open("./../configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg.get("scheduler").get("host", "")
    host = cfg.get('scheduler').get("host", "")
    port = cfg.get('scheduler').get("port", "")
    print(f"currnt port in inference phase {port_base}")
    str_lst = 100*(int(str(port_base)[-1])+1)
    port = int(str_lst) + port_base+ port
    
    #model post
    model_original_post = int(cfg.get("services", {}).get(model, [])[0].split(":")[-1])
    model_adjust_port = int(str_lst) + port_base+ model_original_post
    
    url = f"http://{host}:{port}/infer/{model}"
    print(f"do inference at url {url} with service port {model_original_post}")
    
    print(id_picture)
    picture = os.listdir(SHARED_PATH)[int(id_picture)]
    path = os.path.join(SHARED_PATH, picture)
    t0 = time.perf_counter()
    
    img = Image.open(path)
    # img = img.resize((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0) 
    print("read picture finished")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:                
            files = {"file": (os.path.basename(path), buf, "application/octet-stream")}
            data = {"port": str(model_adjust_port)}
            r = await client.post(url, files=files, data=data)
            r.raise_for_status()
        dt = (time.perf_counter() - t0)
        return dt, r.json()
    except Exception as e:
        print(f"error {e}")
        return None, None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id_picture", type=int, required=True)
    ap.add_argument("--model", type=str, default="resnet18")
    args = ap.parse_args()
    asyncio.run(inference(args.id_picture, args.model))
    
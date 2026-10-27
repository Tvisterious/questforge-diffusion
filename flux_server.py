import base64
import io
import os
import threading
import time
import argparse

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional

import importlib
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers import BitsAndBytesConfig as DBnB
from transformers import BitsAndBytesConfig as TBnB

app = FastAPI(title="QuestForge Image Server")

pipe = None
pipe_lock = threading.Lock()   # одна генерация за раз: GPU одна, очередь не нужна
ARGS = None


# ---------- OpenAI-совместимая схема запроса (+ наши удобные поля) ----------
class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")   # чужие/лишние поля не ломают запрос
    model: str = "flux"
    prompt: str
    n: int = 1
    size: str = "512x512"
    width: Optional[int] = None 
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    response_format: str = "b64_json"


# ---------- Опциональный токен: тот самый "путь + токен" из конфига ----------
@app.middleware("http")
async def auth(request: Request, call_next):
    token = os.environ.get("IMAGE_SERVER_TOKEN", "")
    if token and not request.url.path.startswith(("/health",)):
        if request.headers.get("authorization", "") != f"Bearer {token}":
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


def parse_size(size: str):
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 512, 512


# ---------- Загрузка модели ----------
def load_pipeline():
    global pipe
    dtype = torch.float16
    # ТОТ ЖЕ конфиг квантизации, что и на обучении
    q = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

    kwargs = {}
    if ARGS.quant == "nf4":
        kwargs["transformer"] = Flux2Transformer2DModel.from_pretrained(
            ARGS.base_model, subfolder="transformer",
            torch_dtype=dtype, quantization_config=DBnB(**q),
        )
        cfg = Flux2KleinPipeline.load_config(ARGS.base_model)
        te_cls = getattr(importlib.import_module("transformers"),
                         cfg["text_encoder"][1])
        try:
            kwargs["text_encoder"] = te_cls.from_pretrained(
                ARGS.base_model, subfolder="text_encoder",
                torch_dtype=dtype, device_map="cuda",
                quantization_config=TBnB(**q),
            )
        except Exception as e:
            print(f"[warn] TE не заквантован ({e}) — загрузится как есть")

    pipe = Flux2KleinPipeline.from_pretrained(
        ARGS.base_model, torch_dtype=dtype, **kwargs)

    if ARGS.lora:
        pipe.load_lora_weights(ARGS.lora, adapter_name="questforge")
        pipe.set_adapters(["questforge"], adapter_weights=[ARGS.lora_scale])

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
        # ПЕРЕНОСИМ VAE НА GPU
    try:
        pipe.vae.to("cuda")
        print("VAE перенесён на GPU")
    except Exception as e:
        print(f"VAE не влез на GPU ({e}), остаётся на CPU")
    
    pipe.set_progress_bar_config(disable=True)
    
    print("[ok] модель загружена")
    print("devices:",
          "transformer=", pipe.transformer.device,
          "text_encoder=", pipe.text_encoder.device,
          "vae=", pipe.vae.device, flush=True)


# ---------- Эндпоинты ----------
@app.get("/health")
def health():
    return {"status": "ok" if pipe else "loading"}


@app.get("/v1/models")
def models():
    return {"data": [{"id": "flux-klein-questforge", "object": "model"}]}


def _generate(req: ImageRequest):
    if pipe is None:
        raise HTTPException(503, "model is still loading")

    if req.width and req.height:
        w, h = req.width, req.height
    else:
        w, h = parse_size(req.size)
    w = min(w, ARGS.max_side)
    h = min(h, ARGS.max_side)

    steps = req.steps or ARGS.steps
    guidance = req.guidance_scale if req.guidance_scale is not None else ARGS.guidance
    gen = torch.Generator("cuda").manual_seed(ARGS.seed) if ARGS.seed >= 0 else None
    
    prompt = req.prompt
    if ARGS.trigger_word:
        prompt = f"{ARGS.trigger_word} {prompt}"

    t0 = time.time()

    def cb(pipe, step, timestep, callback_kwargs):
        print(
            f"  step {step} @ {time.time() - t0:.1f}s",
            flush=True,
        )
        return callback_kwargs

    with pipe_lock:
        image = pipe(
            prompt=prompt,
            width=w, height=h,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            callback_on_step_end=cb,
        ).images[0]
    print(f"[gen] total {time.time() - t0:.1f}s", flush=True)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}

@app.post("/v1/images/generations")
def images_generations(req: ImageRequest):
    return _generate(req)


@app.post("/image")
def image_alias(req: ImageRequest):
    return _generate(req)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="путь/HF-id базового Flux (diffusers)")
    p.add_argument("--lora", default=None, help="папка или .safetensors с LoRA-весами")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--trigger-word", default="", help="instance_prompt обучения, напр. QWRTY9")
    p.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    p.add_argument("--offload", action="store_true", help="cpu offload (только при --quant none)")
    p.add_argument("--fuse", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--port", type=int, default=9000)
    ARGS = p.parse_args()

    load_pipeline()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
# QuestForge Diffusion

Отдельный сервис генерации иллюстраций для образовательных квестов QuestForge. Принимает текстовый промпт слайда и возвращает картинку в стиле проекта (ч/б пиксель-арт) через OpenAI-совместимый API.

## Что это и зачем

Бэкенд QuestForge генерирует сценарий квеста, где у каждого слайда есть поле `picture` — промпт для иллюстрации. Этот сервис рисует эти иллюстрации локально, без внешних API и платных ключей:

- **База:** `black-forest-labs/FLUX.2-klein-base-4B` (Apache 2.0) — компактная диффузионная модель.
- **Стиль:** собственная LoRA, дообученная (DreamBooth, diffusers) на 25 ч/б пиксель-арт изображениях; триггер стиля — `QWRTY9. Black and white colors, pixelart.`.
- **Интерфейс:** OpenAI-совместимый `POST /v1/images/generations` — бэкенд QuestForge подключается одной строчкой в конфиге (`driver: openai_images`).

Сервер рассчитан на наличие GPU: веса квантуются в NF4 и помещаются в 6 ГБ VRAM.

## Требования к системе

| Компонент | Минимум | Проверено на |
|---|---|---|
| GPU | NVIDIA, 6 ГБ VRAM | RTX 3060 Laptop (6 ГБ) |
| CUDA | 12.x (для 13.x — bitsandbytes из main) | CUDA 13.4 |
| RAM | 12 ГБ | 16 ГБ |
| Диск | ~20 ГБ (веса модели + кэш HF) | — |
| Python | 3.10+ | 3.14 |

Пакеты: `torch`, `diffusers`, `transformers`, `accelerate`, `peft`, `bitsandbytes`, `fastapi`, `uvicorn`, `pillow`, `safetensors`.

## Развертывание и запуск

### 1. Установка

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> Для CUDA 13.x ставьте bitsandbytes из main:
> `pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git`

### 2. Веса

- Базовая модель скачивается автоматически с Hugging Face при первом запуске (кэш `~/.cache/huggingface`).
- LoRA-адаптер (`flux_lora_final/` с `pytorch_lora_weights.safetensors`) положите в `./models/` — это собственные веса, обученные в Kaggle (DreamBooth, rank 16, alpha 16, 1200 шагов, resolution 512).

### 3. Команда запуска

```bash
python flux_server.py \
  --base-model black-forest-labs/FLUX.2-klein-base-4B \
  --lora ./models/flux_lora_final \
  --trigger-word "QWRTY9. Black and white colors, pixelart." \
  --quant nf4 \
  --steps 12 \
  --max-side 384 \
  --port 9000
```

Порт откроется после загрузки и квантования весов (~1–2 минуты). Проверка: `curl http://localhost:9000/health`.

### Параметры CLI

| Флаг | Назначение | По умолчанию |
|---|---|---|
| `--base-model` | HF-id или путь к базе FLUX.2 klein 4B | обязателен |
| `--lora` | Папка/файл с LoRA-весами | — |
| `--lora-scale` | Сила адаптера | 1.0 |
| `--trigger-word` | Префикс-триггер стиля, добавляется к каждому промпту | — |
| `--quant` | `nf4` (6 ГБ VRAM) или `none` | `nf4` |
| `--offload` | CPU-offload (только при `--quant none`) | выкл |
| `--steps` | Шаги денонизации по умолчанию | 20 |
| `--guidance` | guidance scale по умолчанию | 3.5 |
| `--max-side` | Ограничение стороны картинки | 1024 |
| `--seed` | Фиксированный seed (-1 — случайный) | -1 |
| `--port` | Порт сервера | 9000 |

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Статус загрузки модели |
| GET | `/v1/models` | Список моделей (OpenAI-совместимость) |
| POST | `/v1/images/generations` | Генерация (OpenAI-формат, `b64_json`) |
| POST | `/image` | Алиас для ручной отладки |

Пример запроса:

```bash
curl -X POST http://localhost:9000/image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a knight in a neon alley", "size": "512x512", "steps": 10}'
```

Ответ: `{"created": ..., "data": [{"b64_json": "iVBOR..."}]}`.

## Интеграция с QuestForge Backend

```json
{
  "image": {
    "driver": "openai_images",
    "base_url": "http://<ip-gpu-хоста>:9000",
    "model": "flux-klein-questforge",
    "token": "",
    "width": 384,
    "height": 384,
    "steps": 10,
    "timeout_s": 300
  }
}
```

Применяется через `PUT /config` бэкенда без перезапуска контейнеров. Нюансы:
- `base_url` указывается **без** `/v1` (бэкенд добавляет сам);
- если бэкенд в Docker, а сервер на хосте — используйте gateway docker-сети (например, `172.19.0.1`), не `localhost`;
- защита токеном: задайте env `IMAGE_SERVER_TOKEN` на сервере и то же значение в поле `token` конфига.

## Использованные модель и адаптеры

| Компонент | Источник | Примечание |
|---|---|---|
| База | `black-forest-labs/FLUX.2-klein-base-4B` | Apache 2.0, ~4B параметров |
| LoRA стиля | собственная, `flux_lora_final` | DreamBooth/diffusers, rank 16, 25 изображений, триггер `QWRTY9` |
| Квантизация инференса | NF4 + double quant, fp16 compute | совпадает с конфигом обучения |

## Troubleshooting

| Симптом | Причина | Решение |
|---|---|---|
| `PEFT backend is required` | не установлен peft | `pip install -U peft` |
| `unexpected keyword 'negative_prompt'` | FLUX не поддерживает негативные промпты | не передавать `negative_prompt` в пайплайн |
| OOM на этапе VAE (после последнего шага) | не хватает VRAM/RAM под декодер | уменьшить `--max-side` и `width/height` в конфиге; `torch.cuda.empty_cache()` после генерации |
| 404 на `/v1/v1/images/generations` | дублирование `/v1` в `base_url` | убрать `/v1` из `base_url` |
| Первая генерация очень долгая | прогрев CUDA-ядер | норма, последующие быстрее |
| Стиль не применяется | триггер не в промпте | задать `--trigger-word` либо дописывать слова стиля в промпт |

## Ограничения

- Одна генерация за раз (глобальная блокировка GPU).
- Разрешение до 384–768 px на 6 ГБ VRAM; для пиксель-арта этого достаточно.
- LoRA обучена на 512×512 — генерация сильно большего размера может терять стиль.
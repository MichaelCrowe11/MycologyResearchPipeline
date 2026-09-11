# Mycology Research Pipeline

Flask web app for recording mushroom samples and analyses, with demo image identification, a scikit-learn bioactivity model, and lookups against PubMed, GBIF, and iNaturalist.

## Status

experimental

Development stopped on 2025-11-09 (last commit on `main`, from `git log`; that commit added Fly.io launch files, and the last code change was 2025-05-28). On 2026-09-10 the app installed from the lockfile and served pages with no configuration. The parts that do not work are listed under Limits: species identification returns demo values, the bioactivity model has no trained weights, nine of fifteen tests error, and the Fly.io app is down.

## Install and first run

Run on 2026-09-10 with uv 0.11.23 and Python 3.13.14, from a fresh clone:

```
uv sync --frozen
# Installed 3.3 s into .venv from uv.lock

uv pip install -p .venv/bin/python -r requirements-dev.txt
# pytest is not in pyproject.toml; this file adds it

.venv/bin/python -m pytest -q tests
# 6 passed, 9 errors in 2.57s
# every tests/test_api.py case errors: ImportStringError ... No module named 'testing'

.venv/bin/gunicorn --bind 127.0.0.1:5077 --workers 1 main:app
# worker booted in about 12 s (cv2 4.11.0 import, Prometheus metrics init)
# wrote instance/mycology_research.db (SQLite) with no env vars set

curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/         # 200
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/docs     # 200 (Swagger UI, 7 API paths)
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/api/health   # 200
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/api/samples  # 200 (empty list)
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/auth/login   # 200
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/metrics      # 200
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5077/health       # 404
```

Not run: the Docker build, `docker-compose.yml`, a Fly.io deploy, user registration, image upload, batch jobs, Stripe checkout, OpenAI calls, literature fetches, and `import_research_kit.py`.

Optional environment variables are listed in `.env.example`. `DATABASE_URL` switches from SQLite to Postgres. `OPENAI_API_KEY` enables the assistant pages in `ai_routes.py`. `STRIPE_*` keys and price IDs enable `/payment`.

## What runs today

- `app.py` builds the Flask app, registers six blueprints, and creates the tables in `models.py` (samples, compounds, analyses, batch_jobs, versions, research_logs, literature_references, users, memberships, subscriptions, payments, oauth_tokens, ai_assistant_queries).
- `api_routes.py`: `/api/health`, `/api/samples`, `/api/samples/<id>`, `/api/analyses/<id>`, `/api/process`, `/api/batch`, `/api/batch/<id>`. Documented by flasgger at `/docs`.
- `web_routes.py`: pages for samples, analyses, batch jobs, research logs, literature search, a parameter generator, image analysis, and a prediction dashboard (39 templates).
- `auth_routes.py`: login, register, profile, API keys, membership pages (Flask-Login).
- `computer_vision.py`: OpenCV image loading, Otsu segmentation, and contour measurements (area, perimeter, bounding box, circularity).
- `model.py`: a scikit-learn RandomForest wrapper with fit, predict, save, and load.
- `scientific_databases.py`, `literature.py`, `fetch_mycology_literature.py`: request code for PubMed, GBIF, iNaturalist, MycoBank, and Index Fungorum. Not exercised in this run.
- `monitoring.py`: Prometheus metrics at `/metrics`.
- `tests/test_models.py`: 6 passing tests.

## Limits

- Species identification is a demo. `identify_species` in `computer_vision.py` (lines 215 to 247) picks a name from a fixed list by the image's average color and reports a confidence drawn from `np.random`. The code comments say "For demonstration purposes, we'll return mock results." Do not use its output to identify a mushroom or to decide whether one is safe to eat.
- Bioactivity prediction has no trained model. `ml_bioactivity.py` fills missing inputs with fixed placeholder numbers ("For demonstration, we'll create synthetic features", line 72). `MODEL_PATH=models/` in `.env.example` points at a directory that does not exist in the repository.
- The old README claimed "30,000+ authentic bioactivity records". The repository holds three CSV files in `research_kit/` of 178, 113, and 211 bytes.
- `ai_assistant.py` calls the OpenAI API with model `gpt-3.5-turbo` and needs `OPENAI_API_KEY`. Without it the module logs a warning and requests fail.
- `payment_routes.py` falls back to placeholder Stripe price IDs such as `price_1OXyZ2ABC123DEF456GHI7J` when the `STRIPE_PRICE_*` variables are unset. Checkout cannot work with those values. Two handlers carry `TODO: Implement notification system`.
- `tests/test_api.py` calls `create_app('testing')`, but `create_app` expects a config object, so all nine API tests error.
- `Dockerfile` probes `/health`, which returns 404. The health route is `/api/health`. The Dockerfile also installs from `requirements.txt` (Flask 3.0.0 pinned) while `uv.lock` resolves `pyproject.toml` (Flask 3.1.1 or newer), so the Docker image and the uv environment differ.
- `fly.toml` names the app `mycologyresearchpipeline`. On 2026-09-10 the hostname resolved to Fly's edge but connections on ports 80 and 443 were reset. Nothing is served there.
- The old README linked docs.mycologyresearch.com, support and security addresses at mycologyresearch.com, and a Discord. docs.mycologyresearch.com does not resolve, and mycologyresearch.com is a third-party site not connected to this repository.
- `GET /api/samples` answered without authentication in this run; the old README's statement that all API endpoints require a JWT is not what the code does.
- The `.devcontainer` and `.replit` files describe the Replit workspace this was written in.

## License and contact

MIT License (see `LICENSE`, copyright 2025 Mycology Research Pipeline Contributors).

Contact: michael@crowelogic.com

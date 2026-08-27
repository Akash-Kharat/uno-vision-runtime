# UNO Vision Runtime — Backend

Industrial edge-AI vision runtime backend built with FastAPI.

## Python Version Requirements

* Minimum supported Python version: **3.11**.
* Development environments may use Python 3.11 or newer.
* The exact Python version on the Arduino UNO Q target must be verified during hardware validation.

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Run tests

```bash
python -m pytest
```

## Configuration

Configuration is loaded from environment variables. Copy `.env.example` to `.env` to customise:

| Variable      | Default              | Description            |
| ------------- | -------------------- | ---------------------- |
| `APP_NAME`    | `uno-vision-runtime` | Application name       |
| `APP_VERSION` | `0.1.0`              | Application version    |
| `API_HOST`    | `0.0.0.0`            | Server bind address    |
| `API_PORT`    | `8000`               | Server bind port       |
| `LOG_LEVEL`   | `INFO`               | Logging level          |

A `.env` file is **not required** — defaults are used when not present.

## API Endpoints

| Method | Path              | Description                |
| ------ | ----------------- | -------------------------- |
| GET    | `/`               | Service information        |
| GET    | `/api/v1/health`  | Health check               |

## Project Structure

```
backend/
├── app/
│   ├── api/          # Route handlers
│   ├── core/         # Logging, exceptions, lifecycle
│   ├── schemas/      # Pydantic response models
│   ├── config.py     # Environment-based configuration
│   └── main.py       # Application factory
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

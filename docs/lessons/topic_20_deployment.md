# Тема 20: Deployment — деплой LLM-приложений

> **Пререквизиты:** [Тема 10: Production-паттерны](topic_10_production_patterns.md), знакомство с Docker
> **Зависимости:** `docker`, `langserve[all]`, `uvicorn`, `gunicorn`

---

## Теория

### 1. Особенности деплоя LLM-приложений

Деплой приложения, использующего LLM, принципиально отличается от деплоя обычного REST API. Обычный эндпоинт отвечает за 50-200мс, потребляет мегабайты памяти и ведёт себя детерминированно. LLM-приложение ломает все три предположения.

**Время ответа.** Вызов LLM API занимает 5-60 секунд в зависимости от длины контекста и генерации. Assessment запрос с большой рубрикой и эссе студента — это 3000-5000 input tokens + 500-2000 output tokens. Claude Sonnet обрабатывает это за 10-30 секунд. Это значит, что стандартные таймауты nginx (60 секунд), AWS ALB (60 секунд), Cloudflare (100 секунд) — впритык или недостаточны. Один медленный запрос к LLM API может триггернуть timeout вашего reverse proxy, и клиент получит 504 Gateway Timeout вместо результата оценки.

**Потребление памяти.** Если приложение загружает embedding-модель локально (например, для vector store), это +500MB-2GB RAM при старте. Даже без локальных моделей, обработка больших документов в памяти (PDF-парсинг, chunking) может съедать сотни мегабайт. Контейнер с 256MB лимитом, привычный для микросервисов, будет OOMKilled.

**Недетерминированность.** Один и тот же промпт при temperature > 0 даёт разные ответы. Это нормально для LLM, но усложняет тестирование, мониторинг и дебаг в production. Нельзя просто сравнить expected output — нужны метрики качества, а не точные совпадения.

**Стоимость.** LLM API стоит денег. Claude Sonnet: $3/M input, $15/M output tokens. Один assessment запрос — $0.01-0.02. Тысяча оценок в день — $10-20. Звучит немного, но баг в коде (бесконечный цикл retry, дублированные запросы) может привести к счёту в тысячи долларов за ночь. Мониторинг расходов — не опция, а необходимость.

**Stateless vs Stateful.** Типичный LLM API — stateless: получил запрос, обработал, вернул результат. Это идеально для горизонтального масштабирования. Но conversation-based приложения (чат-бот, tutor) требуют хранения истории диалога — они stateful и нуждаются в session affinity или внешнем хранилище сессий (Redis).

Чеклист перед деплоем LLM-приложения:

| Компонент | Что проверить |
|---|---|
| Environment variables | Все API ключи через env, не в коде |
| Health checks | `/health/live`, `/health/ready` реализованы |
| Timeouts | Reverse proxy timeout ≥ 120s |
| Logging | Структурированные логи, без PII |
| Error handling | Все LLM-вызовы в try/except |
| Rate limiting | Защита от злоупотреблений |
| Cost monitoring | Трекинг токенов и расходов (LangFuse) |
| Secrets | .env не в Docker image, не в git |

### 2. Docker — контейнеризация FastAPI + LLM

Docker решает главную проблему деплоя: «у меня локально работает». Контейнер содержит операционную систему, Python, все зависимости и код приложения — одинаковые на машине разработчика, CI/CD и production-сервере.

**Выбор базового образа.** Для Python-приложений есть два основных варианта:

- `python:3.12-slim` — минимальный Debian с Python. Размер ~150MB. Подходит для большинства FastAPI-приложений, где все зависимости — pure Python wheels.
- `python:3.12-bookworm` — полный Debian. Размер ~900MB. Нужен, если зависимости требуют C-компиляцию: `numpy`, `scipy`, `tokenizers`, `chromadb` с C-расширениями.

Правило: начинайте с `slim`. Если `pip install` падает с ошибкой компиляции — переходите на полный образ или добавьте `build-essential` в slim.

**Multi-stage build.** Ключевая оптимизация. Первый этап (builder) устанавливает зависимости, включая build tools. Второй этап (runtime) копирует только готовые пакеты, без компилятора и исходников. Результат: образ в 2-3 раза меньше.

```dockerfile
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

COPY --from=builder /install /usr/local

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY app/ ./app/
COPY scripts/ ./scripts/

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching.** Docker кэширует каждый слой (каждую инструкцию). Если файл не изменился, слой берётся из кэша. Порядок COPY критичен:

```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
```

`requirements.txt` меняется редко → `pip install` кэшируется. Код приложения меняется часто → пересобирается только последний слой. Если поменять порядок (сначала `COPY app/`, потом `requirements.txt`), любое изменение кода инвалидирует кэш `pip install` — каждая сборка будет устанавливать все зависимости заново.

**Non-root user.** По умолчанию контейнер работает под root. Это плохо: если злоумышленник получит RCE через уязвимость, он получит root в контейнере. Создаём непривилегированного пользователя:

```dockerfile
RUN useradd --create-home appuser
USER appuser
```

Все последующие инструкции (CMD, ENTRYPOINT) выполняются под `appuser`. Важно: файлы, скопированные до `USER`, принадлежат root — appuser может их только читать. Это нормально для кода приложения, но если нужна запись (логи, кэш), создайте директорию с правильными правами до переключения пользователя.

**.dockerignore.** Аналог `.gitignore` для Docker build context. Без него `docker build` отправляет всю директорию проекта демону Docker, включая `.git` (сотни мегабайт), `__pycache__`, `.env` (секреты!), `data/` (датасеты).

```
.git
.env
.env.*
__pycache__
*.pyc
.venv
data/
*.md
.mypy_cache
.ruff_cache
.pytest_cache
node_modules
```

Критично: `.env` в `.dockerignore` предотвращает утечку API-ключей в Docker image. Даже если `.env` не используется в COPY, он всё равно попадает в build context и теоретически доступен.

**Оптимизация размера образа:**

| Техника | Экономия |
|---|---|
| Multi-stage build | 40-60% |
| `--no-cache-dir` при pip install | 10-20% |
| slim вместо full base | 70% от base |
| .dockerignore | build speed |
| Нет dev-зависимостей | 10-30% |

### 3. Docker Compose — полный стек

Реальное LLM-приложение — это не один контейнер. Assessment API нуждается в vector store (ChromaDB), кэше (Redis), трекинге (LangFuse + PostgreSQL). Docker Compose оркестрирует все эти сервисы локально и в staging.

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8100
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      chromadb:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:0.5.23
    ports:
      - "8100:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=False
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - backend

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse_secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - backend

  langfuse:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse_secret@postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: mysecret
      SALT: mysalt
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - backend

volumes:
  chroma_data:
  redis_data:
  postgres_data:

networks:
  backend:
    driver: bridge
```

**Networking.** Все сервисы в одной сети `backend` видят друг друга по имени сервиса. `app` обращается к ChromaDB как `http://chromadb:8000`, к Redis как `redis://redis:6379`. Внешние порты (`8100:8000` для chromadb) нужны только для отладки с хоста — в production их можно не экспонировать.

**Volumes.** Без volumes данные ChromaDB, Redis и PostgreSQL теряются при перезапуске контейнера. Named volumes (`chroma_data`, `redis_data`, `postgres_data`) хранят данные на диске хоста и переживают `docker-compose down`.

**Health checks и depends_on.** `depends_on` с `condition: service_healthy` гарантирует, что `app` запустится только когда ChromaDB и Redis полностью готовы. Без этого приложение может стартовать раньше базы и упасть при первом запросе.

**Profiles.** Docker Compose поддерживает профили для разных окружений. В dev нужны все сервисы, включая LangFuse. В production LangFuse и PostgreSQL могут быть отдельными managed-сервисами:

```yaml
services:
  langfuse:
    profiles:
      - dev
      - monitoring
    # ...

  postgres:
    profiles:
      - dev
      - monitoring
    # ...
```

Запуск: `docker compose --profile dev up` поднимет всё, `docker compose up` — только `app`, `chromadb`, `redis`.

### 4. ASGI Server — uvicorn и gunicorn

FastAPI — это ASGI-фреймворк. Ему нужен ASGI-сервер для обработки HTTP-запросов. Два основных варианта: uvicorn (для разработки и простых деплоев) и gunicorn с uvicorn workers (для production).

**Uvicorn** — лёгкий async ASGI-сервер на базе `uvloop` и `httptools`. Быстрый, простой, идеальный для разработки:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` автоматически перезапускает при изменении кода. Никогда не используйте в production — он сканирует файловую систему и добавляет overhead.

**Gunicorn + uvicorn workers** — production-конфигурация. Gunicorn — mature process manager, который управляет pool'ом worker-процессов. Каждый worker — отдельный процесс с собственным event loop:

```bash
gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
```

**Сколько workers?** Классическая формула для CPU-bound приложений: `2 * CPU_CORES + 1`. Но LLM-приложение — I/O-bound: worker ждёт ответа от LLM API 10-30 секунд, не нагружая CPU. Кажется, можно поставить больше workers. Но каждый worker — отдельный процесс со своим Python interpreter'ом и памятью (~100-200MB). Если приложение загружает embedding-модель, каждый worker загрузит свою копию (+500MB-2GB на worker).

Практические рекомендации:

| Конфигурация | Workers | Когда использовать |
|---|---|---|
| Только LLM API calls | 2-4 | Малая memory footprint |
| LLM API + embedding model | 1-2 | Высокое потребление памяти |
| Batch processing | 1 | Один worker, много async tasks |

**Timeout** — критичный параметр. Gunicorn по умолчанию убивает worker через 30 секунд. LLM запрос может занять 60 секунд. Решение: `--timeout 120` или выше. Но не ставьте слишком большой timeout (600+), иначе зависший worker будет занимать ресурсы бесконечно.

**Graceful shutdown.** При перезапуске (deploy, scaling) gunicorn отправляет SIGTERM workers. `--graceful-timeout 30` даёт 30 секунд на завершение текущих запросов. Для LLM это мало — в идеале `--graceful-timeout 60`, чтобы длинный assessment запрос успел завершиться.

**Keep-alive.** `--keep-alive 5` — время (секунды), в течение которого keep-alive соединение остаётся открытым. За load balancer'ом обычно 2-5 секунд. Если keep-alive сервера короче, чем у load balancer'а, возникают race condition'ы и спорадические 502 ошибки.

Полный `entrypoint.sh` для production:

```bash
#!/bin/bash
set -e

WORKERS=${WORKERS:-2}
TIMEOUT=${TIMEOUT:-120}
GRACEFUL_TIMEOUT=${GRACEFUL_TIMEOUT:-60}
BIND=${BIND:-0.0.0.0:8000}
LOG_LEVEL=${LOG_LEVEL:-info}

exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "$WORKERS" \
    --bind "$BIND" \
    --timeout "$TIMEOUT" \
    --graceful-timeout "$GRACEFUL_TIMEOUT" \
    --keep-alive 5 \
    --log-level "$LOG_LEVEL" \
    --access-logfile - \
    --error-logfile -
```

`exec` заменяет shell-процесс на gunicorn — контейнер получает PID 1 для gunicorn, что важно для корректной обработки SIGTERM при остановке контейнера.

### 5. Health Checks — мониторинг компонентов

Health checks — обязательный элемент production-деплоя. Без них оркестратор (Docker, Kubernetes) не знает, работает ли приложение. Контейнер может быть «запущен» (процесс жив), но не готов обрабатывать запросы (LLM API недоступен, vector store ещё загружается).

Три типа проверок:

**Liveness** (`/health/live`) — «приложение не зависло». Самый простой check: если endpoint отвечает 200, процесс жив. Если не отвечает — перезапустить контейнер. Не должен проверять внешние зависимости — только то, что Python-процесс responsive.

**Readiness** (`/health/ready`) — «приложение готово принимать трафик». Проверяет все зависимости: LLM API доступен, ChromaDB отвечает, Redis работает. Если readiness check fails, оркестратор убирает pod/контейнер из load balancer'а, но не перезапускает — возможно, проблема временная (LLM API rate limit).

**Startup** (`/health/startup`) — «первичная инициализация завершена». Нужен, если приложение долго стартует (загрузка embedding-модели, прогрев кэша). Kubernetes проверяет startup probe и переключается на liveness/readiness только после успеха.

**Кэширование health checks.** LLM API проверять на каждый health check нельзя — это расходует rate limit и добавляет задержку. Решение: кэшировать результат проверки на 30-60 секунд. Если последняя проверка была меньше 30 секунд назад и она была успешной — вернуть cached result.

Паттерн реализации:

```python
from datetime import datetime, timedelta

_health_cache: dict[str, tuple[bool, datetime]] = {}
CACHE_TTL = timedelta(seconds=30)


def _is_cached(component: str) -> bool | None:
    if component not in _health_cache:
        return None
    is_healthy, checked_at = _health_cache[component]
    if datetime.now() - checked_at > CACHE_TTL:
        return None
    return is_healthy


def _update_cache(component: str, is_healthy: bool):
    _health_cache[component] = (is_healthy, datetime.now())
```

**Агрегация результатов.** Readiness check проверяет несколько компонентов и возвращает общий статус. Если хотя бы один компонент down — весь сервис not ready:

```python
async def check_all_components() -> dict:
    results = {}

    results["llm_api"] = await check_llm_api()
    results["chromadb"] = await check_chromadb()
    results["redis"] = await check_redis()

    all_healthy = all(r["status"] == "healthy" for r in results.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "components": results,
    }
```

Ответ выглядит так:

```json
{
  "status": "healthy",
  "components": {
    "llm_api": {"status": "healthy", "latency_ms": 245},
    "chromadb": {"status": "healthy", "latency_ms": 12},
    "redis": {"status": "healthy", "latency_ms": 3}
  }
}
```

### 6. LangServe — деплой chains как REST API

LangServe — библиотека от LangChain, которая автоматически создаёт REST API из любого Runnable (chain, LLM, retriever). Одной строкой кода вы получаете полноценный API с invoke, stream, batch и встроенный playground.

```python
from langserve import add_routes

add_routes(app, my_chain, path="/assess")
```

Этот один вызов создаёт 6 эндпоинтов:

| Endpoint | Метод | Описание |
|---|---|---|
| `/assess/invoke` | POST | Синхронный вызов chain |
| `/assess/stream` | POST | Streaming ответ (SSE) |
| `/assess/batch` | POST | Батч-обработка нескольких inputs |
| `/assess/input_schema` | GET | JSON Schema входных данных |
| `/assess/output_schema` | GET | JSON Schema выходных данных |
| `/assess/playground` | GET | Интерактивный UI для тестирования |

**Playground** — killer feature для прототипирования. Открываете `http://localhost:8000/assess/playground` в браузере и тестируете chain через веб-интерфейс. Видите input, output, промежуточные шаги. Идеально для demo и debug.

**Input/Output types.** LangServe автоматически генерирует JSON Schema из типов chain. Если chain принимает Pydantic-модель, playground покажет форму с полями. Можно указать типы явно:

```python
from pydantic import BaseModel


class AssessInput(BaseModel):
    student_work: str
    rubric: str


class AssessOutput(BaseModel):
    score: int
    feedback: str


add_routes(
    app,
    assessment_chain,
    path="/assess",
    input_type=AssessInput,
    output_type=AssessOutput,
)
```

**Когда использовать LangServe:**

- Быстрый прототип — за минуты получить работающий API
- Demo для стейкхолдеров — playground не требует Postman/curl
- Внутренний tooling — не нужен сложный auth и rate limiting
- ML-команда выкатывает chain, backend-команда интегрирует

**Когда НЕ использовать LangServe:**

- Нужна кастомная аутентификация и авторизация (multi-tenant)
- Сложная бизнес-логика вокруг chain (conditional routing, A/B testing)
- Высоконагруженный production с кастомным rate limiting и circuit breaking
- Нужен контроль над форматом ответа (свой error handling, envelope)

LangServe удобен как отдельный сервис-песочница, где можно тестировать chains. Для production API с кастомной логикой часто пишут вручную через FastAPI или другой фреймворк — это даёт полный контроль.

### 7. Environment Management и Secrets

Управление конфигурацией и секретами — одна из самых частых причин security-инцидентов. API-ключ к Claude стоит $0 для кражи и может стоить тысячи долларов для владельца.

**Pydantic Settings** — правильный способ управления конфигурацией в FastAPI:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
    )

    anthropic_api_key: str
    openai_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    chroma_host: str = "localhost"
    chroma_port: int = 8100
    redis_url: str = "redis://localhost:6379/0"

    log_level: str = "info"
    environment: str = "development"
```

С `env_prefix="APP_"` переменные окружения ищутся с префиксом: `APP_ANTHROPIC_API_KEY`, `APP_MODEL_NAME`. Это предотвращает конфликты имён с другими сервисами.

**Разные конфигурации для разных окружений:**

| Файл | Окружение | Ключи | LLM |
|---|---|---|---|
| `.env` | Local dev | Test keys, low limits | Claude Haiku (дешевле) |
| `.env.staging` | Staging | Test keys, prod-like | Claude Sonnet |
| `.env.prod` | Production | Prod keys, high limits | Claude Sonnet |

В Docker: `docker run --env-file .env.prod ...` или `docker compose --env-file .env.staging up`.

**Docker secrets.** Для production в Swarm или Kubernetes `.env` файлы недостаточны — они видны в `docker inspect`. Docker secrets монтируются как файлы в `/run/secrets/`:

```yaml
services:
  app:
    secrets:
      - anthropic_api_key
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key

secrets:
  anthropic_api_key:
    file: ./secrets/anthropic_key.txt
```

**Ротация ключей без downtime.** Anthropic и OpenAI позволяют создавать несколько API-ключей одновременно. Процедура ротации:

1. Создать новый API-ключ в консоли провайдера
2. Обновить секрет в production (env var, secret manager)
3. Rolling restart — каждый pod перезапускается с новым ключом
4. Убедиться, что все pod'ы используют новый ключ
5. Отозвать старый ключ

**Правила логирования:**

- НИКОГДА не логировать API-ключи (даже частично)
- НИКОГДА не логировать полное содержимое промптов, если там может быть PII
- Логировать метаданные: model, token count, latency, status code
- Маскировать ключи в конфигурации: `sk-ant-...****`

```python
import logging

logger = logging.getLogger(__name__)


def log_llm_call(model: str, input_tokens: int, output_tokens: int, latency_ms: float):
    logger.info(
        "llm_call",
        extra={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": _estimate_cost(model, input_tokens, output_tokens),
        },
    )
```

### 8. Scaling и Production Patterns

Когда assessment API получает больше трафика, чем один контейнер может обработать, нужно масштабирование. LLM-приложения масштабируются иначе, чем обычные API — bottleneck не CPU, а rate limits и latency LLM API.

**Horizontal scaling.** Самый простой подход — несколько реплик за load balancer'ом:

```
Client → Nginx (LB) → [App-1, App-2, App-3] → LLM API
```

Stateless LLM API — любая реплика может обработать любой запрос. Docker Compose:

```yaml
services:
  app:
    deploy:
      replicas: 3
```

Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-app
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: app
          image: llm-app:latest
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

**Connection pooling.** Каждый вызов LLM API — HTTP-запрос. Создавать новое TCP-соединение на каждый запрос — расточительство. Используйте общий `httpx.AsyncClient` с connection pool:

```python
import httpx

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client
```

LangChain по умолчанию создаёт свой HTTP-клиент, но можно передать кастомный через `http_client` параметр.

**Background tasks.** Если assessment занимает 30 секунд, а пользователь не хочет ждать — используйте фоновую обработку:

```
POST /assess → 202 Accepted, {"task_id": "abc123"}
GET  /assess/abc123 → {"status": "processing"} или {"status": "done", "result": {...}}
```

Реализация через Celery, ARQ или даже простой `asyncio.create_task` + Redis для хранения результатов. ARQ — лёгкая альтернатива Celery для async Python:

```python
from arq import create_pool
from arq.connections import RedisSettings


async def assess_background(ctx, student_work: str, rubric: str):
    result = await assess(student_work, rubric, ctx["chain"])
    await ctx["redis"].set(f"result:{ctx['job_id']}", result.model_dump_json())


class WorkerSettings:
    functions = [assess_background]
    redis_settings = RedisSettings(host="redis")
    job_timeout = 120
```

**Rate limiting.** Защита от злоупотреблений. Если каждый LLM-запрос стоит $0.01-0.02, без rate limiting один злонамеренный пользователь может создать счёт в тысячи долларов.

Два уровня rate limiting:

1. **Nginx/reverse proxy** — простой, не требует изменений в коде:
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;

server {
    location /api/v1/assess {
        limit_req zone=api burst=5 nodelay;
        proxy_pass http://app:8000;
        proxy_read_timeout 120s;
    }
}
```

2. **In-app middleware** — гибкий, учитывает API-ключи и тарифы:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.post("/api/v1/assess")
@limiter.limit("10/minute")
async def assess_endpoint(request: Request):
    pass
```

**Blue-green deployment.** Две идентичные среды — blue (текущая) и green (новая). Деплой новой версии в green, прогоняем health checks и smoke tests. Если всё ок — переключаем load balancer на green. Если проблемы — мгновенный rollback на blue.

```
Load Balancer
    ├── Blue  (v1.2.0) ← текущий трафик
    └── Green (v1.3.0) ← новая версия, тестируется
```

Для LLM-приложений blue-green особенно важен: новая версия промпта может дать неожиданно плохие результаты, и нужна возможность быстрого отката.

---

## Справочник API / Config Reference

### Dockerfile Directives

| Директива | Описание | Пример |
|---|---|---|
| `FROM` | Базовый образ | `FROM python:3.12-slim AS builder` |
| `RUN` | Выполнить команду при сборке | `RUN pip install -r requirements.txt` |
| `COPY` | Копировать файлы в образ | `COPY app/ ./app/` |
| `ENV` | Переменная окружения | `ENV PYTHONUNBUFFERED=1` |
| `USER` | Переключить пользователя | `USER appuser` |
| `EXPOSE` | Документировать порт | `EXPOSE 8000` |
| `CMD` | Команда по умолчанию | `CMD ["uvicorn", "app.main:app"]` |
| `ENTRYPOINT` | Фиксированная точка входа | `ENTRYPOINT ["./entrypoint.sh"]` |
| `WORKDIR` | Рабочая директория | `WORKDIR /home/appuser/app` |
| `HEALTHCHECK` | Встроенная проверка | `HEALTHCHECK CMD curl -f http://localhost:8000/health` |

`CMD` vs `ENTRYPOINT`: `ENTRYPOINT` нельзя переопределить при `docker run` (без `--entrypoint`). `CMD` — аргументы по умолчанию, легко переопределяются. Рекомендация: `ENTRYPOINT` для production (фиксированный процесс), `CMD` для dev (гибкость).

### docker-compose.yml Structure

```yaml
services:
  <service_name>:
    image: <image>           # Готовый образ
    build:                   # Или сборка из Dockerfile
      context: .
      dockerfile: Dockerfile
    ports:
      - "host:container"
    volumes:
      - named_vol:/path
      - ./local:/path        # Bind mount (для dev)
    environment:
      KEY: value
    env_file:
      - .env
    depends_on:
      other_service:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s      # Не проверять первые 30s (время на старт)
    restart: unless-stopped   # Перезапуск при краше
    deploy:
      replicas: 3            # Количество реплик
      resources:
        limits:
          memory: 1G
    profiles:
      - dev                  # Только в профиле dev
    networks:
      - backend

volumes:
  named_vol:

networks:
  backend:
    driver: bridge
```

### Gunicorn Config

| Параметр | Описание | Рекомендация для LLM |
|---|---|---|
| `--workers` | Количество worker-процессов | 2-4 (I/O-bound) |
| `--worker-class` | Тип worker | `uvicorn.workers.UvicornWorker` |
| `--timeout` | Таймаут worker (сек) | 120+ (LLM запросы долгие) |
| `--graceful-timeout` | Время на graceful shutdown | 60 (дать закончить LLM) |
| `--keep-alive` | Keep-alive (сек) | 5 |
| `--bind` | Адрес и порт | `0.0.0.0:8000` |
| `--log-level` | Уровень логирования | `info` (prod), `debug` (dev) |
| `--access-logfile` | Access log файл | `-` (stdout для Docker) |
| `--max-requests` | Перезапуск после N запросов | 1000 (борьба с memory leaks) |
| `--max-requests-jitter` | Разброс для max-requests | 50 (не все workers одновременно) |

Gunicorn config file (`gunicorn.conf.py`):

```python
import multiprocessing

bind = "0.0.0.0:8000"
workers = min(multiprocessing.cpu_count(), 4)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
graceful_timeout = 60
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

### Uvicorn Config

| Параметр | Описание | Dev | Prod |
|---|---|---|---|
| `--host` | Bind address | `127.0.0.1` | `0.0.0.0` |
| `--port` | Port | `8000` | `8000` |
| `--workers` | Worker processes | 1 | через gunicorn |
| `--reload` | Auto-reload | Да | Нет |
| `--log-level` | Log level | `debug` | `info` |
| `--ssl-keyfile` | SSL key | — | путь к ключу |
| `--ssl-certfile` | SSL cert | — | путь к сертификату |
| `--limit-concurrency` | Max concurrent connections | — | 100 |

### LangServe `add_routes()`

```python
from langserve import add_routes

add_routes(
    app,                      # FastAPI app instance
    runnable,                 # Any LangChain Runnable (chain, LLM, retriever)
    path="/my-chain",         # URL prefix
    input_type=MyInput,       # Pydantic model для input (опционально)
    output_type=MyOutput,     # Pydantic model для output (опционально)
    config_keys=["tags"],     # Какие config ключи доступны клиенту
    enable_feedback_endpoint=True,   # Endpoint для фидбека
    enable_public_trace_link_endpoint=True,  # LangSmith trace links
    per_req_config_modifier=add_auth,  # Функция модификации config per request
)
```

### Pydantic Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",              # Файл с переменными
        env_file_encoding="utf-8",    # Кодировка
        env_prefix="APP_",            # Префикс переменных
        env_nested_delimiter="__",    # Разделитель для вложенных моделей
        case_sensitive=False,         # Регистронезависимые имена
        extra="ignore",              # Игнорировать лишние переменные
    )

    api_key: str                     # Обязательное (нет default)
    debug: bool = False              # Опциональное с default
    port: int = 8000

    class DatabaseConfig(BaseSettings):
        host: str = "localhost"
        port: int = 5432
        name: str = "mydb"

    database: DatabaseConfig = DatabaseConfig()
```

С `env_nested_delimiter="__"`: переменная `APP_DATABASE__HOST=db.example.com` установит `settings.database.host`.

---

## Практика

### Шаг 1. Health Checks

Автономный модуль проверки компонентов с кэшированием результатов. Его можно подключить к любому ASGI-фреймворку (FastAPI, Starlette, LangServe).

Зачем три разных эндпоинта: Docker и Kubernetes используют разные типы проб. Liveness — перезапустить зависший контейнер. Readiness — убрать из балансировки, пока зависимость недоступна. Startup — дождаться первичной инициализации.

```python
import os
from datetime import datetime, timedelta

import httpx

_health_cache: dict[str, tuple[dict, datetime]] = {}
CACHE_TTL = timedelta(seconds=30)


def _get_cached(component: str) -> dict | None:
    if component not in _health_cache:
        return None
    result, checked_at = _health_cache[component]
    if datetime.now() - checked_at > CACHE_TTL:
        return None
    return result


def _set_cache(component: str, result: dict):
    _health_cache[component] = (result, datetime.now())


async def liveness() -> dict:
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


async def readiness() -> dict:
    components = {}

    components["llm_api"] = await _check_llm_api()
    components["chromadb"] = await _check_chromadb()
    components["redis"] = await _check_redis()

    all_healthy = all(c["status"] == "healthy" for c in components.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }


async def startup_check() -> dict:
    return {"status": "started", "timestamp": datetime.now().isoformat()}


async def _check_llm_api() -> dict:
    cached = _get_cached("llm_api")
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.get(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            )
        result = {"status": "healthy", "detail": "API reachable"}
    except httpx.RequestError as e:
        result = {"status": "unhealthy", "detail": str(e)}

    _set_cache("llm_api", result)
    return result


async def _check_chromadb() -> dict:
    cached = _get_cached("chromadb")
    if cached:
        return cached

    host = os.getenv("CHROMA_HOST", "localhost")
    port = os.getenv("CHROMA_PORT", "8100")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://{host}:{port}/api/v1/heartbeat")
            response.raise_for_status()
        result = {"status": "healthy", "detail": "ChromaDB reachable"}
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        result = {"status": "unhealthy", "detail": str(e)}

    _set_cache("chromadb", result)
    return result


async def _check_redis() -> dict:
    cached = _get_cached("redis")
    if cached:
        return cached

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=True)
        await r.ping()
        await r.aclose()
        result = {"status": "healthy", "detail": "Redis reachable"}
    except Exception as e:
        result = {"status": "unhealthy", "detail": str(e)}

    _set_cache("redis", result)
    return result
```

Пример подключения к ASGI-приложению:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health/live")
async def health_live():
    return await liveness()


@app.get("/health/ready")
async def health_ready():
    return await readiness()


@app.get("/health/startup")
async def health_startup():
    return await startup_check()
```

### Шаг 2. Dockerfile

Production-ready Dockerfile для Python LLM-приложения. Multi-stage build, non-root user, оптимизированное кэширование слоёв.

```dockerfile
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser

WORKDIR /home/appuser/app

COPY src/ ./src/
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["gunicorn", "src.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "60", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

`.dockerignore`:

```
.git
.gitignore
.env
.env.*
__pycache__
*.pyc
.venv
venv
data/
docs/
tests/
*.md
.mypy_cache
.ruff_cache
.pytest_cache
.coverage
htmlcov/
node_modules
docker-compose*.yml
```

Сборка и проверка размера:

```bash
docker build -t llm-app:latest .
docker images llm-app
```

Ожидаемый размер: 200-350MB в зависимости от зависимостей.

### Шаг 3. docker-compose.yml

Полный стек для LLM-приложения: основной сервис, vector store, кэш, observability.

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      CHROMA_HOST: chromadb
      CHROMA_PORT: "8100"
      REDIS_URL: redis://redis:6379/0
      WORKERS: "2"
      LOG_LEVEL: info
    depends_on:
      chromadb:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G

  langserve:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uvicorn", "langserve_app:app", "--host", "0.0.0.0", "--port", "8001"]
    ports:
      - "8001:8001"
    env_file:
      - .env
    networks:
      - backend
    profiles:
      - dev

  chromadb:
    image: chromadb/chroma:0.5.23
    ports:
      - "8100:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      ANONYMIZED_TELEMETRY: "False"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
    networks:
      - backend
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - backend
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse_secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - backend
    restart: unless-stopped
    profiles:
      - dev
      - monitoring

  langfuse:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse_secret@postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: dev-secret-change-in-prod
      SALT: dev-salt-change-in-prod
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - backend
    restart: unless-stopped
    profiles:
      - dev
      - monitoring

volumes:
  chroma_data:
  redis_data:
  postgres_data:

networks:
  backend:
    driver: bridge
```

Запуск:

```bash
docker compose up -d

docker compose --profile dev up -d

docker compose ps

docker compose logs -f app
```

### Шаг 4. LangServe

Автономное LangServe-приложение, которое экспонирует LLM-chain как REST API с playground. Не зависит от конкретного проекта — достаточно установить зависимости и задать переменные окружения.

```python
import os

from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLM LangServe Demo",
    description="LangServe playground for LLM chains",
    version="0.1.0",
)


class SummarizeInput(BaseModel):
    text: str = Field(description="Text to summarize")
    max_length: str = Field(
        default="2-3 sentences",
        description="Desired summary length",
    )


class SummarizeOutput(BaseModel):
    summary: str = Field(description="Summarized text")
    key_points: list[str] = Field(description="Key points extracted")


llm = ChatAnthropic(
    model=os.getenv("MODEL_NAME", "claude-sonnet-4-20250514"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", "0.3")),
    max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
)

summarize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert summarizer. "
        "Summarize the given text concisely, extracting key points.",
    ),
    (
        "human",
        "Desired length: {max_length}\n\nText:\n{text}",
    ),
])

structured_chain = summarize_prompt | llm.with_structured_output(SummarizeOutput)

add_routes(
    app,
    structured_chain,
    path="/summarize",
    input_type=SummarizeInput,
    output_type=SummarizeOutput,
)

plain_chain = summarize_prompt | llm

add_routes(
    app,
    plain_chain,
    path="/summarize-text",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

Запуск:

```bash
uvicorn langserve_app:app --host 0.0.0.0 --port 8001 --reload
```

Открыть playground в браузере: `http://localhost:8001/summarize/playground`

Тест через curl:

```bash
curl -X POST http://localhost:8001/summarize/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Docker is a platform for developing, shipping, and running applications in containers. Containers package code and dependencies together, ensuring consistent behavior across environments.",
      "max_length": "1 sentence"
    }
  }'
```

Получить JSON Schema входных данных:

```bash
curl http://localhost:8001/summarize/input_schema
```

### Шаг 5. Environment Configuration и Entrypoint

Конфигурация через Pydantic Settings — единый способ управления переменными окружения для любого Python LLM-приложения:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    anthropic_api_key: str
    openai_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    chroma_host: str = "localhost"
    chroma_port: int = 8100
    redis_url: str = "redis://localhost:6379/0"

    log_level: str = "info"
    environment: str = "development"
    workers: int = 2
    timeout: int = 120


settings = Settings()
print(f"Model: {settings.model_name}, Environment: {settings.environment}")
```

Production entrypoint-скрипт. Конфигурируется через переменные окружения, поддерживает graceful shutdown:

```bash
#!/bin/bash
set -e

echo "=== LLM Application Starting ==="
echo "Environment: ${ENVIRONMENT:-development}"
echo "Workers: ${WORKERS:-2}"
echo "Timeout: ${TIMEOUT:-120}s"
echo "Bind: ${BIND:-0.0.0.0:8000}"

if [ "${ENVIRONMENT}" = "development" ]; then
    echo "Starting in development mode with auto-reload..."
    exec uvicorn src.main:app \
        --host 0.0.0.0 \
        --port "${PORT:-8000}" \
        --reload \
        --log-level "${LOG_LEVEL:-debug}"
fi

echo "Starting in production mode with gunicorn..."
exec gunicorn src.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "${WORKERS:-2}" \
    --bind "${BIND:-0.0.0.0:8000}" \
    --timeout "${TIMEOUT:-120}" \
    --graceful-timeout "${GRACEFUL_TIMEOUT:-60}" \
    --keep-alive 5 \
    --max-requests "${MAX_REQUESTS:-1000}" \
    --max-requests-jitter 50 \
    --log-level "${LOG_LEVEL:-info}" \
    --access-logfile - \
    --error-logfile -
```

`exec` заменяет shell-процесс на gunicorn — контейнер получает PID 1 для gunicorn, что важно для корректной обработки SIGTERM при остановке контейнера.

Обновление Dockerfile для использования entrypoint:

```dockerfile
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
```

### Шаг 6. Проверка

После создания всех файлов — проверяем работоспособность.

**6.1. Сборка и запуск:**

```bash
docker compose build

docker compose up -d

docker compose ps
```

Ожидаемый результат — все сервисы в статусе `healthy`.

**6.2. Health checks:**

```bash
curl http://localhost:8000/health/live
```

```json
{"status": "alive", "timestamp": "2026-03-26T10:00:00"}
```

```bash
curl http://localhost:8000/health/ready
```

```json
{
  "status": "ready",
  "components": {
    "llm_api": {"status": "healthy", "detail": "API reachable"},
    "chromadb": {"status": "healthy", "detail": "ChromaDB reachable"},
    "redis": {"status": "healthy", "detail": "Redis reachable"}
  },
  "timestamp": "2026-03-26T10:00:01"
}
```

**6.3. LangServe playground (если запущен с профилем dev):**

```bash
docker compose --profile dev up -d
```

Открыть в браузере: `http://localhost:8001/summarize/playground`

**6.4. Логи и дебаг:**

```bash
docker compose logs -f app

docker compose logs --tail=50 chromadb
```

**6.5. Остановка и очистка:**

```bash
docker compose down

docker compose down -v
```

Флаг `-v` удаляет volumes — данные ChromaDB, Redis, PostgreSQL будут потеряны. Используйте только при полной очистке.

### Связь с теорией

| Шаг | Раздел теории | Что применили |
|---|---|---|
| Шаг 1. Health Checks | §5 Health Checks | Liveness, readiness, startup probes; кэширование проверок |
| Шаг 2. Dockerfile | §2 Docker | Multi-stage build, non-root user, layer caching |
| Шаг 3. Docker Compose | §3 Docker Compose | Полный стек, networking, volumes, health checks, profiles |
| Шаг 4. LangServe | §6 LangServe | Chain как REST API с playground |
| Шаг 5. Entrypoint | §4 ASGI Server, §7 Secrets | Gunicorn + uvicorn workers, Pydantic Settings |
| Шаг 6. Проверка | §1 Особенности LLM | Проверка таймаутов, health checks |

---

## Чеклист самопроверки

1. **Dockerfile собирается без ошибок:** `docker build -t llm-app .` завершается успешно, образ меньше 400MB.

2. **Multi-stage build работает:** в финальном образе нет `build-essential`, `gcc`, pip cache. Проверить: `docker run --rm llm-app pip list` показывает только runtime-зависимости.

3. **Non-root user:** `docker run --rm llm-app whoami` возвращает `appuser`, а не `root`.

4. **docker-compose up поднимает все сервисы:** после `docker compose up -d` команда `docker compose ps` показывает все контейнеры в статусе `healthy` или `running`.

5. **Health checks работают:** `curl http://localhost:8000/health/live` возвращает 200. `curl http://localhost:8000/health/ready` возвращает JSON с компонентами.

6. **Переменные окружения через .env:** в Dockerfile и коде нет захардкоженных API-ключей. `.env` в `.dockerignore`.

7. **Gunicorn с правильным timeout:** `docker compose logs app | grep timeout` или `docker exec <container> ps aux` — видно `--timeout 120`.

8. **LangServe playground доступен:** `http://localhost:8001/summarize/playground` открывается в браузере и показывает интерфейс тестирования.

9. **LLM-вызов работает через Docker:** POST на LangServe endpoint через curl возвращает результат (нужен валидный API-ключ в `.env`).

10. **Graceful shutdown:** `docker compose stop` не обрывает текущие запросы — gunicorn ждёт `graceful-timeout` секунд.

---

## Частые ошибки

### 1. Timeout слишком короткий для LLM запросов

```yaml
services:
  app:
    # ...
```

```nginx
proxy_read_timeout 60s;
```

LLM-запрос с большим контекстом легко занимает 30-60 секунд. Nginx по умолчанию обрывает соединение через 60 секунд. Gunicorn по умолчанию убивает worker через 30 секунд. Результат: 504 Gateway Timeout или worker постоянно перезапускается.

Решение:

```bash
gunicorn src.main:app --timeout 120 --graceful-timeout 60
```

```nginx
proxy_read_timeout 120s;
proxy_connect_timeout 10s;
proxy_send_timeout 120s;
```

Правило: timeout reverse proxy ≥ timeout gunicorn ≥ максимальное время LLM-запроса + буфер.

### 2. .env в Docker image (утечка секретов)

```dockerfile
COPY . .
```

Без `.dockerignore` эта команда скопирует `.env` с API-ключами в Docker image. Любой, кто получит доступ к image (Docker registry, CI cache), получит ваши ключи.

Решение:

```
# .dockerignore
.env
.env.*
```

И копировать только нужные файлы:

```dockerfile
COPY src/ ./src/
COPY requirements.txt .
```

### 3. Root user в контейнере

```dockerfile
FROM python:3.12-slim
COPY . .
CMD ["uvicorn", "src.main:app"]
```

Контейнер работает под root. RCE-уязвимость = root-доступ.

Решение:

```dockerfile
RUN useradd --create-home appuser
USER appuser
```

### 4. Нет health checks → сервис "запущен" но не готов

```yaml
services:
  app:
    depends_on:
      - chromadb
```

`depends_on` без `condition: service_healthy` гарантирует только порядок старта контейнеров, а не их готовность. ChromaDB запускается за 5-10 секунд, но `depends_on` без условия не ждёт — приложение стартует и падает при первом запросе к ChromaDB.

Решение:

```yaml
services:
  app:
    depends_on:
      chromadb:
        condition: service_healthy
  chromadb:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 5. Один worker для LLM API → все запросы sequential

```bash
uvicorn src.main:app --workers 1
```

Один worker с одним event loop. Хотя `asyncio` позволяет конкурентные I/O-операции, один worker имеет ограничения по throughput. Если worker занят CPU-операцией (парсинг большого PDF, сериализация), все остальные запросы ждут.

Решение:

```bash
gunicorn src.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2
```

Но не переборщите: каждый worker — отдельный Python-процесс, ~100-200MB RAM. С embedding-моделью — ещё больше.

### 6. requirements.txt без pin'ов → невоспроизводимые сборки

```
fastapi
langchain
pydantic
```

Сегодня собирается с `langchain==0.3.x`, завтра — с `0.4.x`, который сломал API. Docker cache усугубляет: на одной машине кэшированная версия, на другой — новая.

Решение:

```
fastapi==0.115.6
langchain==0.3.14
langchain-anthropic==0.3.12
pydantic==2.10.5
pydantic-settings==2.7.1
uvicorn==0.34.0
gunicorn==23.0.0
langserve[all]==0.3.1
httpx==0.28.1
redis==5.2.1
sse-starlette==2.2.1
```

Генерация: `pip freeze > requirements.txt` или `uv pip freeze`.

---

## Что читать дальше

- [Docker Best Practices for Python](https://docs.docker.com/guides/python/) — официальный гайд Docker
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/) — документация FastAPI по деплою
- [Gunicorn Settings](https://docs.gunicorn.org/en/stable/settings.html) — все параметры gunicorn
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/) — production deployment uvicorn
- [LangServe](https://python.langchain.com/docs/langserve/) — документация LangServe
- [12 Factor App](https://12factor.net/) — методология для cloud-native приложений
- [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/) — probes в Kubernetes

**Следующая тема:** масштабирование и advanced production patterns — CI/CD, monitoring dashboards, auto-scaling по метрикам LLM.

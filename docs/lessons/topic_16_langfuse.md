# Тема 16: Langfuse Deep Dive — платформа для LLM observability

> **Пререквизиты:** [Тема 8: Observability](topic_08_observability.md)
> **Что добавляем в проект:** `app/api/v1/langfuse.py`, расширение `app/services/langfuse_service.py`, `app/schemas/langfuse.py`
> **Зависимости:** `langfuse`, `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. Langfuse как платформа — обзор возможностей

В теме 8 мы познакомились с Langfuse как инструментом трейсинга: отправляли traces, записывали generations, считали токены. Но трейсинг — лишь один из пяти столпов Langfuse. Платформа покрывает полный lifecycle LLM-приложения:

| Столп | Что решает | Без Langfuse |
|-------|-----------|--------------|
| **Tracing** | Логирование каждого LLM-вызова: промпт, ответ, токены, латентность | `print()` или самописные логи |
| **Prompt Management** | Версионирование промптов, A/B тесты, горячая замена без деплоя | Промпты захардкожены в коде |
| **Evaluation** | Scores для каждого trace — автоматические и ручные | Субъективная оценка «вроде работает» |
| **Datasets** | Golden datasets для регрессионного тестирования | Ручные curl-запросы |
| **Cost Analytics** | Расходы по моделям, пользователям, фичам | Счёт от провайдера в конце месяца |

Каждый столп решает конкретную проблему в production LLM-приложении. Трейсинг показывает, что происходит. Prompt management позволяет менять поведение без деплоя. Evaluation отвечает на вопрос «стало лучше или хуже?». Datasets дают воспроизводимость. Cost analytics — контроль бюджета.

**Архитектура Langfuse:**

```
┌─────────────────────────────────────────────┐
│                  Langfuse UI                │
│  (Dashboard, Prompt Editor, Annotations)    │
├─────────────────────────────────────────────┤
│               Langfuse Server               │
│  (API, ingestion, processing, analytics)    │
├──────────┬──────────────┬───────────────────┤
│ Postgres │  (ClickHouse │   S3 / Blob       │
│  (core)  │   optional)  │  (media, export)  │
└──────────┴──────────────┴───────────────────┘
```

**Self-hosted vs Langfuse Cloud:**

| Аспект | Langfuse Cloud | Self-hosted |
|--------|---------------|-------------|
| Развёртывание | Регистрация на cloud.langfuse.com | Docker Compose |
| Стоимость | Бесплатный tier (50K observations/мес) | Инфраструктурные расходы |
| Data residency | EU / US (выбор при регистрации) | Полный контроль |
| Обновления | Автоматические | `docker-compose pull` |
| SSO, RBAC | Enterprise plan | Настраиваемо |
| Для обучения | Идеально | Избыточно |

Для нашего bootcamp-проекта используем Langfuse Cloud — бесплатный tier более чем достаточен.

**Модель данных** строится иерархически:

```
Project
└── Trace (полный запрос пользователя)
    ├── Span (логический шаг)
    │   └── Generation (LLM-вызов)
    ├── Generation (прямой LLM-вызов)
    ├── Event (точечное событие)
    └── Score (оценка качества)
```

Интеграции покрывают основные фреймворки: LangChain (CallbackHandler), LlamaIndex (инструментация), OpenAI SDK (drop-in wrapper), Vercel AI SDK, и low-level Python/JS SDK для кастомных интеграций. В нашем проекте мы уже используем `CallbackHandler` из темы 8 — теперь расширим его до полноценного workflow.

### 2. Prompt Management — версионирование промптов

Типичная проблема LLM-проектов: промпты живут в коде как строковые константы. Изменение промпта требует коммита, code review, деплоя. A/B тестирование невозможно — нельзя переключить 10% трафика на новый промпт без feature flags.

Langfuse Prompt Management решает эти проблемы:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Langfuse UI │────▶│  Prompt v1   │────▶│  production  │
│  (редактор)  │     │  Prompt v2   │     │  staging     │
│              │     │  Prompt v3   │     │  latest      │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Your App    │
                     │  get_prompt  │
                     │  (runtime)   │
                     └──────────────┘
```

**Workflow:**

1. Создаёте промпт в Langfuse UI (или через API)
2. Редактируете — каждое изменение создаёт новую версию
3. Тестируете в Playground (встроенный в UI)
4. Назначаете метку `production` проверенной версии
5. Код загружает промпт по имени и метке в runtime
6. Промпт автоматически привязывается к generations в traces

**Типы промптов:**

| Тип | Формат | Аналог в LangChain |
|-----|--------|-------------------|
| `text` | Одна строка с переменными `{{var}}` | `PromptTemplate` |
| `chat` | Массив сообщений `[{role, content}]` | `ChatPromptTemplate` |

Переменные обозначаются двойными фигурными скобками: `{{student_work}}`, `{{rubric}}`. При вызове `prompt.compile(student_work="...", rubric="...")` Langfuse подставляет значения.

**Кэширование.** По умолчанию SDK кэширует промпт и перезагружает его раз в 60 секунд. Параметр `cache_ttl_seconds` управляет интервалом. Для production рекомендуется 300 секунд (5 минут) — достаточно быстро для hotfix, но не создаёт нагрузку на API.

**Связь с traces.** Когда вы загружаете промпт через `langfuse.get_prompt()` и используете его в chain, Langfuse автоматически связывает generation с версией промпта. В UI видно: «этот trace использовал assessment-prompt v3». Это критично для анализа — вы можете сравнить метрики до и после смены промпта.

### 3. Datasets и Experiments

Dataset в Langfuse — это набор тестовых примеров (items), каждый из которых содержит input, expected output и metadata. Datasets решают проблему воспроизводимости: вместо «запустил curl 5 раз, вроде работает» вы прогоняете фиксированный набор из 20-50 примеров и сравниваете результаты количественно.

**Структура dataset:**

```
Dataset: "assessment-golden"
├── Item 1: {input: essay_1, expected_output: {score: 85, ...}}
├── Item 2: {input: essay_2, expected_output: {score: 62, ...}}
├── Item 3: {input: code_1, expected_output: {score: 91, ...}}
└── ...
```

**Dataset Run (Experiment):** прогон всего dataset через chain. Каждый run — это одна «версия» эксперимента. Langfuse сохраняет результаты и позволяет сравнивать runs:

```
Dataset: "assessment-golden" (20 items)
├── Run "prompt-v1-sonnet" (2024-01-15)
│   ├── Item 1 → score 82 (expected 85) → accuracy_score: 0.96
│   └── ... (avg accuracy: 0.89)
├── Run "prompt-v2-sonnet" (2024-01-16)
│   ├── Item 1 → score 84 (expected 85) → accuracy_score: 0.99
│   └── ... (avg accuracy: 0.94)  ← лучше!
└── Run "prompt-v2-haiku" (2024-01-16)
    └── ... (avg accuracy: 0.81)  ← дешевле, но хуже
```

**Workflow эксперимента:**

1. Создаёте dataset с golden examples (вручную или из production traces)
2. Запускаете chain на каждом item — создаётся trace
3. Связываете trace с dataset item через `observation.link()`
4. Считаете scores (автоматически или вручную)
5. Сравниваете runs в UI — таблица с метриками по каждому item

Этот workflow — основа для data-driven development промптов. Вместо «мне кажется, новый промпт лучше» вы видите: «prompt v2 увеличил accuracy с 0.89 до 0.94 на golden dataset из 20 примеров».

### 4. Scores — оценка качества

Score — числовая или категориальная оценка, привязанная к trace или observation. Scores — это мост между «система работает» и «система работает хорошо».

**Типы scores:**

| Тип | Значение | Пример |
|-----|---------|--------|
| `NUMERIC` | Число в заданном диапазоне | accuracy: 0.85 (0..1) |
| `CATEGORICAL` | Строка из набора категорий | quality: "good" \| "bad" \| "neutral" |
| `BOOLEAN` | true / false | is_hallucination: false |

**Источники scores:**

1. **Automated — LLM-as-judge.** Отдельный LLM-вызов оценивает результат основного. Например: основная chain оценила эссе на 85 баллов. LLM-judge проверяет: «Корректна ли оценка 85 для этого эссе по данной рубрике?» и выставляет score 0.9.

2. **Automated — Heuristic.** Программная проверка без LLM: длина ответа > 100 символов, JSON валиден, score в диапазоне 0-100, все обязательные поля заполнены. Дешево и быстро, но ограничено по глубине.

3. **Manual — Annotation.** Человек просматривает trace в UI и выставляет оценку. Самый точный, но самый дорогой метод. Используется для создания golden datasets и калибровки automated scores.

4. **User feedback.** Пользователь нажимает «палец вверх» / «палец вниз» — приложение отправляет score в Langfuse. Масштабируется лучше всего, но шумный сигнал.

**Evaluation pipeline:**

```
Trace (assessment) → Run evaluators → Post scores → Dashboard
     │                    │                │
     │              ┌─────┴─────┐    ┌─────┴─────┐
     │              │ LLM-judge │    │ Heuristic  │
     │              │ "Is score │    │ "score     │
     │              │  accurate?"│    │  in 0-100?"│
     │              └───────────┘    └───────────┘
     │                    │                │
     └────────────────────┴────────────────┘
                          │
                    Score: accuracy=0.9
                    Score: format_valid=true
```

Custom evaluators — это обычные Python-функции, которые принимают trace data и возвращают scores. Вы можете запускать их синхронно (после каждого вызова) или асинхронно (batch job по расписанию).

### 5. Annotation Queues

Когда у вас 1000 traces в день, невозможно проверить каждый вручную. Annotation queues организуют процесс ручной проверки.

**Workflow:**

1. Настраиваете фильтр в UI: «последние 24 часа, модель sonnet, без scores»
2. Создаёте queue — Langfuse выбирает подходящие traces
3. Reviewer открывает queue, берёт первый item
4. Видит: input (эссе студента), output (оценка от LLM), metadata
5. Выставляет scores: accuracy, completeness, tone
6. Переходит к следующему item

**Зачем annotation queues:**

- **Quality assurance.** Выборочная проверка 5% traces — statistical sampling достаточен для оценки качества системы.
- **Dataset creation.** Проверенные traces с scores становятся golden dataset items.
- **Evaluator calibration.** Ручные scores сравниваются с automated scores — если расхождение > 10%, evaluator нуждается в настройке.
- **Compliance.** В регулируемых отраслях (education, healthcare) требуется human-in-the-loop review.

Annotation queues настраиваются через UI Langfuse. В API нет эндпоинтов для создания queues — это сознательное решение, чтобы workflow review оставался в UI, а не автоматизировался.

### 6. Cost Analytics и Dashboards

Langfuse автоматически рассчитывает стоимость каждого LLM-вызова по модели и количеству токенов. Для поддерживаемых моделей (Claude, GPT-4, Gemini, Llama) цены встроены. Для кастомных моделей цены добавляются через UI.

**Dashboard views:**

| Метрика | Описание | Использование |
|---------|----------|--------------|
| Cost by model | Расходы в разрезе моделей | Оптимизация: перевод простых задач на дешёвую модель |
| Cost by user | Расходы по user_id | Выявление аномалий, rate limiting |
| Cost by feature | Расходы по tags/metadata | ROI каждой фичи |
| Latency p50/p95/p99 | Перцентили латентности | SLA мониторинг |
| Token usage | Input vs output tokens, тренды | Оптимизация промптов |
| Trace count | Количество вызовов по времени | Capacity planning |

**Latency tracking** особенно важен для LLM-приложений. Типичные значения:

```
Endpoint: /assess
├── p50: 3.2s   ← половина запросов укладывается
├── p95: 8.1s   ← 95% запросов укладывается
└── p99: 15.4s  ← только 1% дольше (retry, long essays)
```

Если p95 растёт, но p50 стабилен — проблема в edge cases (длинные входы, retry loops). Если растёт p50 — системная деградация.

**Alerts:** Langfuse позволяет настроить уведомления при превышении бюджета или аномальной латентности. Для production-приложений рекомендуется alert при дневных расходах > 2x от среднего.

### 7. Интеграция с LangChain — продвинутые паттерны

В теме 8 мы использовали `CallbackHandler` для базового трейсинга. Теперь разберём продвинутые паттерны.

**Metadata и tags для фильтрации:**

Metadata — произвольный dict, привязанный к trace. Tags — список строк для быстрой фильтрации. Используйте metadata для структурированных данных (`user_id`, `feature`, `prompt_version`), tags — для категорий (`production`, `assessment`, `experiment`).

**Session tracking:**

`session_id` группирует traces одного пользователя в одну сессию. Все traces с одинаковым `session_id` отображаются вместе в UI. Для нашего assessment API — если студент отправляет несколько работ подряд, каждый trace получает один `session_id`.

**Nested traces:**

Сложные chains создают вложенные observations автоматически. LangChain CallbackHandler отслеживает parent-child relationships:

```
Trace "assess-essay"
├── Span "prompt_formatting"
├── Generation "claude-sonnet" (input: 1250 tokens, output: 800 tokens)
└── Span "output_parsing"
```

**Flushing:**

Langfuse SDK отправляет данные асинхронно в фоне. При shutdown приложения часть данных может потеряться. Вызов `langfuse.flush()` гарантирует отправку всех буферизованных данных. В FastAPI это делается в lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()
```

**Загрузка промптов + LangChain:**

Промпт из Langfuse конвертируется в `ChatPromptTemplate` LangChain. Это позволяет использовать managed промпты в стандартных LCEL chains без изменения архитектуры.

### 8. Self-hosting Langfuse

Self-hosted Langfuse разворачивается через Docker Compose. Минимальная конфигурация требует два контейнера: Langfuse server и PostgreSQL.

**Минимальный Docker Compose:**

```yaml
version: "3.8"
services:
  langfuse:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@db:5432/langfuse
      NEXTAUTH_SECRET: mysecret
      SALT: mysalt
      NEXTAUTH_URL: http://localhost:3000
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

**Ключевые environment variables:**

| Переменная | Описание | Обязательная |
|-----------|----------|-------------|
| `DATABASE_URL` | Connection string для PostgreSQL | Да |
| `NEXTAUTH_SECRET` | Secret для сессий (NextAuth.js) | Да |
| `SALT` | Salt для хеширования API-ключей | Да |
| `NEXTAUTH_URL` | Public URL Langfuse | Да |
| `LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES` | Включение экспериментальных фич | Нет |

**Production setup:**

- Reverse proxy (Nginx/Caddy) для SSL termination
- PostgreSQL с бэкапами (pg_dump по расписанию)
- Optional: ClickHouse для аналитики на больших объёмах данных
- Обновления: `docker-compose pull && docker-compose up -d`

**Когда self-host:**

- Data privacy: LLM-данные (промпты, ответы) не покидают вашу инфраструктуру
- Compliance: требования регулятора к data residency
- Custom domain: Langfuse на вашем домене
- Для обучения и небольших проектов — Langfuse Cloud проще и бесплатен

---

## Справочник API

### 1. Langfuse — клиент

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `public_key` | `str` | `LANGFUSE_PUBLIC_KEY` env | Публичный ключ проекта |
| `secret_key` | `str` | `LANGFUSE_SECRET_KEY` env | Секретный ключ проекта |
| `host` | `str` | `"https://cloud.langfuse.com"` | URL Langfuse-инстанса |
| `release` | `str \| None` | `None` | Версия приложения (git sha, semver) |
| `debug` | `bool` | `False` | Логирование SDK в stdout |
| `threads` | `int` | `1` | Количество потоков для отправки данных |
| `flush_at` | `int` | `15` | Размер батча перед отправкой |
| `flush_interval` | `float` | `0.5` | Интервал отправки в секундах |
| `enabled` | `bool` | `True` | Глобальный выключатель |

Если `public_key` и `secret_key` не переданы явно, SDK читает их из переменных окружения `LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY`. Это рекомендуемый подход — ключи не попадают в код.

### 2. langfuse.get_prompt()

```python
prompt = langfuse.get_prompt(
    name="assessment-prompt",
    label="production",
    cache_ttl_seconds=300,
    type="chat",
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `name` | `str` | — (обязательный) | Имя промпта |
| `version` | `int \| None` | `None` | Конкретная версия (1, 2, 3...) |
| `label` | `str \| None` | `None` | Метка: `"production"`, `"staging"`, `"latest"` |
| `cache_ttl_seconds` | `int` | `60` | TTL кэша в секундах |
| `type` | `str` | `"text"` | `"text"` или `"chat"` |
| `fallback` | `list \| str \| None` | `None` | Fallback-промпт, если Langfuse недоступен |

Возвращает объект `TextPromptClient` или `ChatPromptClient` в зависимости от `type`.

**Методы промпта:**

| Метод | Описание |
|-------|----------|
| `prompt.compile(var1="...", var2="...")` | Подстановка переменных, возвращает строку или список сообщений |
| `prompt.get_langchain_prompt()` | Конвертация в LangChain `ChatPromptTemplate` |
| `prompt.config` | Dict с конфигурацией (model, temperature и т.д.) |
| `prompt.version` | Номер версии |
| `prompt.labels` | Список меток |

Пример с `compile`:

```python
prompt = langfuse.get_prompt(name="assessment-prompt", type="chat")
messages = prompt.compile(
    student_work="Essay about climate change...",
    rubric="Clarity: 0-25, Argumentation: 0-25..."
)
```

### 3. langfuse.create_dataset() / dataset.create_item()

```python
dataset = langfuse.create_dataset(
    name="assessment-golden",
    description="Golden dataset for assessment quality testing",
    metadata={"version": "1.0", "created_by": "team"},
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `name` | `str` | — (обязательный) | Уникальное имя dataset |
| `description` | `str \| None` | `None` | Описание |
| `metadata` | `dict \| None` | `None` | Произвольные метаданные |

```python
langfuse.create_dataset_item(
    dataset_name="assessment-golden",
    input={"student_work": "Essay text...", "rubric_id": "essay_default"},
    expected_output={"overall_score": 85, "summary": "Good essay..."},
    metadata={"source": "manual", "difficulty": "medium"},
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `dataset_name` | `str` | — (обязательный) | Имя dataset |
| `input` | `dict \| str` | — (обязательный) | Входные данные |
| `expected_output` | `dict \| str \| None` | `None` | Ожидаемый результат |
| `metadata` | `dict \| None` | `None` | Произвольные метаданные |
| `id` | `str \| None` | `None` | Кастомный ID (для upsert) |
| `status` | `str` | `"ACTIVE"` | `"ACTIVE"` или `"ARCHIVED"` |

Для получения dataset и его items:

```python
dataset = langfuse.get_dataset("assessment-golden")
for item in dataset.items:
    print(item.input, item.expected_output)
```

### 4. langfuse.score()

```python
langfuse.score(
    trace_id="trace-123",
    name="accuracy",
    value=0.85,
    data_type="NUMERIC",
    comment="Score within 5 points of expected",
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `trace_id` | `str` | — (обязательный) | ID trace |
| `observation_id` | `str \| None` | `None` | ID конкретного observation (span/generation) |
| `name` | `str` | — (обязательный) | Имя score |
| `value` | `float \| str` | — (обязательный) | Значение |
| `data_type` | `str \| None` | `None` | `"NUMERIC"`, `"CATEGORICAL"`, `"BOOLEAN"` |
| `comment` | `str \| None` | `None` | Пояснение |
| `id` | `str \| None` | `None` | Кастомный ID (для upsert) |
| `config_id` | `str \| None` | `None` | ID score config |

Для numeric scores рекомендуемый диапазон 0..1. Для categorical — строго определённый набор значений. Boolean — `True`/`False` (передаётся как `1`/`0`).

### 5. langfuse.trace()

```python
trace = langfuse.trace(
    name="assessment",
    input={"student_work": "...", "rubric_id": "essay_default"},
    metadata={"user_id": "student-42", "feature": "essay_assessment"},
    tags=["production", "v2"],
    session_id="session-abc",
    user_id="student-42",
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `name` | `str \| None` | `None` | Имя trace |
| `input` | `dict \| str \| None` | `None` | Входные данные |
| `output` | `dict \| str \| None` | `None` | Выходные данные (обычно устанавливается позже) |
| `metadata` | `dict \| None` | `None` | Произвольные метаданные |
| `tags` | `list[str] \| None` | `None` | Теги для фильтрации |
| `session_id` | `str \| None` | `None` | ID сессии (группировка traces) |
| `user_id` | `str \| None` | `None` | ID пользователя |
| `id` | `str \| None` | `None` | Кастомный ID |
| `version` | `str \| None` | `None` | Версия приложения |
| `release` | `str \| None` | `None` | Релиз приложения |
| `public` | `bool \| None` | `None` | Публичный доступ по ссылке |

**Методы trace:**

```python
span = trace.span(name="preprocessing", input={"raw": "..."})
generation = trace.generation(
    name="llm-call",
    model="claude-sonnet-4-20250514",
    input=[{"role": "user", "content": "..."}],
    output={"content": "..."},
    usage={"input": 1250, "output": 800},
)
trace.update(output={"result": "..."})
```

### 6. CallbackHandler — продвинутая конфигурация

```python
from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
    trace_name="assessment",
    session_id="session-abc",
    user_id="student-42",
    metadata={"feature": "essay_assessment"},
    tags=["production"],
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `public_key` | `str \| None` | env var | Публичный ключ |
| `secret_key` | `str \| None` | env var | Секретный ключ |
| `host` | `str \| None` | env var | URL Langfuse |
| `trace_name` | `str \| None` | `None` | Имя trace |
| `session_id` | `str \| None` | `None` | ID сессии |
| `user_id` | `str \| None` | `None` | ID пользователя |
| `metadata` | `dict \| None` | `None` | Метаданные trace |
| `tags` | `list[str] \| None` | `None` | Теги trace |
| `trace_id` | `str \| None` | `None` | Привязка к существующему trace |
| `stateful_client` | `StatefulTraceClient \| None` | `None` | Привязка к существующему stateful trace |
| `enabled` | `bool` | `True` | Включение/выключение |

Использование с LangChain:

```python
result = await chain.ainvoke(
    {"student_work": "..."},
    config={"callbacks": [handler]},
)

trace_id = handler.get_trace_id()
trace_url = handler.get_trace_url()
```

### 7. dataset_item.link() / observation.link()

```python
handler = CallbackHandler()
result = await chain.ainvoke(input_data, config={"callbacks": [handler]})

handler.trace.update(output=result)

for item in dataset.items:
    handler = CallbackHandler(trace_name=f"experiment-{item.id}")
    output = await chain.ainvoke(item.input, config={"callbacks": [handler]})
    handler.trace.update(output=output)
    item.link(
        trace_or_observation=handler.trace,
        run_name="prompt-v2-experiment",
        run_metadata={"model": "claude-sonnet-4-20250514"},
    )
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `trace_or_observation` | `StatefulTraceClient \| StatefulSpanClient` | — (обязательный) | Trace или observation для связывания |
| `run_name` | `str` | — (обязательный) | Имя эксперимента (группирует все items одного прогона) |
| `run_metadata` | `dict \| None` | `None` | Метаданные прогона |
| `run_description` | `str \| None` | `None` | Описание прогона |

### 8. langfuse.flush()

```python
langfuse.flush()
```

Блокирующий вызов, отправляющий все буферизованные данные на сервер. Параметров нет. Вызывайте при shutdown приложения или после завершения эксперимента, чтобы гарантировать доставку данных.

Типичные места вызова:
- Lifespan FastAPI (shutdown)
- После завершения dataset run
- В finally-блоке скриптов

---

## Практика: роутер `/api/v1/langfuse`

### Шаг 1. Схемы — `app/schemas/langfuse.py`

Создаём Pydantic-модели для запросов и ответов четырёх эндпоинтов.

```python
from pydantic import BaseModel, Field


class PromptTestRequest(BaseModel):
    student_work: str = Field(min_length=10)
    rubric_id: str = "essay_default"
    prompt_name: str = "assessment-prompt"
    prompt_label: str = "production"


class PromptComparisonResult(BaseModel):
    source: str
    overall_score: int
    summary: str
    latency_ms: float
    token_usage: dict[str, int]


class PromptTestResponse(BaseModel):
    langfuse_prompt: PromptComparisonResult
    hardcoded_prompt: PromptComparisonResult
    trace_id: str
    trace_url: str


class DatasetItem(BaseModel):
    student_work: str = Field(min_length=10)
    expected_score: int = Field(ge=0, le=100)
    metadata: dict | None = None


class DatasetRunRequest(BaseModel):
    dataset_name: str = "assessment-golden"
    items: list[DatasetItem] = Field(min_length=1, max_length=50)
    prompt_versions: list[str] = Field(
        default=["v1", "v2"],
        min_length=1,
        max_length=5,
    )
    run_prefix: str = "experiment"


class ExperimentResult(BaseModel):
    run_name: str
    prompt_version: str
    avg_score_diff: float
    items_processed: int
    total_cost_usd: float
    avg_latency_ms: float


class ExperimentResponse(BaseModel):
    dataset_name: str
    runs: list[ExperimentResult]
    trace_ids: list[str]


class ScoreEntry(BaseModel):
    name: str
    value: float = Field(ge=0, le=1)
    comment: str | None = None


class ScoreRequest(BaseModel):
    student_work: str = Field(min_length=10)
    rubric_id: str = "essay_default"
    auto_score: bool = True
    custom_scores: list[ScoreEntry] | None = None


class ScoreResponse(BaseModel):
    trace_id: str
    assessment_result: dict
    scores: list[dict[str, str | float]]
    trace_url: str


class AnalyticsRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    tags: list[str] | None = None


class CostBreakdown(BaseModel):
    model: str
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    call_count: int


class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float


class AnalyticsResponse(BaseModel):
    total_traces: int
    total_cost_usd: float
    cost_by_model: list[CostBreakdown]
    latency: LatencyStats
    avg_scores: dict[str, float]
    period: str
```

### Шаг 2. Сервис — расширение `app/services/langfuse_service.py`

Расширяем сервис из темы 8 четырьмя функциями для новых эндпоинтов.

```python
import time
import statistics
from collections import defaultdict

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.config import Settings
from app.schemas.assessment import AssessmentResponse


def create_langfuse_client(settings: Settings) -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def create_callback_handler(
    settings: Settings,
    trace_name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> CallbackHandler:
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
    )


async def get_or_create_prompt(
    langfuse: Langfuse,
    name: str,
    template: list[dict[str, str]],
    label: str = "production",
    config: dict | None = None,
) -> object:
    try:
        prompt = langfuse.get_prompt(name=name, label=label, type="chat")
    except Exception:
        langfuse.create_prompt(
            name=name,
            prompt=template,
            labels=[label],
            type="chat",
            config=config or {},
        )
        prompt = langfuse.get_prompt(name=name, label=label, type="chat")
    return prompt


async def run_with_prompt(
    student_work: str,
    rubric_text: str,
    prompt_messages: list,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> tuple[AssessmentResponse, float, dict[str, int]]:
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain: Runnable = prompt | llm.with_structured_output(AssessmentResponse)

    start = time.perf_counter()
    result = await chain.ainvoke(
        {"student_work": student_work, "rubric": rubric_text},
        config={"callbacks": [handler]},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    usage = {"input_tokens": 0, "output_tokens": 0}
    if hasattr(handler, "trace") and handler.trace:
        try:
            trace_data = handler.trace
            usage = {
                "input_tokens": getattr(trace_data, "input_tokens", 0) or 0,
                "output_tokens": getattr(trace_data, "output_tokens", 0) or 0,
            }
        except Exception:
            pass

    return result, elapsed_ms, usage


async def create_dataset_with_items(
    langfuse: Langfuse,
    name: str,
    items: list[dict],
    description: str | None = None,
) -> str:
    langfuse.create_dataset(
        name=name,
        description=description or f"Dataset {name}",
    )

    for item in items:
        langfuse.create_dataset_item(
            dataset_name=name,
            input={"student_work": item["student_work"]},
            expected_output={"overall_score": item["expected_score"]},
            metadata=item.get("metadata"),
        )

    langfuse.flush()
    return name


async def run_experiment(
    langfuse: Langfuse,
    settings: Settings,
    dataset_name: str,
    llm: ChatAnthropic,
    prompt_template: list[dict[str, str]],
    run_name: str,
) -> dict:
    dataset = langfuse.get_dataset(dataset_name)
    results = []
    trace_ids = []

    prompt = ChatPromptTemplate.from_messages(prompt_template)
    chain = prompt | llm.with_structured_output(AssessmentResponse)

    for item in dataset.items:
        handler = create_callback_handler(
            settings,
            trace_name=f"{run_name}-{item.id}",
            tags=["experiment", run_name],
        )

        start = time.perf_counter()
        result = await chain.ainvoke(
            item.input,
            config={"callbacks": [handler]},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        handler.trace.update(output=result.model_dump())

        expected_score = item.expected_output.get("overall_score", 0)
        score_diff = abs(result.overall_score - expected_score)
        langfuse.score(
            trace_id=handler.get_trace_id(),
            name="score_accuracy",
            value=max(0, 1 - score_diff / 100),
        )

        item.link(
            trace_or_observation=handler.trace,
            run_name=run_name,
        )

        trace_ids.append(handler.get_trace_id())
        results.append({
            "score_diff": score_diff,
            "latency_ms": elapsed_ms,
            "predicted": result.overall_score,
            "expected": expected_score,
        })

    langfuse.flush()

    avg_diff = statistics.mean(r["score_diff"] for r in results)
    avg_latency = statistics.mean(r["latency_ms"] for r in results)

    return {
        "run_name": run_name,
        "avg_score_diff": round(avg_diff, 2),
        "items_processed": len(results),
        "avg_latency_ms": round(avg_latency, 1),
        "trace_ids": trace_ids,
    }


async def score_trace(
    langfuse: Langfuse,
    trace_id: str,
    scores: list[dict],
) -> list[dict]:
    posted = []
    for s in scores:
        langfuse.score(
            trace_id=trace_id,
            name=s["name"],
            value=s["value"],
            comment=s.get("comment"),
            data_type="NUMERIC",
        )
        posted.append({
            "name": s["name"],
            "value": s["value"],
            "trace_id": trace_id,
        })
    langfuse.flush()
    return posted


async def llm_judge_score(
    student_work: str,
    assessment_result: dict,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> list[dict]:
    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a quality evaluator. Given a student work and an AI assessment, "
            "rate the assessment quality on three dimensions.\n"
            "Return JSON with exactly these fields:\n"
            "- accuracy (0.0-1.0): how accurate is the score\n"
            "- completeness (0.0-1.0): does feedback cover all aspects\n"
            "- helpfulness (0.0-1.0): how actionable is the feedback",
        ),
        (
            "human",
            "Student work:\n{student_work}\n\n"
            "Assessment result:\n{assessment_result}\n\n"
            "Rate the assessment quality as JSON:",
        ),
    ])

    from pydantic import BaseModel, Field

    class JudgeScores(BaseModel):
        accuracy: float = Field(ge=0, le=1)
        completeness: float = Field(ge=0, le=1)
        helpfulness: float = Field(ge=0, le=1)

    chain = judge_prompt | llm.with_structured_output(JudgeScores)
    judge_result = await chain.ainvoke(
        {
            "student_work": student_work,
            "assessment_result": str(assessment_result),
        },
        config={"callbacks": [handler]},
    )

    return [
        {"name": "accuracy", "value": judge_result.accuracy},
        {"name": "completeness", "value": judge_result.completeness},
        {"name": "helpfulness", "value": judge_result.helpfulness},
    ]


def compute_analytics(
    langfuse: Langfuse,
    limit: int = 50,
    tags: list[str] | None = None,
) -> dict:
    traces = langfuse.fetch_traces(limit=limit)

    if not traces.data:
        return {
            "total_traces": 0,
            "total_cost_usd": 0,
            "cost_by_model": [],
            "latency": {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "avg_ms": 0},
            "avg_scores": {},
            "period": "no data",
        }

    model_costs = defaultdict(lambda: {
        "total_cost": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "count": 0,
    })
    latencies = []
    score_values = defaultdict(list)
    total_cost = 0

    for trace in traces.data:
        if tags and not any(t in (trace.tags or []) for t in tags):
            continue

        if trace.latency:
            latencies.append(trace.latency * 1000)

        observations = langfuse.fetch_observations(trace_id=trace.id)
        for obs in observations.data:
            if obs.type == "GENERATION":
                model_name = obs.model or "unknown"
                cost = obs.calculated_total_cost or 0
                total_cost += cost
                model_costs[model_name]["total_cost"] += cost
                model_costs[model_name]["count"] += 1
                if obs.usage:
                    model_costs[model_name]["input_tokens"] += (
                        obs.usage.input or 0
                    )
                    model_costs[model_name]["output_tokens"] += (
                        obs.usage.output or 0
                    )

        scores = langfuse.fetch_scores(trace_id=trace.id)
        for sc in scores.data if hasattr(scores, "data") else []:
            if sc.value is not None:
                score_values[sc.name].append(sc.value)

    sorted_latencies = sorted(latencies) if latencies else [0]
    n = len(sorted_latencies)

    cost_breakdown = [
        {
            "model": model,
            "total_cost_usd": round(data["total_cost"], 6),
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "call_count": data["count"],
        }
        for model, data in model_costs.items()
    ]

    avg_scores = {
        name: round(statistics.mean(vals), 3)
        for name, vals in score_values.items()
    }

    return {
        "total_traces": len(traces.data),
        "total_cost_usd": round(total_cost, 6),
        "cost_by_model": cost_breakdown,
        "latency": {
            "p50_ms": round(sorted_latencies[n // 2], 1),
            "p95_ms": round(sorted_latencies[int(n * 0.95)], 1),
            "p99_ms": round(sorted_latencies[int(n * 0.99)], 1),
            "avg_ms": round(statistics.mean(sorted_latencies), 1),
        },
        "avg_scores": avg_scores,
        "period": f"last {len(traces.data)} traces",
    }
```

### Шаг 3. Router — `app/api/v1/langfuse.py`

Четыре эндпоинта, каждый демонстрирует отдельный аспект Langfuse.

```python
from fastapi import APIRouter, HTTPException

from app.dependencies import LLMDep, RubricStoreDep
from app.config import get_settings
from app.schemas.langfuse import (
    PromptTestRequest,
    PromptTestResponse,
    PromptComparisonResult,
    DatasetRunRequest,
    ExperimentResponse,
    ExperimentResult,
    ScoreRequest,
    ScoreResponse,
    AnalyticsRequest,
    AnalyticsResponse,
    CostBreakdown,
    LatencyStats,
)
from app.services.langfuse_service import (
    create_langfuse_client,
    create_callback_handler,
    get_or_create_prompt,
    run_with_prompt,
    create_dataset_with_items,
    run_experiment,
    score_trace,
    llm_judge_score,
    compute_analytics,
)

router = APIRouter(prefix="/langfuse", tags=["langfuse"])

HARDCODED_ASSESSMENT_PROMPT = [
    (
        "system",
        "You are an expert educator. Assess the student work against the rubric.\n"
        "Rubric:\n{rubric}",
    ),
    (
        "human",
        "Student work:\n{student_work}",
    ),
]

LANGFUSE_PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": "You are an expert educator and assessment specialist.\n"
        "Evaluate the student work carefully against each criterion in the rubric.\n"
        "Be specific in your feedback and provide actionable suggestions.\n\n"
        "Rubric:\n{{rubric}}",
    },
    {
        "role": "user",
        "content": "Please assess the following student work:\n\n{{student_work}}",
    },
]


def _format_rubric(rubric_id: str, rubrics) -> str:
    if rubric_id not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rubric_id}' not found")
    rubric = rubrics[rubric_id]
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


@router.post("/prompt-test")
async def prompt_test(
    request: PromptTestRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> PromptTestResponse:
    settings = get_settings()
    langfuse = create_langfuse_client(settings)
    rubric_text = _format_rubric(request.rubric_id, rubrics)

    prompt = await get_or_create_prompt(
        langfuse,
        name=request.prompt_name,
        template=LANGFUSE_PROMPT_TEMPLATE,
        label=request.prompt_label,
    )

    langfuse_messages = []
    compiled = prompt.compile(student_work="{student_work}", rubric="{rubric}")
    for msg in compiled:
        langfuse_messages.append((msg["role"], msg["content"]))

    handler_lf = create_callback_handler(
        settings,
        trace_name="prompt-test-langfuse",
        tags=["prompt-test", "langfuse-managed"],
    )
    lf_result, lf_latency, lf_usage = await run_with_prompt(
        request.student_work, rubric_text, langfuse_messages, llm, handler_lf,
    )

    handler_hc = create_callback_handler(
        settings,
        trace_name="prompt-test-hardcoded",
        tags=["prompt-test", "hardcoded"],
    )
    hc_result, hc_latency, hc_usage = await run_with_prompt(
        request.student_work, rubric_text, HARDCODED_ASSESSMENT_PROMPT, llm, handler_hc,
    )

    trace_id = handler_lf.get_trace_id()
    langfuse.flush()

    return PromptTestResponse(
        langfuse_prompt=PromptComparisonResult(
            source="langfuse",
            overall_score=lf_result.overall_score,
            summary=lf_result.summary,
            latency_ms=round(lf_latency, 1),
            token_usage=lf_usage,
        ),
        hardcoded_prompt=PromptComparisonResult(
            source="hardcoded",
            overall_score=hc_result.overall_score,
            summary=hc_result.summary,
            latency_ms=round(hc_latency, 1),
            token_usage=hc_usage,
        ),
        trace_id=trace_id,
        trace_url=handler_lf.get_trace_url(),
    )


@router.post("/experiment")
async def run_dataset_experiment(
    request: DatasetRunRequest,
    llm: LLMDep,
) -> ExperimentResponse:
    settings = get_settings()
    langfuse = create_langfuse_client(settings)

    items = [
        {
            "student_work": item.student_work,
            "expected_score": item.expected_score,
            "metadata": item.metadata,
        }
        for item in request.items
    ]

    await create_dataset_with_items(
        langfuse,
        name=request.dataset_name,
        items=items,
        description="Experiment dataset for prompt comparison",
    )

    runs = []
    all_trace_ids = []

    prompt_templates = {
        "v1": [
            ("system", "Assess the student work.\nRubric:\n{rubric}"),
            ("human", "{student_work}"),
        ],
        "v2": [
            (
                "system",
                "You are an expert educator. Assess the student work against "
                "each criterion. Be specific and constructive.\nRubric:\n{rubric}",
            ),
            ("human", "Student work:\n{student_work}"),
        ],
    }

    for version in request.prompt_versions:
        template = prompt_templates.get(version, prompt_templates["v1"])
        run_name = f"{request.run_prefix}-{version}"

        result = await run_experiment(
            langfuse, settings, request.dataset_name, llm, template, run_name,
        )

        runs.append(ExperimentResult(
            run_name=result["run_name"],
            prompt_version=version,
            avg_score_diff=result["avg_score_diff"],
            items_processed=result["items_processed"],
            total_cost_usd=0.0,
            avg_latency_ms=result["avg_latency_ms"],
        ))
        all_trace_ids.extend(result["trace_ids"])

    langfuse.flush()

    return ExperimentResponse(
        dataset_name=request.dataset_name,
        runs=runs,
        trace_ids=all_trace_ids,
    )


@router.post("/score")
async def assess_and_score(
    request: ScoreRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> ScoreResponse:
    settings = get_settings()
    langfuse = create_langfuse_client(settings)
    rubric_text = _format_rubric(request.rubric_id, rubrics)

    handler = create_callback_handler(
        settings,
        trace_name="assess-and-score",
        tags=["scoring", "llm-judge"],
    )

    from app.schemas.assessment import AssessmentResponse
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(HARDCODED_ASSESSMENT_PROMPT)
    chain = prompt | llm.with_structured_output(AssessmentResponse)

    result = await chain.ainvoke(
        {"student_work": request.student_work, "rubric": rubric_text},
        config={"callbacks": [handler]},
    )

    trace_id = handler.get_trace_id()
    all_scores = []

    if request.auto_score:
        judge_handler = create_callback_handler(
            settings,
            trace_name="llm-judge",
            tags=["scoring", "judge"],
        )
        judge_scores = await llm_judge_score(
            request.student_work,
            result.model_dump(),
            llm,
            judge_handler,
        )
        posted = await score_trace(langfuse, trace_id, judge_scores)
        all_scores.extend(posted)

    if request.custom_scores:
        custom = [
            {"name": s.name, "value": s.value, "comment": s.comment}
            for s in request.custom_scores
        ]
        posted = await score_trace(langfuse, trace_id, custom)
        all_scores.extend(posted)

    langfuse.flush()

    return ScoreResponse(
        trace_id=trace_id,
        assessment_result=result.model_dump(),
        scores=all_scores,
        trace_url=handler.get_trace_url(),
    )


@router.get("/analytics")
async def get_analytics(
    limit: int = 50,
    tags: str | None = None,
) -> AnalyticsResponse:
    settings = get_settings()
    langfuse = create_langfuse_client(settings)

    tag_list = tags.split(",") if tags else None
    data = compute_analytics(langfuse, limit=limit, tags=tag_list)

    return AnalyticsResponse(
        total_traces=data["total_traces"],
        total_cost_usd=data["total_cost_usd"],
        cost_by_model=[CostBreakdown(**c) for c in data["cost_by_model"]],
        latency=LatencyStats(**data["latency"]),
        avg_scores=data["avg_scores"],
        period=data["period"],
    )
```

### Шаг 4. Регистрация в роутере

В `app/api/router.py` добавляем новый модуль:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, prompts, langfuse

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(langfuse.router)
```

В `app/config.py` добавляем настройки Langfuse:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

В `.env` добавляем ключи:

```
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Шаг 5. Тестирование с curl

**1. Тест промптов — сравнение Langfuse-managed vs hardcoded:**

```bash
curl -X POST http://localhost:8000/api/v1/langfuse/prompt-test \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is a significant global challenge. Rising temperatures lead to melting ice caps, rising sea levels, and extreme weather events. Governments must implement carbon reduction policies. Renewable energy adoption is critical for sustainability.",
    "rubric_id": "essay_default",
    "prompt_name": "assessment-prompt",
    "prompt_label": "production"
  }'
```

Ожидаемый ответ:

```json
{
  "langfuse_prompt": {
    "source": "langfuse",
    "overall_score": 72,
    "summary": "Solid overview of climate change with clear structure...",
    "latency_ms": 3450.2,
    "token_usage": {"input_tokens": 1250, "output_tokens": 800}
  },
  "hardcoded_prompt": {
    "source": "hardcoded",
    "overall_score": 70,
    "summary": "Good general overview but lacks depth...",
    "latency_ms": 3120.8,
    "token_usage": {"input_tokens": 1100, "output_tokens": 750}
  },
  "trace_id": "abc-123-def",
  "trace_url": "https://cloud.langfuse.com/trace/abc-123-def"
}
```

**2. Эксперимент — прогон dataset с двумя версиями промпта:**

```bash
curl -X POST http://localhost:8000/api/v1/langfuse/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "assessment-test-001",
    "items": [
      {
        "student_work": "The water cycle describes how water moves through the environment. Water evaporates from oceans, forms clouds through condensation, and returns as precipitation.",
        "expected_score": 65
      },
      {
        "student_work": "Photosynthesis is the process by which plants convert sunlight into energy. Using chlorophyll in their leaves, plants absorb CO2 and water to produce glucose and oxygen. This process is fundamental to life on Earth, forming the base of most food chains and producing the oxygen we breathe.",
        "expected_score": 82
      },
      {
        "student_work": "Shakespeare wrote plays.",
        "expected_score": 15
      }
    ],
    "prompt_versions": ["v1", "v2"],
    "run_prefix": "bootcamp-exp"
  }'
```

Ожидаемый ответ:

```json
{
  "dataset_name": "assessment-test-001",
  "runs": [
    {
      "run_name": "bootcamp-exp-v1",
      "prompt_version": "v1",
      "avg_score_diff": 12.3,
      "items_processed": 3,
      "total_cost_usd": 0.0,
      "avg_latency_ms": 3200.5
    },
    {
      "run_name": "bootcamp-exp-v2",
      "prompt_version": "v2",
      "avg_score_diff": 8.1,
      "items_processed": 3,
      "total_cost_usd": 0.0,
      "avg_latency_ms": 3450.2
    }
  ],
  "trace_ids": ["id-1", "id-2", "id-3", "id-4", "id-5", "id-6"]
}
```

**3. Оценка с auto-scoring (LLM-as-judge):**

```bash
curl -X POST http://localhost:8000/api/v1/langfuse/score \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The French Revolution began in 1789 and fundamentally transformed French society. Key causes included fiscal crisis, social inequality under the Estates system, and Enlightenment ideas. The storming of the Bastille symbolized the uprising. The revolution led to the Declaration of the Rights of Man and eventually Napoleons rise to power.",
    "rubric_id": "essay_default",
    "auto_score": true,
    "custom_scores": [
      {"name": "relevance", "value": 0.9, "comment": "On topic and comprehensive"}
    ]
  }'
```

Ожидаемый ответ:

```json
{
  "trace_id": "trace-xyz",
  "assessment_result": {
    "overall_score": 78,
    "max_overall_score": 100,
    "criterion_scores": [...],
    "summary": "Well-structured essay on the French Revolution...",
    "strengths": ["Clear chronological structure", "Key events identified"],
    "improvements": ["Add more specific dates", "Discuss social impacts"]
  },
  "scores": [
    {"name": "accuracy", "value": 0.85, "trace_id": "trace-xyz"},
    {"name": "completeness", "value": 0.78, "trace_id": "trace-xyz"},
    {"name": "helpfulness", "value": 0.82, "trace_id": "trace-xyz"},
    {"name": "relevance", "value": 0.9, "trace_id": "trace-xyz"}
  ],
  "trace_url": "https://cloud.langfuse.com/trace/trace-xyz"
}
```

**4. Аналитика — стоимость и латентность:**

```bash
curl "http://localhost:8000/api/v1/langfuse/analytics?limit=100&tags=production"
```

Ожидаемый ответ:

```json
{
  "total_traces": 87,
  "total_cost_usd": 1.234567,
  "cost_by_model": [
    {
      "model": "claude-sonnet-4-20250514",
      "total_cost_usd": 1.15,
      "input_tokens": 125000,
      "output_tokens": 80000,
      "call_count": 87
    }
  ],
  "latency": {
    "p50_ms": 3200.0,
    "p95_ms": 8100.0,
    "p99_ms": 15400.0,
    "avg_ms": 4150.3
  },
  "avg_scores": {
    "accuracy": 0.856,
    "completeness": 0.791,
    "helpfulness": 0.823
  },
  "period": "last 87 traces"
}
```

### Связь с теорией

| Эндпоинт | Секции теории |
|----------|--------------|
| `POST /langfuse/prompt-test` | §2 Prompt Management — загрузка промпта через `get_prompt`, сравнение с hardcoded |
| `POST /langfuse/experiment` | §3 Datasets — создание dataset, запуск experiment, `item.link()` |
| `POST /langfuse/score` | §4 Scores — LLM-as-judge, `langfuse.score()`, evaluation pipeline |
| `GET /langfuse/analytics` | §6 Cost Analytics — cost by model, latency percentiles |

Каждый эндпоинт демонстрирует конкретный столп Langfuse из §1:

- **prompt-test** → Prompt Management + Tracing
- **experiment** → Datasets + Evaluation
- **score** → Evaluation (automated + manual)
- **analytics** → Cost Analytics

---

## Чеклист самопроверки

- [ ] Назовите 5 столпов Langfuse и объясните, какую проблему решает каждый.
- [ ] Чем `get_prompt(label="production")` отличается от `get_prompt(version=3)`? Когда использовать каждый вариант?
- [ ] Опишите workflow эксперимента: от создания dataset до сравнения runs. Какие API-вызовы нужны на каждом шаге?
- [ ] Какие типы scores поддерживает Langfuse? Приведите пример использования каждого типа в контексте assessment системы.
- [ ] Что произойдёт, если не вызвать `langfuse.flush()` при shutdown FastAPI-приложения?
- [ ] Как `session_id` помогает при анализе traces? Приведите пример для multi-turn interaction.
- [ ] Почему `cache_ttl_seconds=300` рекомендуется для production, а не `cache_ttl_seconds=0`?
- [ ] Чем LLM-as-judge отличается от heuristic scoring? Когда выбрать какой подход?
- [ ] Объясните, как `item.link(trace_or_observation, run_name)` связывает trace с dataset item. Зачем нужен `run_name`?
- [ ] Какие метрики латентности (p50, p95, p99) вы бы мониторили в production и какие пороги установили?

---

## Частые ошибки

### 1. Забытый flush при shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
```

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()
```

Langfuse SDK буферизует данные и отправляет их батчами. Без `flush()` при shutdown последние traces могут потеряться. Особенно критично для коротких скриптов и экспериментов.

### 2. Hardcoded ключи вместо env vars

```python
langfuse = Langfuse(
    public_key="pk-lf-abc123",
    secret_key="sk-lf-xyz789",
)
```

```python
langfuse = Langfuse()
```

SDK автоматически читает `LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY` из окружения. Хардкод ключей — security risk: они попадут в git history.

### 3. Неправильный формат переменных в промптах

```python
langfuse.create_prompt(
    name="my-prompt",
    prompt=[{
        "role": "user",
        "content": "Assess: {student_work}"
    }],
    type="chat",
)
```

```python
langfuse.create_prompt(
    name="my-prompt",
    prompt=[{
        "role": "user",
        "content": "Assess: {{student_work}}"
    }],
    type="chat",
)
```

Langfuse использует двойные фигурные скобки `{{var}}` для переменных, не одинарные `{var}`. Одинарные скобки не будут распознаны как placeholder.

### 4. Создание нового CallbackHandler для каждого вызова в цепочке

```python
handler1 = CallbackHandler(trace_name="step1")
result1 = await chain1.ainvoke(data, config={"callbacks": [handler1]})

handler2 = CallbackHandler(trace_name="step2")
result2 = await chain2.ainvoke(result1, config={"callbacks": [handler2]})
```

```python
handler = CallbackHandler(trace_name="full-pipeline")
result1 = await chain1.ainvoke(data, config={"callbacks": [handler]})
result2 = await chain2.ainvoke(result1, config={"callbacks": [handler]})
```

Два handler'а создадут два отдельных trace. Один handler объединит оба вызова в один trace с nested observations — что отражает реальный pipeline.

### 5. Score без trace_id

```python
langfuse.score(
    name="accuracy",
    value=0.85,
)
```

```python
langfuse.score(
    trace_id=handler.get_trace_id(),
    name="accuracy",
    value=0.85,
)
```

`trace_id` — обязательный параметр. Score без привязки к trace бесполезен: его невозможно найти в UI и связать с конкретным вызовом.

### 6. Использование get_prompt без fallback

```python
prompt = langfuse.get_prompt(name="my-prompt", label="production")
```

```python
fallback_template = [{"role": "user", "content": "Assess: {{student_work}}"}]

prompt = langfuse.get_prompt(
    name="my-prompt",
    label="production",
    fallback=fallback_template,
)
```

Если Langfuse недоступен (сеть, maintenance), `get_prompt` без fallback бросит исключение и сломает весь pipeline. Fallback гарантирует работу приложения даже при недоступности Langfuse.

---

## Что читать дальше

- [Langfuse Documentation](https://langfuse.com/docs) — полная документация платформы
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python/decorators) — reference по Python SDK
- [Prompt Management](https://langfuse.com/docs/prompts/get-started) — управление промптами
- [Datasets & Experiments](https://langfuse.com/docs/datasets/overview) — работа с datasets
- [Scores & Evaluation](https://langfuse.com/docs/scores/overview) — система оценки качества
- [LangChain Integration](https://langfuse.com/docs/integrations/langchain/tracing) — интеграция с LangChain
- [Self-hosting Guide](https://langfuse.com/docs/deployment/self-host) — развёртывание self-hosted
- [Langfuse Cookbook](https://langfuse.com/docs/guides/cookbook) — примеры и рецепты

**Следующая тема:** [Тема 17: LangSmith](topic_17_langsmith.md) — альтернативная платформа для LLM ops.

# Тема 16: Langfuse Deep Dive — платформа для LLM observability

> **Пререквизиты:** [Тема 8: Observability](topic_08_observability.md)
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

## Практика

### Пример 1. Prompt Management — версионирование промптов

Создаём промпт в Langfuse, загружаем его по имени и метке, компилируем с переменными и используем в LangChain chain.

```python
from langfuse import Langfuse

langfuse = Langfuse()

langfuse.create_prompt(
    name="assessment-prompt",
    prompt=[
        {
            "role": "system",
            "content": (
                "You are an expert educator and assessment specialist.\n"
                "Evaluate the student work carefully against each criterion "
                "in the rubric.\n"
                "Be specific in your feedback and provide actionable suggestions.\n\n"
                "Rubric:\n{{rubric}}"
            ),
        },
        {
            "role": "user",
            "content": "Please assess the following student work:\n\n{{student_work}}",
        },
    ],
    labels=["production"],
    type="chat",
    config={"model": "claude-sonnet-4-20250514", "temperature": 0.3},
)

langfuse.flush()
print("Prompt created")
```

Загрузка промпта и компиляция с переменными:

```python
prompt = langfuse.get_prompt(
    name="assessment-prompt",
    label="production",
    type="chat",
    cache_ttl_seconds=300,
)

print(f"Version: {prompt.version}")
print(f"Labels: {prompt.labels}")
print(f"Config: {prompt.config}")

messages = prompt.compile(
    student_work="Climate change is a significant global challenge...",
    rubric="Clarity: 0-25, Argumentation: 0-25, Evidence: 0-25, Structure: 0-25",
)
for msg in messages:
    print(f"[{msg['role']}] {msg['content'][:80]}...")
```

Загружаем промпт как LangChain `ChatPromptTemplate` и прогоняем chain:

```python
from langfuse.callback import CallbackHandler
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


class AssessmentResult(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    summary: str
    strengths: list[str]
    improvements: list[str]


prompt = langfuse.get_prompt(
    name="assessment-prompt", label="production", type="chat",
)
lc_prompt = prompt.get_langchain_prompt()

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)
chain = lc_prompt | llm.with_structured_output(AssessmentResult)

handler = CallbackHandler(
    trace_name="prompt-test",
    tags=["prompt-management", "notebook"],
)

result = chain.invoke(
    {
        "student_work": (
            "Climate change is a significant global challenge. "
            "Rising temperatures lead to melting ice caps, rising sea levels, "
            "and extreme weather events. Governments must implement carbon "
            "reduction policies. Renewable energy adoption is critical."
        ),
        "rubric": (
            "Clarity: 0-25, Argumentation: 0-25, "
            "Evidence: 0-25, Structure: 0-25"
        ),
    },
    config={"callbacks": [handler]},
)

print(f"Score: {result.overall_score}")
print(f"Summary: {result.summary}")
print(f"Trace URL: {handler.get_trace_url()}")
langfuse.flush()
```

Сравнение Langfuse-managed промпта с hardcoded:

```python
import time
from langchain_core.prompts import ChatPromptTemplate

hardcoded_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess the student work.\nRubric:\n{rubric}"),
    ("human", "{student_work}"),
])

managed_prompt = langfuse.get_prompt(
    name="assessment-prompt", label="production", type="chat",
)

student_text = (
    "The water cycle describes how water moves through the environment. "
    "Water evaporates from oceans, forms clouds through condensation, "
    "and returns as precipitation."
)
rubric_text = (
    "Clarity: 0-25, Argumentation: 0-25, "
    "Evidence: 0-25, Structure: 0-25"
)

for label, prompt_tpl in [
    ("hardcoded", hardcoded_prompt),
    ("managed", managed_prompt.get_langchain_prompt()),
]:
    handler = CallbackHandler(
        trace_name=f"compare-{label}",
        tags=["comparison", label],
    )
    chain = prompt_tpl | llm.with_structured_output(AssessmentResult)
    start = time.perf_counter()
    res = chain.invoke(
        {"student_work": student_text, "rubric": rubric_text},
        config={"callbacks": [handler]},
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[{label}] score={res.overall_score}, latency={elapsed:.0f}ms")

langfuse.flush()
```

### Пример 2. Datasets и эксперименты

Создаём golden dataset, прогоняем chain на каждом item, связываем trace с dataset через `item.link()` и сравниваем runs.

```python
from langfuse import Langfuse

langfuse = Langfuse()

langfuse.create_dataset(
    name="assessment-golden-v1",
    description="Golden dataset for assessment quality testing",
    metadata={"version": "1.0"},
)

test_items = [
    {
        "input": {
            "student_work": (
                "The water cycle describes how water moves through "
                "the environment. Water evaporates from oceans, "
                "forms clouds, returns as rain."
            ),
            "rubric": (
                "Clarity: 0-25, Argumentation: 0-25, "
                "Evidence: 0-25, Structure: 0-25"
            ),
        },
        "expected": {"overall_score": 65},
    },
    {
        "input": {
            "student_work": (
                "Photosynthesis is the process by which plants convert "
                "sunlight into energy. Using chlorophyll, plants absorb "
                "CO2 and water to produce glucose and oxygen. This process "
                "is fundamental to life on Earth."
            ),
            "rubric": (
                "Clarity: 0-25, Argumentation: 0-25, "
                "Evidence: 0-25, Structure: 0-25"
            ),
        },
        "expected": {"overall_score": 82},
    },
    {
        "input": {
            "student_work": "Shakespeare wrote plays.",
            "rubric": (
                "Clarity: 0-25, Argumentation: 0-25, "
                "Evidence: 0-25, Structure: 0-25"
            ),
        },
        "expected": {"overall_score": 15},
    },
]

for item in test_items:
    langfuse.create_dataset_item(
        dataset_name="assessment-golden-v1",
        input=item["input"],
        expected_output=item["expected"],
    )

langfuse.flush()
print(f"Dataset created with {len(test_items)} items")
```

Прогон эксперимента — chain обрабатывает каждый item, результат связывается с dataset:

```python
import statistics
from langfuse.callback import CallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AssessmentResult(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    summary: str
    strengths: list[str]
    improvements: list[str]


llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

prompt_versions = {
    "v1": ChatPromptTemplate.from_messages([
        ("system", "Assess the student work.\nRubric:\n{rubric}"),
        ("human", "{student_work}"),
    ]),
    "v2": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert educator. Assess the student work against "
            "each criterion. Be specific and constructive.\n"
            "Rubric:\n{rubric}",
        ),
        ("human", "Student work:\n{student_work}"),
    ]),
}

dataset = langfuse.get_dataset("assessment-golden-v1")

for version_name, prompt_tpl in prompt_versions.items():
    run_name = f"experiment-{version_name}"
    chain = prompt_tpl | llm.with_structured_output(AssessmentResult)
    diffs = []

    for item in dataset.items:
        handler = CallbackHandler(
            trace_name=f"{run_name}-{item.id}",
            tags=["experiment", run_name],
        )
        result = chain.invoke(item.input, config={"callbacks": [handler]})
        handler.trace.update(output=result.model_dump())

        expected_score = item.expected_output["overall_score"]
        diff = abs(result.overall_score - expected_score)
        diffs.append(diff)

        langfuse.score(
            trace_id=handler.get_trace_id(),
            name="score_accuracy",
            value=max(0, 1 - diff / 100),
        )

        item.link(
            trace_or_observation=handler.trace,
            run_name=run_name,
        )

    avg_diff = statistics.mean(diffs)
    print(f"[{version_name}] avg_score_diff={avg_diff:.1f}, items={len(diffs)}")

langfuse.flush()
```

### Пример 3. Online evaluation — scoring

Запускаем chain с трейсингом, затем оцениваем результат тремя способами: эвристика, LLM-as-judge, кастомный score.

```python
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AssessmentResult(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    summary: str
    strengths: list[str]
    improvements: list[str]


langfuse = Langfuse()
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert educator. Assess the student work "
        "against the rubric.\nRubric:\n{rubric}",
    ),
    ("human", "Student work:\n{student_work}"),
])
chain = prompt | llm.with_structured_output(AssessmentResult)

handler = CallbackHandler(
    trace_name="assess-and-score",
    tags=["scoring", "notebook"],
)

result = chain.invoke(
    {
        "student_work": (
            "The French Revolution began in 1789 and fundamentally "
            "transformed French society. Key causes included fiscal crisis, "
            "social inequality, and Enlightenment ideas. The storming of "
            "the Bastille symbolized the uprising."
        ),
        "rubric": "Clarity: 0-25, Argumentation: 0-25, Evidence: 0-25, Structure: 0-25",
    },
    config={"callbacks": [handler]},
)

trace_id = handler.get_trace_id()
print(f"Assessment score: {result.overall_score}")
print(f"Trace ID: {trace_id}")
```

Эвристические scores — простые программные проверки без LLM:

```python
langfuse.score(
    trace_id=trace_id,
    name="format_valid",
    value=1 if 0 <= result.overall_score <= 100 else 0,
    data_type="BOOLEAN",
    comment="Score within valid range",
)

langfuse.score(
    trace_id=trace_id,
    name="feedback_length",
    value=min(1.0, len(result.summary) / 200),
    data_type="NUMERIC",
    comment="Summary length normalized to 200 chars",
)

print(f"Heuristic scores posted for trace {trace_id}")
```

LLM-as-judge — отдельный LLM оценивает качество результата:

```python
class JudgeScores(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    helpfulness: float = Field(ge=0, le=1)


judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a quality evaluator. Given a student work and an AI "
        "assessment, rate the assessment quality on three dimensions.\n"
        "Return JSON with fields: accuracy (0-1), completeness (0-1), "
        "helpfulness (0-1).",
    ),
    (
        "human",
        "Student work:\n{student_work}\n\n"
        "Assessment:\n{assessment}\n\nRate:",
    ),
])

judge_chain = judge_prompt | llm.with_structured_output(JudgeScores)

judge_handler = CallbackHandler(
    trace_name="llm-judge",
    tags=["scoring", "judge"],
)

judge_result = judge_chain.invoke(
    {
        "student_work": (
            "The French Revolution began in 1789 and fundamentally "
            "transformed French society..."
        ),
        "assessment": result.model_dump_json(),
    },
    config={"callbacks": [judge_handler]},
)

for score_name in ["accuracy", "completeness", "helpfulness"]:
    langfuse.score(
        trace_id=trace_id,
        name=score_name,
        value=getattr(judge_result, score_name),
        data_type="NUMERIC",
    )

print(f"accuracy={judge_result.accuracy}")
print(f"completeness={judge_result.completeness}")
print(f"helpfulness={judge_result.helpfulness}")
```

Кастомный score — оценка из кода (или user feedback):

```python
langfuse.score(
    trace_id=trace_id,
    name="relevance",
    value=0.9,
    data_type="NUMERIC",
    comment="On topic and comprehensive",
)

langfuse.flush()
print("All scores posted")
```

### Пример 4. Dashboard queries — аналитика

Загружаем traces и observations через API для расчёта стоимости по моделям, латентности и средних scores.

```python
import statistics
from collections import defaultdict
from langfuse import Langfuse

langfuse = Langfuse()

traces = langfuse.fetch_traces(limit=50)
print(f"Total traces: {len(traces.data)}")

model_costs = defaultdict(lambda: {
    "cost": 0, "input_tokens": 0, "output_tokens": 0, "count": 0,
})
latencies = []
score_values = defaultdict(list)
total_cost = 0

for trace in traces.data:
    if trace.latency:
        latencies.append(trace.latency * 1000)

    observations = langfuse.fetch_observations(trace_id=trace.id)
    for obs in observations.data:
        if obs.type == "GENERATION":
            model_name = obs.model or "unknown"
            cost = obs.calculated_total_cost or 0
            total_cost += cost
            model_costs[model_name]["cost"] += cost
            model_costs[model_name]["count"] += 1
            if obs.usage:
                model_costs[model_name]["input_tokens"] += obs.usage.input or 0
                model_costs[model_name]["output_tokens"] += obs.usage.output or 0

    scores = langfuse.fetch_scores(trace_id=trace.id)
    for sc in scores.data if hasattr(scores, "data") else []:
        if sc.value is not None:
            score_values[sc.name].append(sc.value)
```

```python
print(f"Total cost: ${total_cost:.4f}\n")

for model, data in model_costs.items():
    print(f"Model: {model}")
    print(f"  Calls: {data['count']}")
    print(f"  Cost: ${data['cost']:.4f}")
    print(f"  Input tokens: {data['input_tokens']}")
    print(f"  Output tokens: {data['output_tokens']}")
    print()

if latencies:
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    print(f"Latency p50: {sorted_lat[n // 2]:.0f}ms")
    print(f"Latency p95: {sorted_lat[int(n * 0.95)]:.0f}ms")
    print(f"Latency p99: {sorted_lat[int(n * 0.99)]:.0f}ms")
    print(f"Latency avg: {statistics.mean(sorted_lat):.0f}ms\n")

for name, vals in score_values.items():
    print(f"Score '{name}': avg={statistics.mean(vals):.3f} (n={len(vals)})")
```

### Связь с теорией

| Пример | Секции теории |
|--------|--------------|
| Пример 1: Prompt Management | §2 — загрузка промпта через `get_prompt`, `compile`, `get_langchain_prompt` |
| Пример 2: Datasets и эксперименты | §3 — создание dataset, запуск experiment, `item.link()` |
| Пример 3: Online evaluation | §4 — Scores, LLM-as-judge, heuristic scoring, `langfuse.score()` |
| Пример 4: Dashboard queries | §6 — Cost analytics, latency percentiles, score aggregation |

Каждый пример демонстрирует конкретный столп Langfuse из §1:

- **Пример 1** → Prompt Management + Tracing
- **Пример 2** → Datasets + Evaluation
- **Пример 3** → Evaluation (automated + manual)
- **Пример 4** → Cost Analytics

---


## Чеклист самопроверки

- [ ] Назовите 5 столпов Langfuse и объясните, какую проблему решает каждый.
- [ ] Чем `get_prompt(label="production")` отличается от `get_prompt(version=3)`? Когда использовать каждый вариант?
- [ ] Опишите workflow эксперимента: от создания dataset до сравнения runs. Какие API-вызовы нужны на каждом шаге?
- [ ] Какие типы scores поддерживает Langfuse? Приведите пример использования каждого типа в контексте assessment системы.
- [ ] Что произойдёт, если не вызвать `langfuse.flush()` в конце скрипта или эксперимента?
- [ ] Как `session_id` помогает при анализе traces? Приведите пример для multi-turn interaction.
- [ ] Почему `cache_ttl_seconds=300` рекомендуется для production, а не `cache_ttl_seconds=0`?
- [ ] Чем LLM-as-judge отличается от heuristic scoring? Когда выбрать какой подход?
- [ ] Объясните, как `item.link(trace_or_observation, run_name)` связывает trace с dataset item. Зачем нужен `run_name`?
- [ ] Какие метрики латентности (p50, p95, p99) вы бы мониторили в production и какие пороги установили?

---

## Частые ошибки

### 1. Забытый flush

```python
result = chain.invoke(...)
langfuse.score(trace_id=trace_id, name="accuracy", value=0.9)
```

```python
result = chain.invoke(...)
langfuse.score(trace_id=trace_id, name="accuracy", value=0.9)
langfuse.flush()
```

Langfuse SDK буферизует данные и отправляет их батчами. Без `flush()` в конце скрипта или notebook-ячейки последние traces и scores могут потеряться. Вызывайте `flush()` после завершения эксперимента или в `finally`-блоке.

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

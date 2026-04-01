# Тема 8: Observability

> **Пререквизиты:** [Тема 1-4](topic_01_prompt_engineering.md), рекомендуется [Тема 6](topic_06_langgraph_agents.md)
> **Зависимости:** `langfuse`

---

## Теория

### 1. Зачем трейсинг для LLM

LLM-вызовы кардинально отличаются от обычных API-вызовов, и классические инструменты мониторинга (Datadog, Prometheus, Grafana) не покрывают специфику LLM:

**Недетерминизм.** Один и тот же input может давать разные output — даже при `temperature=0` есть микро-вариации. Нельзя написать unit-тест "на вход X ожидаем выход Y". Вместо этого нужно логировать каждый вызов и анализировать распределение результатов. Trace-данные — это основа для offline-evaluation: вы можете перезапустить промпт на тех же данных и сравнить качество.

**Высокая стоимость.** Каждый вызов стоит деньги (входные + выходные токены). Неоптимальный промпт может стоить в 5-10 раз дороже оптимального — разница между 2000 и 200 input-токенов при сохранении качества. Без трейсинга вы не знаете, сколько платите за каждую операцию, и не можете обнаружить аномалии: случайно раздутый промпт, retry-цикл, который делает 5 вызовов вместо одного.

**Латентность.** LLM-вызов занимает 2-30 секунд — на порядки больше, чем обычный API-вызов (~50ms). В цепочке из нескольких LLM-вызовов (assessment → summarization → scoring) общая латентность может достигать минуты. Трейсинг показывает, где именно bottleneck: в промпте (слишком длинный), в модели (слишком мощная для задачи) или в оркестрации (последовательные вызовы, которые можно параллелить).

**Качество.** "Работает" и "работает хорошо" — разные вещи. Модель может давать синтаксически корректный JSON, но с бессмысленными оценками. Трейсинг позволяет привязать **scores** к каждому вызову — автоматические (критерии из evaluation) или от пользователей (thumbs up/down). Со временем это строит dataset для анализа качества.

Трейсинг для LLM — это как APM для обычных сервисов, но с фокусом на четыре измерения: промпты (что отправлено), токены (сколько стоит), латентность (сколько занимает) и качество (насколько хорошо).

### 2. Langfuse — архитектура и модель данных

Langfuse — open-source платформа для observability LLM-приложений. Можно использовать cloud-версию (cloud.langfuse.com, бесплатный tier) или развернуть self-hosted через Docker.

Langfuse моделирует данные через иерархию вложенных сущностей:

```
Trace (полный запрос пользователя)
├── Span "preprocessing" (обработка input)
│   └── Generation "summarize" (LLM-вызов для суммаризации)
├── Span "assessment" (основная оценка)
│   └── Generation "assess" (LLM-вызов: промпт → ответ, токены, стоимость)
├── Span "postprocessing" (парсинг, валидация)
└── Score "overall_quality" = 0.75
```

| Сущность | Описание | Ключевые поля |
|----------|----------|---------------|
| **Trace** | Полный запрос пользователя от входа до ответа. Корневой элемент иерархии. | `id`, `name`, `input`, `output`, `metadata`, `user_id`, `session_id`, `tags` |
| **Span** | Логический шаг внутри trace: preprocessing, retrieval, formatting. | `name`, `input`, `output`, `start_time`, `end_time`, `metadata` |
| **Generation** | LLM-вызов: содержит промпт, ответ, модель, токены, стоимость, латентность. | `model`, `input`, `output`, `usage` (input/output tokens), `model_parameters`, `total_cost` |
| **Score** | Оценка качества, привязанная к trace или observation. | `name`, `value` (float или string), `trace_id`, `observation_id`, `comment` |

**Sessions.** Trace может быть частью сессии (`session_id`). Все traces одной сессии группируются — это полезно для multi-turn conversations, где каждый ход = отдельный trace, но логически они связаны.

**Projects.** Langfuse поддерживает несколько проектов. Стандартная практика: `assessment-dev` для разработки, `assessment-prod` для production. Разные ключи API, разные данные, единый dashboard.

Langfuse автоматически считает стоимость по модели и количеству токенов. Для поддерживаемых моделей (Claude, GPT-4, Llama и др.) стоимость рассчитывается без конфигурации:

```
Generation: claude-sonnet-4-20250514
  Input tokens:  1,250 → $0.00375
  Output tokens:   800 → $0.01200
  Total cost:           $0.01575
```

### 3. Callback handlers в LangChain

LangChain реализует паттерн **Observer**: на каждом этапе выполнения chain вызываются callback-функции. Langfuse предоставляет готовый callback handler, который перехватывает все события и автоматически создаёт traces, spans и generations в Langfuse.

Последовательность событий при вызове assessment chain:

1. `on_chain_start` — chain начинает выполнение → Langfuse создаёт **Trace**
2. Prompt рендерится (переменные подставляются в шаблон) → Langfuse записывает **rendered prompt**
3. `on_chat_model_start` — LLM получает сообщения → Langfuse создаёт **Generation**, записывает input
4. `on_llm_end` — LLM вернул ответ → Langfuse записывает output, токены, latency в Generation
5. Output парсится (structured output → Pydantic) → Langfuse записывает parsed output
6. `on_chain_end` — chain завершил работу → Langfuse закрывает Trace

Всё это происходит **автоматически** — достаточно передать callback handler в конфигурацию вызова:

```python
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

handler = LangfuseCallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
    trace_name="essay_assessment",
    user_id="teacher-42",
    metadata={"rubric": "essay_default"},
)

result = await chain.ainvoke(
    {"student_work": "...", "rubric": "..."},
    config={"callbacks": [handler]},
)
```

Каждый вызов `ainvoke` с handler создаёт отдельный trace. Handler можно переиспользовать или создавать новый для каждого запроса (рекомендуется для production, чтобы каждый trace имел уникальные metadata).

Callback-система расширяема: вы можете написать свой handler, наследуя `BaseCallbackHandler` или `AsyncCallbackHandler`, и ловить любые события — для логирования в stdout, отправки метрик в Prometheus, или алертинга при аномалиях.

### 4. Cost tracking и оптимизация

Langfuse автоматически агрегирует стоимость на уровне trace, позволяя анализировать расходы:

| Метрика | Как получить | Зачем |
|---------|-------------|-------|
| Стоимость за оценку | `trace.total_cost` | Понять unit economics (сколько стоит одна оценка) |
| Стоимость за день/неделю | Dashboard → Usage | Бюджетирование, планирование |
| Token breakdown | `generation.usage.input` / `output` | Найти длинные промпты (input >> output = промпт слишком большой) |
| Аномалии | Traces с cost > 2× среднего | Обнаружить retry-циклы, раздутые контексты |

Типичные стратегии оптимизации стоимости:

1. **Prompt compression.** Убрать избыточный текст из system prompt. Часто few-shot примеры можно сократить на 50% без потери качества.
2. **Model selection.** Использовать более дешёвую модель для простых задач: Claude Haiku вместо Sonnet для классификации, GPT-4o-mini вместо GPT-4o для извлечения данных.
3. **Caching.** Для идентичных input (один и тот же промпт + данные) возвращать кешированный результат вместо нового LLM-вызова.
4. **Batching.** Вместо N отдельных вызовов — один вызов с N элементами (если модель поддерживает).

### 5. Prompt management

Langfuse хранит **версии промптов** на своём сервере — можно менять промпт в UI без деплоя кода:

```python
from langfuse import Langfuse

langfuse = Langfuse()
prompt = langfuse.get_prompt("assessment-prompt", version=3)
system_text = prompt.compile(rubric="...", few_shot_good="...")
```

Жизненный цикл промпта с Langfuse prompt management:

1. **Создание** — загрузить промпт через API или UI. Каждый промпт имеет `name` и автоинкрементный `version`.
2. **Тестирование** — запустить chain с новой версией, посмотреть traces в dashboard. Labels (например, `staging`) помечают версии для разных окружений.
3. **Деплой** — пометить версию как `production`. Код в production загружает промпт по имени и label, а не по фиксированной версии.
4. **Откат** — если качество упало, пометить предыдущую версию как `production`. Без деплоя кода, без PR.

Преимущества перед хардкодом в `templates.py`:
- **Версионирование** — история изменений, diff между версиями
- **A/B testing** — разные версии промпта для разных пользователей через labels
- **Не-инженеры** — product manager может редактировать промпт в UI без PR и деплоя
- **Связь с traces** — каждый trace показывает, какая версия промпта использовалась

### 6. Scores и evaluation

Scores в Langfuse привязываются к traces или observations и бывают трёх типов:

| Тип | Пример | Кто создаёт |
|-----|--------|-------------|
| **Автоматический** | `overall_score = 0.75` (нормализованный балл) | Код после получения результата |
| **User feedback** | `thumbs_up = 1` | Пользователь через UI приложения |
| **Evaluation** | `hallucination = 0.1` | Evaluation pipeline (offline) |

Программное добавление score через Langfuse client:

```python
langfuse = Langfuse()
langfuse.score(
    trace_id="trace-abc123",
    name="overall_score",
    value=0.75,
    comment="75/100 normalized to 0-1",
)
```

Scores критичны для мониторинга качества: dashboard показывает распределение scores по времени, позволяя заметить деградацию (средний score упал после обновления промпта) или аномалии (внезапный кластер низких оценок).

---

## Справочник API

### LangfuseCallbackHandler

**Описание:** callback handler для LangChain/LangGraph, автоматически трейсит все этапы выполнения chain в Langfuse. Каждый вызов `ainvoke` / `invoke` с этим handler создаёт отдельный trace.

**Импорт:**

```python
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `public_key` | `str \| None` | `env LANGFUSE_PUBLIC_KEY` | Public key проекта |
| `secret_key` | `str \| None` | `env LANGFUSE_SECRET_KEY` | Secret key проекта |
| `host` | `str \| None` | `env LANGFUSE_HOST` | URL Langfuse (cloud или self-hosted) |
| `trace_name` | `str \| None` | `None` | Имя trace (отображается в dashboard) |
| `session_id` | `str \| None` | `None` | ID сессии (группировка traces) |
| `user_id` | `str \| None` | `None` | ID пользователя |
| `release` | `str \| None` | `None` | Версия приложения (git sha, semver) |
| `version` | `str \| None` | `None` | Версия для фильтрации в dashboard |
| `metadata` | `dict \| None` | `None` | Произвольные метаданные trace |
| `tags` | `list[str] \| None` | `None` | Теги для фильтрации |
| `debug` | `bool` | `False` | Вывод debug-логов в stdout |
| `sample_rate` | `float \| None` | `None` | Доля сэмплируемых traces (0.0-1.0) |
| `enabled` | `bool` | `True` | Включить/выключить трейсинг |

**Основные методы:**

| Метод | Возвращает | Описание |
|-------|-----------|----------|
| `get_trace_id()` | `str` | ID созданного trace (доступен после вызова chain) |
| `get_trace_url()` | `str` | Полный URL trace в Langfuse dashboard |
| `flush()` | `None` | Отправить все буферизованные данные в Langfuse |
| `auth_check()` | `bool` | Проверить валидность ключей |

**Пример:**

```python
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

handler = LangfuseCallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
    trace_name="essay_assessment",
    user_id="teacher-42",
    metadata={"rubric": "essay_default", "essay_length": 350},
    tags=["assessment", "production"],
)

result = await chain.ainvoke(
    {"student_work": "...", "rubric": "..."},
    config={"callbacks": [handler]},
)

trace_id = handler.get_trace_id()
trace_url = handler.get_trace_url()
handler.flush()
```

---

### Langfuse (клиент)

**Описание:** Python-клиент для программного взаимодействия с Langfuse API. Позволяет создавать traces, scores, управлять промптами и получать данные для аналитики.

**Импорт:**

```python
from langfuse import Langfuse
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `public_key` | `str \| None` | `env LANGFUSE_PUBLIC_KEY` | Public key проекта |
| `secret_key` | `str \| None` | `env LANGFUSE_SECRET_KEY` | Secret key проекта |
| `host` | `str \| None` | `env LANGFUSE_HOST` | URL Langfuse |
| `release` | `str \| None` | `None` | Версия приложения |
| `debug` | `bool` | `False` | Debug-режим |
| `enabled` | `bool` | `True` | Включить/выключить клиент |

**Основные методы:**

| Метод | Сигнатура | Описание |
|-------|-----------|----------|
| `trace` | `(name, input, output, user_id, session_id, metadata, tags) -> StatefulTraceClient` | Создать trace вручную (без callback handler) |
| `score` | `(trace_id, name, value, observation_id, comment, data_type) -> None` | Привязать score к trace или observation |
| `create_prompt` | `(name, prompt, config, labels, type) -> PromptClient` | Создать новую версию промпта |
| `get_prompt` | `(name, version, label, type, cache_ttl_seconds) -> PromptClient` | Получить промпт по имени (с кешированием) |
| `fetch_traces` | `(limit, offset, name, user_id, session_id, tags, order_by) -> FetchTracesResponse` | Получить список traces с фильтрацией |
| `flush` | `() -> None` | Отправить все буферизованные данные |
| `shutdown` | `() -> None` | Отправить данные и закрыть клиент |

**Пример — создание trace и score:**

```python
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(
    name="manual_assessment",
    input={"student_work": "..."},
    user_id="teacher-1",
    metadata={"rubric": "essay_default"},
)

langfuse.score(
    trace_id=trace.id,
    name="overall_score",
    value=0.75,
    comment="75/100 normalized",
)

langfuse.flush()
```

**Пример — prompt management:**

```python
langfuse = Langfuse()

prompt_client = langfuse.create_prompt(
    name="assessment-system-prompt",
    prompt="You are an expert assessor. Rubric: {{rubric}}",
    config={"temperature": 0.3, "model": "claude-sonnet-4-20250514"},
    labels=["staging"],
)

prompt = langfuse.get_prompt("assessment-system-prompt", label="staging")
compiled = prompt.compile(rubric="Essay rubric: ...")
```

**Пример — fetch traces:**

```python
langfuse = Langfuse()

response = langfuse.fetch_traces(limit=50, name="essay_assessment")
for trace in response.data:
    print(f"{trace.name}: cost=${trace.total_cost:.4f}, latency={trace.latency:.1f}s")

langfuse.shutdown()
```

---

### BaseCallbackHandler / AsyncCallbackHandler

**Описание:** базовые классы для создания callback handlers в LangChain. `BaseCallbackHandler` — синхронный, `AsyncCallbackHandler` — асинхронный. Langfuse, LangSmith и другие инструменты строятся поверх этих классов.

**Импорт:**

```python
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
```

**Hook-методы (таблица всех доступных callbacks):**

| Hook | Когда вызывается | Аргументы |
|------|-----------------|-----------|
| `on_llm_start` | Перед вызовом LLM (completion API) | `serialized`, `prompts`, `run_id`, `parent_run_id` |
| `on_chat_model_start` | Перед вызовом chat model | `serialized`, `messages`, `run_id`, `parent_run_id` |
| `on_llm_new_token` | Каждый новый токен при стриминге | `token`, `chunk`, `run_id` |
| `on_llm_end` | После завершения LLM-вызова | `response` (LLMResult), `run_id` |
| `on_llm_error` | При ошибке LLM | `error`, `run_id` |
| `on_chain_start` | Перед запуском chain | `serialized`, `inputs`, `run_id`, `parent_run_id` |
| `on_chain_end` | После завершения chain | `outputs`, `run_id` |
| `on_chain_error` | При ошибке chain | `error`, `run_id` |
| `on_tool_start` | Перед вызовом tool | `serialized`, `input_str`, `run_id` |
| `on_tool_end` | После завершения tool | `output`, `run_id` |
| `on_tool_error` | При ошибке tool | `error`, `run_id` |
| `on_retriever_start` | Перед retriever query | `serialized`, `query`, `run_id` |
| `on_retriever_end` | После получения документов | `documents`, `run_id` |
| `on_retriever_error` | При ошибке retriever | `error`, `run_id` |
| `on_text` | При генерации промежуточного текста | `text`, `run_id` |
| `on_agent_action` | Когда агент выбирает action | `action`, `run_id` |
| `on_agent_finish` | Когда агент завершает работу | `finish`, `run_id` |

**Пример кастомного handler:**

```python
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class CostAlertHandler(AsyncCallbackHandler):
    def __init__(self, max_cost: float = 0.10):
        self.max_cost = max_cost
        self.total_tokens = 0

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "generation_info") and gen.generation_info:
                    usage = gen.generation_info.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)

    async def on_chain_end(self, outputs, **kwargs) -> None:
        estimated_cost = self.total_tokens * 0.00001
        if estimated_cost > self.max_cost:
            print(f"ALERT: Estimated cost ${estimated_cost:.4f} exceeds ${self.max_cost}")
```

---

### CallbackManager

**Описание:** менеджер, который управляет коллекцией callback handlers и маршрутизирует события. Обычно создаётся автоматически LangChain при передаче `config={"callbacks": [...]}`, но можно создать вручную для сложных сценариев.

**Импорт:**

```python
from langchain_core.callbacks import CallbackManager
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `handlers` | `list[BaseCallbackHandler]` | `[]` | Список handlers для текущего run |
| `inheritable_handlers` | `list[BaseCallbackHandler]` | `[]` | Handlers, наследуемые дочерними runs |
| `parent_run_id` | `UUID \| None` | `None` | ID родительского run (для вложенности) |
| `tags` | `list[str]` | `[]` | Теги текущего run |
| `inheritable_tags` | `list[str]` | `[]` | Теги, наследуемые дочерними runs |
| `metadata` | `dict` | `{}` | Метаданные текущего run |
| `inheritable_metadata` | `dict` | `{}` | Метаданные, наследуемые дочерними runs |

**Основные методы:**

| Метод | Описание |
|-------|----------|
| `configure(callbacks, ...)` | Создать `CallbackManager` из списка handlers (classmethod) |
| `add_handler(handler)` | Добавить handler |
| `remove_handler(handler)` | Удалить handler |
| `on_chain_start(serialized, inputs)` | Вызвать `on_chain_start` у всех handlers |
| `on_llm_start(serialized, prompts)` | Вызвать `on_llm_start` у всех handlers |

**Как передавать callbacks в LangChain:**

Самый простой способ — через `config`:

```python
result = await chain.ainvoke(
    {"student_work": "..."},
    config={"callbacks": [langfuse_handler, cost_alert_handler]},
)
```

LangChain автоматически создаёт `CallbackManager`, регистрирует handlers и маршрутизирует события каждому handler.

**`inheritable_handlers`** — handlers, которые автоматически передаются вложенным runs. Например, если chain содержит sub-chain, inheritable handler будет трейсить и sub-chain:

```python
result = await chain.ainvoke(
    {"student_work": "..."},
    config={
        "callbacks": [langfuse_handler],
        "run_name": "essay_assessment",
        "metadata": {"rubric": "essay_default"},
    },
)
```

Поля `run_name` и `metadata` в config наследуются дочерними runs, обеспечивая сквозной трейсинг всей цепочки.

---

## Практика

Для работы с примерами установите зависимости и задайте переменные окружения:

```python
%pip install langfuse langchain-anthropic langchain-core
```

```python
import os

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

---

### Пример 1: Подключение Langfuse callback handler

Создаём callback handler и подключаем к простому chain. Langfuse автоматически трейсит все этапы: промпт, LLM-вызов, ответ, токены, стоимость.

```python
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

handler = LangfuseCallbackHandler(
    trace_name="hello_langfuse",
    user_id="demo-user",
    tags=["lesson-8", "demo"],
)

print(f"Auth check: {handler.auth_check()}")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "{question}"),
])

chain = prompt | llm

result = chain.invoke(
    {"question": "What is observability in LLM applications?"},
    config={"callbacks": [handler]},
)

print(f"Response: {result.content[:200]}")
print(f"Trace ID: {handler.get_trace_id()}")
print(f"Trace URL: {handler.get_trace_url()}")

handler.flush()
```

**Связь с теорией:** callback handler перехватывает события `on_chain_start`, `on_chat_model_start`, `on_llm_end`, `on_chain_end` (раздел 3) и автоматически создаёт Trace → Generation в Langfuse. `flush()` отправляет буферизованные данные.

---

### Пример 2: Трейсинг chain со structured output и score

Трейсим chain с Pydantic-output и добавляем score к trace — основа для мониторинга качества.

```python
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class TextAnalysis(BaseModel):
    summary: str = Field(description="One-sentence summary")
    sentiment: str = Field(description="positive, negative, or neutral")
    key_topics: list[str] = Field(description="Main topics mentioned")
    confidence: float = Field(description="Confidence score 0.0-1.0")


handler = LangfuseCallbackHandler(
    trace_name="text_analysis",
    user_id="analyst-1",
    session_id="analysis-session-001",
    metadata={"task": "sentiment_and_topics", "version": "1.0"},
    tags=["analysis", "structured"],
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
structured_llm = llm.with_structured_output(TextAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the following text. Return structured analysis."),
    ("human", "{text}"),
])

chain = prompt | structured_llm

text = (
    "Climate change is a global crisis. According to NASA, global temperatures "
    "have risen 1.1°C since pre-industrial times. The Paris Agreement aims to "
    "limit warming to 1.5°C, but current policies put us on track for 2.7°C by 2100."
)

result = chain.invoke(
    {"text": text},
    config={"callbacks": [handler]},
)

print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
print(f"Topics: {result.key_topics}")
print(f"Confidence: {result.confidence}")

trace_id = handler.get_trace_id()

langfuse = Langfuse()
langfuse.score(
    trace_id=trace_id,
    name="analysis_quality",
    value=result.confidence,
    comment="Auto-score from model confidence",
)
langfuse.flush()

print(f"\nTrace ID: {trace_id}")
print(f"Trace URL: {handler.get_trace_url()}")

handler.flush()
```

**Связь с теорией:** `session_id` группирует traces в сессии (раздел 2). Score привязывается к trace через `langfuse.score()` (раздел 6) — автоматический score из confidence модели. В dashboard можно отслеживать распределение scores по времени.

---

### Пример 3: Cost tracking — анализ расходов

Получаем traces из Langfuse и агрегируем данные по стоимости и латентности.

```python
from langfuse import Langfuse

langfuse = Langfuse()

response = langfuse.fetch_traces(limit=20)

total_cost = 0.0
for trace in response.data:
    cost = trace.total_cost or 0.0
    latency = (trace.latency or 0.0) * 1000
    total_cost += cost
    print(f"  {trace.name}: cost=${cost:.4f}, latency={latency:.0f}ms")

avg_cost = total_cost / len(response.data) if response.data else 0.0
print(f"\nTotal traces: {len(response.data)}")
print(f"Total cost: ${total_cost:.4f}")
print(f"Average cost per trace: ${avg_cost:.4f}")

expensive = [t for t in response.data if (t.total_cost or 0) > avg_cost * 2]
if expensive:
    print(f"\nAnomalies (cost > 2x average):")
    for t in expensive:
        print(f"  {t.id}: ${t.total_cost:.4f} ({t.name})")

langfuse.shutdown()
```

**Связь с теорией:** `fetch_traces` возвращает данные с `total_cost` и `latency` (раздел 4). Аномалии — traces со стоимостью > 2× среднего — могут указывать на retry-циклы или раздутые контексты. Эти данные — основа для стратегий оптимизации: prompt compression, model selection, caching (раздел 4).

---

### Пример 4: Prompt management — версионирование промптов

Создаём версии промпта в Langfuse, получаем по имени и label. Промпт можно менять в UI без деплоя кода.

```python
from langfuse import Langfuse

langfuse = Langfuse()

prompt_v1 = langfuse.create_prompt(
    name="analysis-system-prompt",
    prompt="You are an expert analyst. Task: {{task}}\nContext: {{context}}",
    config={"temperature": 0.3, "model": "claude-sonnet-4-20250514"},
    labels=["staging"],
)
print(f"Created: {prompt_v1.name} v{prompt_v1.version}")

prompt_v2 = langfuse.create_prompt(
    name="analysis-system-prompt",
    prompt="You are a senior analyst with 10 years of experience. Task: {{task}}\nContext: {{context}}\nBe concise.",
    config={"temperature": 0.2, "model": "claude-sonnet-4-20250514"},
    labels=["production"],
)
print(f"Created: {prompt_v2.name} v{prompt_v2.version}")

staging_prompt = langfuse.get_prompt("analysis-system-prompt", label="staging")
print(f"\nStaging (v{staging_prompt.version}): {staging_prompt.prompt[:80]}...")

prod_prompt = langfuse.get_prompt("analysis-system-prompt", label="production")
print(f"Production (v{prod_prompt.version}): {prod_prompt.prompt[:80]}...")

compiled = prod_prompt.compile(
    task="Summarize quarterly results",
    context="Revenue grew 15% YoY, operating margin improved to 22%.",
)
print(f"\nCompiled prompt:\n{compiled}")

langfuse.shutdown()
```

**Связь с теорией:** жизненный цикл промпта из раздела 5 — создание версий, labels (`staging`, `production`), получение по имени. `compile()` подставляет переменные в шаблон. Откат — назначить предыдущей версии label `production`, без изменений в коде.

---

### Пример 5: Custom spans и metadata

Создаём trace вручную с вложенными spans для multi-step pipeline. Каждый шаг — отдельный span с input/output.

```python
import time
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(
    name="multi_step_pipeline",
    user_id="demo-user",
    input={"raw_text": "Climate change is a global crisis affecting every continent."},
    metadata={"pipeline_version": "1.0", "source": "lesson-8"},
    tags=["pipeline", "demo"],
)

span_preprocess = trace.span(
    name="preprocessing",
    input={"raw_text": "Climate change is a global crisis affecting every continent."},
)
time.sleep(0.1)
processed = "climate change is a global crisis affecting every continent."
span_preprocess.end(output={"cleaned_text": processed, "char_count": len(processed)})

generation = trace.generation(
    name="summarize",
    model="claude-sonnet-4-20250514",
    input=[{"role": "user", "content": f"Summarize in one sentence: {processed}"}],
    model_parameters={"temperature": 0.3},
)
summary = "Climate change is a worldwide emergency impacting all regions."
generation.end(
    output=summary,
    usage={"input": 42, "output": 15},
)

span_postprocess = trace.span(name="postprocessing")
final_result = {"summary": summary, "word_count": len(summary.split())}
span_postprocess.end(output=final_result)

langfuse.score(
    trace_id=trace.id,
    name="pipeline_quality",
    value=0.85,
    comment="Automated quality check passed",
)

trace.update(output=final_result)

print(f"Trace ID: {trace.id}")
print(f"Result: {final_result}")

langfuse.flush()
langfuse.shutdown()
```

**Связь с теорией:** ручное создание trace и spans (раздел 2) для контроля иерархии: Trace → Span (preprocessing) → Generation (LLM) → Span (postprocessing) → Score. `generation()` записывает usage (токены) для cost tracking (раздел 4). `score()` привязывает оценку качества к trace (раздел 6). Metadata и tags позволяют фильтровать в dashboard.

---

## Чеклист самопроверки

- [ ] Назови 4 причины, почему LLM-приложения нуждаются в трейсинге больше, чем обычные API.
- [ ] Что такое Trace, Span, Generation, Score в Langfuse? Нарисуй иерархию для assessment chain.
- [ ] Как callback handler подключается к LangChain chain? Что происходит на каждом этапе?
- [ ] Зачем prompt management в Langfuse? Опиши жизненный цикл промпта: создание → тестирование → production → откат.
- [ ] Как считается стоимость LLM-вызова? Какие стратегии оптимизации стоимости существуют?
- [ ] Что такое graceful degradation? Почему Langfuse не должен быть hard dependency?
- [ ] Какие типы scores существуют в Langfuse? Когда использовать каждый?

---

## Частые ошибки

### 1. Langfuse как hard dependency

```python
handler = LangfuseCallbackHandler(...)
result = await chain.ainvoke(data, config={"callbacks": [handler]})
```

Если ключи не заданы или Langfuse недоступен — приложение падает. Всегда оборачивайте в try/except:

```python
handler = None
if config.langfuse_enabled:
    try:
        handler = LangfuseCallbackHandler(...)
    except Exception:
        logger.warning("Langfuse unavailable, tracing disabled")

invoke_config = {"callbacks": [handler]} if handler else {}
result = await chain.ainvoke(data, config=invoke_config)
```

### 2. Безымянные traces

```python
result = chain.ainvoke(data, config={"callbacks": [handler]})
```

В Langfuse dashboard все traces называются "LangChain" — невозможно найти нужный. Всегда задавайте `trace_name` и `metadata`:

```python
handler = LangfuseCallbackHandler(
    trace_name="essay_assessment",
    metadata={"rubric": "essay_default", "user_id": "teacher-1"},
    tags=["assessment", "production"],
)
```

### 3. Забыть flush()

```python
handler = LangfuseCallbackHandler(...)
result = await chain.ainvoke(data, config={"callbacks": [handler]})
langfuse.score(trace_id=handler.get_trace_id(), name="quality", value=0.8)
```

Langfuse буферизует данные и отправляет батчами. Без `flush()` данные могут потеряться, если процесс завершится до отправки:

```python
handler.flush()
langfuse.flush()
```

При завершении скрипта или notebook kernel буферизованные данные теряются.

### 4. Трейсинг только в production

```python
if env == "production":
    callbacks = [langfuse_handler]
```

В development вы не видите промпты, токены и стоимость — отладка вслепую. Используйте разные проекты в Langfuse для разных окружений:

```
DEV:  LANGFUSE_PUBLIC_KEY=pk-lf-dev-...    (project: assessment-dev)
PROD: LANGFUSE_PUBLIC_KEY=pk-lf-prod-...   (project: assessment-prod)
```

Трейсинг включён везде, но данные изолированы. Development traces не засоряют production dashboard.

---

## Что читать дальше

- [Langfuse Docs](https://langfuse.com/docs) — полная документация
- [Langfuse + LangChain Integration](https://langfuse.com/docs/integrations/langchain/tracing) — настройка callback handler
- [Langfuse Prompt Management](https://langfuse.com/docs/prompts/get-started) — управление промптами
- [Langfuse Scores](https://langfuse.com/docs/scores/overview) — автоматические и пользовательские оценки
- [LangSmith](https://docs.smith.langchain.com/) — альтернатива от LangChain (SaaS)
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/) — система callbacks в LangChain
- **[Тема 16: Langfuse Deep Dive](topic_16_langfuse.md)** — глубокое погружение: prompt management, datasets, experiments, scores, annotation queues, cost analytics. Эта тема раскрывает Langfuse как полноценную платформу — здесь мы рассмотрели только трейсинг.
- **[Тема 17: LangSmith](topic_17_langsmith.md)** — альтернативная платформа: Hub, evaluation, datasets, мониторинг. Сравнение с Langfuse.

**Следующая тема:** [Тема 9: Evaluation](topic_09_evaluation.md) — как измерять качество LLM.

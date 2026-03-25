# Тема 4: Streaming — ответ LLM в реальном времени

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 2](topic_02_langchain_lcel.md), [Тема 3](topic_03_structured_output.md)
> **Что добавим в проект:** `app/api/v1/streaming.py` — роутер с 4 эндпоинтами
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `sse-starlette`, `pydantic`

---

## Теория

### 1. Зачем streaming: TTFT и восприятие пользователя

LLM генерирует текст **токен за токеном**. Скорость генерации зависит от модели и провайдера:

| Модель | Скорость генерации | 1000 токенов |
|---|---|---|
| Claude Sonnet | ~80-100 токенов/сек | 10-12 сек |
| Claude Haiku | ~150-200 токенов/сек | 5-7 сек |
| GPT-4o | ~80-120 токенов/сек | 8-12 сек |

Без стриминга пользователь видит пустой экран 10-20 секунд, затем весь текст появляется разом. Со стримингом первые символы появляются через 0.5-2 секунды — **Time To First Token (TTFT)**.

Исследования UX показывают:
- **< 1 секунды** — система воспринимается как мгновенная.
- **1-3 секунды** — заметная задержка, но пользователь не теряет фокус.
- **3-10 секунд** — пользователь начинает отвлекаться.
- **> 10 секунд** — пользователь уходит или перезагружает страницу.

Стриминг превращает задержку в 15 секунд (без стриминга) в TTFT ~1 секунда плюс постепенное появление текста. Пользователь начинает читать, пока модель ещё генерирует.

### 2. Архитектура стриминга в LangChain

Каждый компонент в LangChain реализует интерфейс `Runnable`, который включает методы стриминга:

```python
class Runnable:
    def stream(self, input, config=None, **kwargs) -> Iterator[T]: ...
    async def astream(self, input, config=None, **kwargs) -> AsyncIterator[T]: ...
    async def astream_events(self, input, *, version, config=None, **kwargs) -> AsyncIterator[StreamEvent]: ...
```

**Как стриминг проходит через chain:**

При вызове `chain.astream(input)` на chain `prompt | llm`:

1. `prompt.invoke(input)` выполняется **полностью** (промпт не стримится).
2. `llm.astream(messages)` начинает стримить `AIMessageChunk` по мере генерации.
3. Каждый chunk содержит `.content` — фрагмент текста (обычно 1-3 токена).

Если chain длиннее, например `prompt | llm | parser`:
- `prompt` и `parser` обрабатывают данные целиком (не стримятся).
- Стримится только `llm`.
- Для `parser` стриминг **невозможен** — ему нужен полный текст для парсинга.

**Тип chunk зависит от последнего элемента chain:**

| Последний элемент | Тип chunk | Содержимое |
|---|---|---|
| ChatModel (llm) | `AIMessageChunk` | `.content` — фрагмент текста |
| StrOutputParser | `str` | Строка-фрагмент |
| PydanticOutputParser | Полный объект | Стриминг невозможен, один chunk |
| `with_structured_output` | Полный объект | Стриминг невозможен, один chunk |

### 3. stream() и astream() — базовый стриминг

```python
async for chunk in chain.astream({"student_work": "...", "rubric": "..."}):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

Каждый `AIMessageChunk` — это инкрементальное обновление. Свойства:

- `chunk.content` — строка с фрагментом текста (может быть пустой для служебных chunk-ов).
- `chunk.id` — идентификатор сообщения (одинаковый для всех chunk-ов одного ответа).
- `chunk.response_metadata` — метаданные (непустые обычно только в первом и последнем chunk-е).
- `chunk.usage_metadata` — информация о токенах (обычно только в последнем chunk-е).

Chunk-и можно складывать оператором `+`:

```python
full_message = None
async for chunk in llm.astream(messages):
    if full_message is None:
        full_message = chunk
    else:
        full_message = full_message + chunk
print(full_message.content)
```

Результат `full_message` — полный `AIMessageChunk`, эквивалентный результату `ainvoke()`.

### 4. astream_events() — детальные события chain

`astream_events()` выдаёт **типизированные события** для каждого этапа выполнения chain. Это основной инструмент для мониторинга, отладки и создания сложных UX с прогрессом.

```python
async for event in chain.astream_events(
    {"student_work": "..."},
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="")
```

**Основные типы событий:**

| Событие | Когда срабатывает | Данные (`event["data"]`) |
|---|---|---|
| `on_chain_start` | Начало chain | `{"input": {...}}` |
| `on_chat_model_start` | LLM начал генерацию | `{"input": {"messages": [...]}}` |
| `on_chat_model_stream` | Каждый токен от LLM | `{"chunk": AIMessageChunk}` |
| `on_chat_model_end` | LLM закончил | `{"output": AIMessage}` |
| `on_chain_end` | Chain завершён | `{"output": ...}` |
| `on_prompt_start` | Промпт начал обработку | `{"input": {...}}` |
| `on_prompt_end` | Промпт сформирован | `{"output": ChatPromptValue}` |

**Важно: version="v2"**

Всегда указывайте `version="v2"`. Версия v1 устарела и имеет отличия в формате данных. В v2:
- Консистентные имена событий.
- Данные всегда в `event["data"]`.
- `event["name"]` содержит имя runnable.

**Фильтрация событий:**

```python
async for event in chain.astream_events(
    input_data,
    version="v2",
    include_types=["chat_model"],
):
    pass
```

Доступные фильтры: `include_names`, `include_types`, `include_tags`, `exclude_names`, `exclude_types`, `exclude_tags`. Фильтрация снижает шум и улучшает производительность.

**Именование runnable для фильтрации:**

```python
named_llm = llm.with_config({"run_name": "assessor_llm"})
chain = prompt | named_llm

async for event in chain.astream_events(
    input_data,
    version="v2",
    include_names=["assessor_llm"],
):
    pass
```

### 5. SSE (Server-Sent Events) — протокол доставки

SSE — стандартизированный HTTP-протокол для **однонаправленной** потоковой передачи от сервера к клиенту. Работает поверх обычного HTTP, не требует специального протокола.

**Wire-формат SSE:**

```
event: start
data: {"message": "Assessment started"}

event: token
data: {"content": "The"}

event: token
data: {"content": " essay"}

event: end
data: {"tokens": 342, "elapsed": 4.2}

```

Каждое сообщение:
- `event:` — тип события (опционально, default — `message`).
- `data:` — полезная нагрузка (может быть многострочной, каждая строка с `data:`).
- `id:` — идентификатор для reconnect (опционально).
- `retry:` — время reconnect в мс (опционально).
- Пустая строка — разделитель между сообщениями.

**Сравнение технологий потоковой передачи:**

| Характеристика | SSE | WebSocket | HTTP/2 Stream |
|---|---|---|---|
| Направление | Сервер → клиент | Двунаправленное | Двунаправленное |
| Протокол | HTTP/1.1+ | WS (upgrade) | HTTP/2 |
| Reconnect | Автоматический | Ручная реализация | Ручная реализация |
| Типы сообщений | text/event-stream | Бинарные / текстовые | Бинарные / текстовые |
| Сложность | Минимальная | Средняя | Высокая |
| Прокси/CDN | Совместимо | Проблемы с некоторыми | Совместимо |
| Идеально для | LLM streaming, уведомления | Чат, игры, коллаборация | gRPC, мультиплексинг |

SSE — оптимальный выбор для LLM streaming: однонаправленная передача токенов, автоматический reconnect, работает через любые прокси.

**Клиентская сторона:**

```javascript
const source = new EventSource("/api/v1/streaming/events", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({student_work: "...", rubric_id: "essay_default"})
});

source.addEventListener("token", (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("output").textContent += data.content;
});

source.addEventListener("end", (e) => {
    const stats = JSON.parse(e.data);
    console.log(`Done: ${stats.tokens} tokens in ${stats.elapsed}s`);
    source.close();
});
```

Стандартный `EventSource` API поддерживает только GET. Для POST используйте библиотеки: `eventsource-parser`, `fetch-event-source`, или ручной парсинг `fetch()` с `ReadableStream`.

### 6. EventSourceResponse в FastAPI

`sse-starlette` предоставляет `EventSourceResponse` — ASGI-совместимый response для SSE:

```python
from sse_starlette.sse import EventSourceResponse

@router.post("/stream")
async def stream(request: Request):
    async def generate():
        yield {"event": "start", "data": "{}"}
        async for token in get_tokens():
            yield {"event": "token", "data": json.dumps({"content": token})}
        yield {"event": "end", "data": "{}"}

    return EventSourceResponse(generate())
```

**Формат yield:**

- `yield {"data": "text"}` — отправит `data: text\n\n`
- `yield {"event": "token", "data": "text"}` — отправит `event: token\ndata: text\n\n`
- `yield {"data": json.dumps(obj)}` — JSON в data

**Обработка отключения клиента:**

Если клиент закрывает соединение, `EventSourceResponse` автоматически отменяет генератор. Но если внутри генератора длительная операция (например, вызов LLM), отмена произойдёт только при следующем `yield`. Для корректной обработки:

```python
from starlette.requests import Request

@router.post("/stream")
async def stream(request: Request):
    async def generate():
        async for chunk in chain.astream(data):
            if await request.is_disconnected():
                break
            if chunk.content:
                yield {"data": chunk.content}

    return EventSourceResponse(generate())
```

### 7. Structured Output и Streaming — несовместимость и обходные пути

`with_structured_output()` использует tool calling API: модель генерирует JSON как аргумент tool call, который парсится целиком. Невозможно распарсить половину JSON в Pydantic-объект — `{"overall_score": 85, "summary": "The ess` — это невалидный JSON.

**Production-стратегия — два эндпоинта:**

| Эндпоинт | Метод | Для кого | Ответ |
|---|---|---|---|
| `POST /assess` | `ainvoke()` | API-to-API | `AssessmentResponse` (JSON) |
| `POST /assess/stream` | `astream()` | UI / фронтенд | Поток токенов (SSE) |

Стриминговый endpoint использует **raw LLM** (без structured output) и передаёт текст токен за токеном. В проекте это реализовано: `assess()` возвращает структурированный ответ, `assess_stream()` стримит текст.

**Экспериментальный вариант — partial JSON parsing:**

Некоторые библиотеки (`partial-json-parser`, `ijson`) позволяют парсить JSON инкрементально. Можно стримить JSON и парсить поля по мере их появления:

```python
import json

buffer = ""
async for chunk in llm.astream(messages):
    buffer += chunk.content
    try:
        partial = json.loads(buffer + "}")
        if "overall_score" in partial:
            yield {"event": "score", "data": str(partial["overall_score"])}
    except json.JSONDecodeError:
        pass
```

Это хрупкий подход и не рекомендуется для production, но полезен для прототипов.

### 8. Callback Handlers — наблюдение за chain

Callback handlers реализуют **Observer pattern**: они получают уведомления о событиях в chain, не влияя на его выполнение. Два базовых класса:

| Класс | Для чего | Методы |
|---|---|---|
| `BaseCallbackHandler` | Синхронные callbacks | `on_llm_start`, `on_llm_end`, ... |
| `AsyncCallbackHandler` | Асинхронные callbacks | `async on_llm_start`, `async on_llm_end`, ... |

Для FastAPI (async) всегда используйте `AsyncCallbackHandler`.

```python
from langchain_core.callbacks import AsyncCallbackHandler

class TokenCounter(AsyncCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    async def on_llm_end(self, response, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
```

**Передача callbacks:**

```python
counter = TokenCounter()
result = await chain.ainvoke(data, config={"callbacks": [counter]})
print(counter.input_tokens, counter.output_tokens)
```

Callbacks передаются через `config`, не как аргумент метода. Это позволяет добавлять несколько наблюдателей без изменения кода chain.

**Жизненный цикл событий при стриминге:**

```
on_chain_start          → chain начал выполнение
  on_prompt_start       → промпт начал формироваться
  on_prompt_end         → промпт готов
  on_chat_model_start   → LLM начал генерацию
    on_llm_new_token    → каждый новый токен (×N)
  on_chat_model_end     → LLM закончил, usage доступен
on_chain_end            → chain завершён
```

`on_llm_end` получает `LLMResult` с `llm_output` — здесь содержится usage metadata от провайдера. Для Anthropic: `{"usage": {"input_tokens": N, "output_tokens": M}}`.

### 9. Backpressure и память при стриминге

Если клиент читает медленнее, чем LLM генерирует, буфер `EventSourceResponse` растёт. В production это обычно не проблема (LLM генерирует ~100 токенов/сек, каждый — несколько байт), но стоит учитывать:

- **Ping-интервал** — `EventSourceResponse` может отправлять пустые SSE-сообщения для поддержания соединения:

```python
EventSourceResponse(generate(), ping=15)
```

- **Таймауты прокси** — nginx по умолчанию закрывает SSE через 60 секунд. Для длинных оценок увеличьте `proxy_read_timeout`.

- **Память при множественных соединениях** — каждый SSE-стрим держит открытое соединение. 100 одновременных стримов — 100 открытых HTTP-соединений. Для высокой нагрузки рассмотрите Redis pub/sub или message broker.

---

## Справочник API

### .stream() / .astream()

**Описание:** Методы интерфейса `Runnable` для потоковой генерации. `.stream()` — синхронный, `.astream()` — асинхронный. Возвращают итератор chunk-ов, тип которых зависит от конкретного Runnable.

```python
def stream(
    input: Input,                           # входные данные (dict, str, list[BaseMessage])
    config: RunnableConfig | None = None,   # конфигурация (callbacks, tags, metadata)
    **kwargs,                               # доп. параметры для конкретного Runnable
) -> Iterator[Output]

async def astream(
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs,
) -> AsyncIterator[Output]
```

**Основные параметры RunnableConfig:**

| Параметр | Тип | Описание |
|---|---|---|
| `callbacks` | `list[BaseCallbackHandler]` | Callback handlers для мониторинга |
| `tags` | `list[str]` | Теги для фильтрации и трейсинга |
| `metadata` | `dict` | Произвольные метаданные |
| `run_name` | `str` | Имя выполнения для логирования |
| `max_concurrency` | `int` | Макс. параллельных операций |

**Пример использования:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
chain = prompt | llm

full_text = ""
async for chunk in chain.astream(
    {"question": "Explain quantum computing"},
    config={"tags": ["experiment"]},
):
    if chunk.content:
        full_text += chunk.content
        print(chunk.content, end="", flush=True)
```

---

### .astream_events()

**Описание:** Асинхронный метод `Runnable`, выдающий детальные типизированные события для каждого этапа выполнения chain. Позволяет отслеживать прогресс, логировать шаги и собирать метрики без изменения кода chain.

```python
async def astream_events(
    input: Input,
    *,
    version: str,                                    # "v2" (обязательный)
    config: RunnableConfig | None = None,
    include_names: list[str] | None = None,          # фильтр по имени runnable
    include_types: list[str] | None = None,          # фильтр по типу ("chat_model", "chain", "prompt")
    include_tags: list[str] | None = None,           # фильтр по тегам
    exclude_names: list[str] | None = None,          # исключить по имени
    exclude_types: list[str] | None = None,          # исключить по типу
    exclude_tags: list[str] | None = None,           # исключить по тегам
    **kwargs,
) -> AsyncIterator[StreamEvent]
```

**Структура StreamEvent:**

| Поле | Тип | Описание |
|---|---|---|
| `event` | `str` | Тип события (`on_chain_start`, `on_chat_model_stream`, ...) |
| `name` | `str` | Имя runnable, сгенерировавшего событие |
| `data` | `dict` | Данные события (`input`, `output`, `chunk`) |
| `run_id` | `str` | UUID выполнения |
| `parent_ids` | `list[str]` | UUID родительских выполнений |
| `tags` | `list[str]` | Теги runnable |
| `metadata` | `dict` | Метаданные |

**Пример использования:**

```python
token_count = 0

async for event in chain.astream_events(
    {"student_work": essay_text, "rubric": rubric_text},
    version="v2",
    include_types=["chat_model"],
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            token_count += 1
            print(chunk.content, end="")
    elif event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        print(f"\nTotal chunks: {token_count}")
```

---

### EventSourceResponse (sse-starlette)

**Описание:** ASGI-совместимый HTTP response для Server-Sent Events. Принимает асинхронный генератор и отправляет события клиенту в формате SSE. Автоматически управляет соединением, поддерживает ping и reconnect.

```python
EventSourceResponse(
    content: AsyncIterator[dict | str],     # генератор событий
    status_code: int = 200,                 # HTTP status code
    headers: dict | None = None,            # доп. HTTP-заголовки
    media_type: str = "text/event-stream",  # MIME-тип (не менять)
    background: BackgroundTask | None = None,  # фоновая задача после завершения
    ping: int = 0,                          # интервал ping в секундах (0 — отключён)
    sep: str | None = None,                 # разделитель событий
    ping_message_factory: Callable | None = None,  # фабрика ping-сообщений
)
```

**Формат генерируемых событий:**

| yield | SSE-вывод |
|---|---|
| `{"data": "hello"}` | `data: hello\n\n` |
| `{"event": "token", "data": "hi"}` | `event: token\ndata: hi\n\n` |
| `{"event": "end", "data": "{}", "id": "1"}` | `event: end\nid: 1\ndata: {}\n\n` |

**Пример использования:**

```python
import json
from sse_starlette.sse import EventSourceResponse
from fastapi import APIRouter

router = APIRouter()

@router.post("/stream")
async def stream_endpoint():
    async def generate():
        yield {"event": "start", "data": json.dumps({"status": "ok"})}
        for i in range(10):
            yield {"event": "token", "data": json.dumps({"index": i, "content": f"word_{i}"})}
        yield {"event": "end", "data": json.dumps({"total": 10})}

    return EventSourceResponse(generate(), ping=15)
```

---

### BaseCallbackHandler / AsyncCallbackHandler

**Описание:** Базовые классы для реализации callbacks, наблюдающих за выполнением LangChain chain. `BaseCallbackHandler` — синхронный, `AsyncCallbackHandler` — асинхронный (для FastAPI). Реализуют Observer pattern — не влияют на выполнение, только наблюдают.

```python
from langchain_core.callbacks import AsyncCallbackHandler

class MyCallback(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs): ...
    async def on_llm_new_token(self, token, **kwargs): ...
    async def on_llm_end(self, response, **kwargs): ...
    async def on_llm_error(self, error, **kwargs): ...
    async def on_chain_start(self, serialized, inputs, **kwargs): ...
    async def on_chain_end(self, outputs, **kwargs): ...
    async def on_chain_error(self, error, **kwargs): ...
    async def on_tool_start(self, serialized, input_str, **kwargs): ...
    async def on_tool_end(self, output, **kwargs): ...
    async def on_tool_error(self, error, **kwargs): ...
```

**Основные методы:**

| Метод | Аргументы | Когда вызывается |
|---|---|---|
| `on_llm_start` | `serialized`, `prompts` | LLM начал генерацию |
| `on_llm_new_token` | `token: str` | Каждый новый токен при стриминге |
| `on_llm_end` | `response: LLMResult` | LLM закончил, usage доступен |
| `on_llm_error` | `error: Exception` | Ошибка при вызове LLM |
| `on_chain_start` | `serialized`, `inputs: dict` | Chain начал выполнение |
| `on_chain_end` | `outputs: dict` | Chain завершён |

**Пример использования:**

```python
from langchain_core.callbacks import AsyncCallbackHandler


class TimingCallback(AsyncCallbackHandler):
    def __init__(self):
        import time
        self.start_time = 0
        self.first_token_time = 0
        self.end_time = 0

    async def on_llm_start(self, serialized, prompts, **kwargs):
        import time
        self.start_time = time.monotonic()

    async def on_llm_new_token(self, token, **kwargs):
        import time
        if self.first_token_time == 0:
            self.first_token_time = time.monotonic()

    async def on_llm_end(self, response, **kwargs):
        import time
        self.end_time = time.monotonic()

    @property
    def ttft(self):
        return self.first_token_time - self.start_time

    @property
    def total_time(self):
        return self.end_time - self.start_time
```

---

### AIMessageChunk

**Описание:** Инкрементальный фрагмент ответа ChatModel при стриминге. Содержит часть текста (обычно 1-3 токена). Chunk-и можно складывать оператором `+` для получения полного сообщения. Последний chunk обычно содержит `usage_metadata`.

```python
class AIMessageChunk:
    content: str | list                     # текст фрагмента (или list content blocks)
    id: str | None                          # ID сообщения (одинаковый для всех chunk-ов)
    response_metadata: dict                 # метаданные ответа (модель, stop reason)
    usage_metadata: dict | None             # токены (обычно только в последнем chunk-е)
    tool_calls: list[ToolCall]              # tool calls (для structured output)
    tool_call_chunks: list[ToolCallChunk]   # инкрементальные tool call фрагменты
```

**Основные операции:**

| Операция | Описание |
|---|---|
| `chunk.content` | Текст фрагмента |
| `chunk + other_chunk` | Объединение двух chunk-ов |
| `chunk.usage_metadata` | `{"input_tokens": N, "output_tokens": M}` или None |
| `chunk.response_metadata` | `{"model": "...", "stop_reason": "..."}` |

**Пример использования:**

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
messages = [("human", "Write a haiku about programming")]

accumulated = None
async for chunk in llm.astream(messages):
    if chunk.content:
        print(chunk.content, end="")
    if accumulated is None:
        accumulated = chunk
    else:
        accumulated = accumulated + chunk

print(f"\nFull content: {accumulated.content}")
print(f"Usage: {accumulated.usage_metadata}")
```

---

## Практика: роутер `/api/v1/streaming`

### Шаг 1. Схемы ответов

Для SSE-эндпоинтов основные ответы отправляются как события. Но нам нужны Pydantic-модели для типизации данных внутри событий и для обычных JSON-эндпоинтов.

```python
from pydantic import BaseModel, Field


class StreamingRequest(BaseModel):
    student_work: str
    rubric_id: str | None = "essay_default"


class ModelPricing(BaseModel):
    input_per_million: float = Field(description="USD per 1M input tokens")
    output_per_million: float = Field(description="USD per 1M output tokens")


class PricingResponse(BaseModel):
    current_model: str
    pricing: ModelPricing
    all_models: dict[str, ModelPricing]
```

### Шаг 2. Создание роутера `app/api/v1/streaming.py`

```python
import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from app.dependencies import LLMDep, RubricStoreDep, SettingsDep
from app.schemas.assessment import AssessmentRequest
from app.schemas.rubric import Rubric
from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_GOOD_EXAMPLE,
    FEW_SHOT_BAD_EXAMPLE,
)

router = APIRouter(prefix="/streaming", tags=["lesson-4-streaming"])

MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
}

DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


class ModelPricing(BaseModel):
    input_per_million: float = Field(description="USD per 1M input tokens")
    output_per_million: float = Field(description="USD per 1M output tokens")


class PricingResponse(BaseModel):
    current_model: str
    pricing: ModelPricing
    all_models: dict[str, ModelPricing]


class UsageTracker(AsyncCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    async def on_llm_end(self, response, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
            if usage:
                self.input_tokens = usage.get("input_tokens", 0)
                self.output_tokens = usage.get("output_tokens", 0)
                return
        for generation_list in response.generations:
            for gen in generation_list:
                msg = getattr(gen, "message", None)
                if msg:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:
                        self.input_tokens = meta.get("input_tokens", 0)
                        self.output_tokens = meta.get("output_tokens", 0)
                        return


def _resolve_rubric(request: AssessmentRequest, rubrics: dict[str, Rubric]) -> Rubric:
    if request.rubric:
        return request.rubric
    rubric_id = request.rubric_id or "essay_default"
    if rubric_id not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rubric_id}' not found")
    return rubrics[rubric_id]


def _format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", ASSESSMENT_SYSTEM_PROMPT),
        ("human", "Please assess the following student work:\n\n{student_work}"),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )


@router.post("/events")
async def stream_events(
    request: AssessmentRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> EventSourceResponse:
    rubric = _resolve_rubric(request, rubrics)
    rubric_text = _format_rubric(rubric)
    prompt = _build_prompt()
    chain = prompt | llm

    async def event_generator():
        token_count = 0
        start_time = time.monotonic()

        yield {"event": "start", "data": json.dumps({"message": "Assessment started"})}

        async for event in chain.astream_events(
            {"student_work": request.student_work, "rubric": rubric_text},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    token_count += 1
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": chunk.content}),
                    }

        elapsed = round(time.monotonic() - start_time, 2)
        yield {
            "event": "end",
            "data": json.dumps({"tokens": token_count, "elapsed": elapsed}),
        }

    return EventSourceResponse(event_generator())


@router.post("/progress")
async def stream_with_progress(
    request: AssessmentRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> EventSourceResponse:
    rubric = _resolve_rubric(request, rubrics)
    rubric_text = _format_rubric(rubric)
    criterion_names = [c.name.lower() for c in rubric.criteria]
    total_criteria = len(criterion_names)
    prompt = _build_prompt()
    chain = prompt | llm

    async def event_generator():
        buffer = ""
        detected: set[str] = set()

        yield {
            "event": "progress",
            "data": json.dumps({
                "stage": "analyzing",
                "current": 0,
                "total": total_criteria,
                "message": "Analyzing student work...",
            }),
        }

        async for chunk in chain.astream(
            {"student_work": request.student_work, "rubric": rubric_text},
        ):
            if not chunk.content:
                continue

            yield {"event": "token", "data": json.dumps({"content": chunk.content})}

            buffer += chunk.content.lower()
            for name in criterion_names:
                if name not in detected and name in buffer:
                    detected.add(name)
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "stage": "scoring",
                            "current": len(detected),
                            "total": total_criteria,
                            "criterion": name,
                            "message": f"Evaluating criterion {len(detected)}/{total_criteria}: {name}",
                        }),
                    }

        yield {
            "event": "progress",
            "data": json.dumps({
                "stage": "complete",
                "current": total_criteria,
                "total": total_criteria,
                "message": "Assessment complete",
            }),
        }

    return EventSourceResponse(event_generator())


@router.post("/with-usage")
async def stream_with_usage(
    request: AssessmentRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
    settings: SettingsDep,
) -> EventSourceResponse:
    rubric = _resolve_rubric(request, rubrics)
    rubric_text = _format_rubric(rubric)
    prompt = _build_prompt()
    chain = prompt | llm
    tracker = UsageTracker()

    async def event_generator():
        start_time = time.monotonic()

        async for chunk in chain.astream(
            {"student_work": request.student_work, "rubric": rubric_text},
            config={"callbacks": [tracker]},
        ):
            if chunk.content:
                yield {"event": "token", "data": json.dumps({"content": chunk.content})}

        elapsed = round(time.monotonic() - start_time, 2)
        pricing = MODEL_PRICING.get(settings.model_name, DEFAULT_PRICING)
        input_cost = tracker.input_tokens * pricing["input"] / 1_000_000
        output_cost = tracker.output_tokens * pricing["output"] / 1_000_000

        yield {
            "event": "usage",
            "data": json.dumps({
                "input_tokens": tracker.input_tokens,
                "output_tokens": tracker.output_tokens,
                "total_tokens": tracker.input_tokens + tracker.output_tokens,
                "cost_usd": round(input_cost + output_cost, 6),
                "elapsed_seconds": elapsed,
                "model": settings.model_name,
            }),
        }

    return EventSourceResponse(event_generator())


@router.get("/pricing")
async def get_pricing(settings: SettingsDep) -> PricingResponse:
    current_pricing = MODEL_PRICING.get(settings.model_name, DEFAULT_PRICING)
    return PricingResponse(
        current_model=settings.model_name,
        pricing=ModelPricing(
            input_per_million=current_pricing["input"],
            output_per_million=current_pricing["output"],
        ),
        all_models={
            name: ModelPricing(
                input_per_million=p["input"],
                output_per_million=p["output"],
            )
            for name, p in MODEL_PRICING.items()
        },
    )
```

**Как каждый эндпоинт связан с теорией:**

| Эндпоинт | Концепция из теории | Что демонстрирует |
|---|---|---|
| `POST /events` | §4 astream_events, §5 SSE | Типизированные SSE-события с фильтрацией по типу |
| `POST /progress` | §4 фильтрация, §3 astream | Эвристическое определение прогресса по содержимому стрима |
| `POST /with-usage` | §8 Callback handlers | Подсчёт токенов и стоимости через AsyncCallbackHandler |
| `GET /pricing` | §9 стоимость | Справочный endpoint для расчёта стоимости |

### Шаг 3. Регистрация в `app/api/router.py`

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, streaming

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(streaming.router)
```

### Шаг 4. Тестирование с curl

Запустите сервер:

```bash
uvicorn app.main:app --reload
```

**POST /events** — типизированные SSE-события:

```bash
curl -N -X POST http://localhost:8000/api/v1/streaming/events \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is a major threat. Rising temperatures cause ice to melt, leading to higher sea levels. Governments should implement carbon taxes and invest in renewable energy. Without action, future generations will suffer.",
    "rubric_id": "essay_default"
  }'
```

Ожидаемый вывод — поток SSE-событий:

```
event: start
data: {"message": "Assessment started"}

event: token
data: {"content": "The"}

event: token
data: {"content": " essay"}

...

event: end
data: {"tokens": 342, "elapsed": 4.21}
```

Флаг `-N` отключает буферизацию curl — без него события будут приходить пачками.

**POST /progress** — прогресс по критериям:

```bash
curl -N -X POST http://localhost:8000/api/v1/streaming/progress \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is a major threat. Rising temperatures cause ice to melt, leading to higher sea levels. Governments should implement carbon taxes and invest in renewable energy. Without action, future generations will suffer.",
    "rubric_id": "essay_default"
  }'
```

Между событиями `token` будут появляться события `progress` при обнаружении имён критериев из рубрики в стриме.

**POST /with-usage** — стриминг с подсчётом токенов:

```bash
curl -N -X POST http://localhost:8000/api/v1/streaming/with-usage \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is a major threat. Rising temperatures cause ice to melt.",
    "rubric_id": "essay_default"
  }'
```

Последнее событие — `usage` с информацией о токенах и стоимости:

```
event: usage
data: {"input_tokens": 1250, "output_tokens": 340, "total_tokens": 1590, "cost_usd": 0.008850, "elapsed_seconds": 3.41, "model": "claude-sonnet-4-20250514"}
```

**GET /pricing** — информация о ценах:

```bash
curl -s http://localhost:8000/api/v1/streaming/pricing | python -m json.tool
```

```json
{
  "current_model": "claude-sonnet-4-20250514",
  "pricing": {
    "input_per_million": 3.0,
    "output_per_million": 15.0
  },
  "all_models": {
    "claude-sonnet-4-20250514": {"input_per_million": 3.0, "output_per_million": 15.0},
    "claude-haiku-3-5-20241022": {"input_per_million": 0.8, "output_per_million": 4.0}
  }
}
```

---

## Чеклист самопроверки

- [ ] Почему structured output и streaming несовместимы? Какая production-стратегия решает это?
- [ ] В чём разница между `astream()` и `astream_events()`? Когда какой использовать?
- [ ] Объясни SSE: протокол, формат сообщения, отличие от WebSocket.
- [ ] Что такое TTFT и почему это ключевая UX-метрика для LLM-приложений?
- [ ] Зачем callback handlers, если можно логировать напрямую в chain?
- [ ] Как `AIMessageChunk` объединяются в полное сообщение?
- [ ] Почему нужен `version="v2"` в `astream_events()`?
- [ ] Как отследить потребление токенов при стриминге через callback?
- [ ] Что произойдёт, если клиент закроет SSE-соединение во время генерации?
- [ ] Как обнаружить прогресс оценки (смену критерия) в текстовом стриме?

---

## Частые ошибки

### 1. Стриминг structured output

```python
async for chunk in structured_chain.astream(data):
    print(chunk)
```

```python
async for chunk in (prompt | llm).astream(data):
    if chunk.content:
        print(chunk.content, end="")
```

`with_structured_output` собирает полный ответ — `astream` отдаст один chunk. Для стриминга используйте raw LLM без structured output.

### 2. Пустые chunk-и в SSE

```python
async for chunk in chain.astream(data):
    yield {"data": chunk.content}
```

```python
async for chunk in chain.astream(data):
    if chunk.content:
        yield {"data": chunk.content}
```

Некоторые chunk-и приходят пустыми (служебные). Без фильтрации клиент получает пустые SSE-события.

### 3. Блокирующий код в async генераторе

```python
async def generate():
    time.sleep(1)
    async for chunk in chain.astream(data):
        yield {"data": chunk.content}
```

```python
async def generate():
    await asyncio.sleep(1)
    async for chunk in chain.astream(data):
        if chunk.content:
            yield {"data": chunk.content}
```

`time.sleep()` блокирует event loop и все другие async-операции. Используйте `asyncio.sleep()`.

### 4. BaseCallbackHandler вместо AsyncCallbackHandler

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        self.tokens = response.llm_output
```

```python
from langchain_core.callbacks import AsyncCallbackHandler

class MyCallback(AsyncCallbackHandler):
    async def on_llm_end(self, response, **kwargs):
        self.tokens = response.llm_output
```

В async-контексте (FastAPI) синхронные callbacks блокируют event loop. Всегда используйте `AsyncCallbackHandler`.

### 5. Отсутствие json.dumps в SSE data

```python
yield {"event": "token", "data": {"content": chunk.content}}
```

```python
yield {"event": "token", "data": json.dumps({"content": chunk.content})}
```

SSE data — строка. Если передать dict, `EventSourceResponse` вызовет `str()`, и клиент получит Python repr вместо JSON.

---

## Что читать дальше

- [LangChain Streaming](https://python.langchain.com/docs/concepts/streaming/) — концепция
- [How to stream responses](https://python.langchain.com/docs/how_to/streaming/) — практика
- [astream_events](https://python.langchain.com/docs/how_to/streaming/#using-stream-events) — продвинутый стриминг
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/) — система callback-ов
- [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html) — спецификация протокола
- [sse-starlette](https://github.com/sysid/sse-starlette) — SSE для FastAPI/Starlette
- [Anthropic Streaming](https://docs.anthropic.com/en/api/streaming) — стриминг в API Anthropic

**Следующая тема:** [Тема 5: RAG](topic_05_rag.md) — как давать LLM контекст из документов.

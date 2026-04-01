# Тема 4: Streaming — ответ LLM в реальном времени

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 2](topic_02_langchain_lcel.md), [Тема 3](topic_03_structured_output.md)
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `pydantic`

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

### 6. SSE (Server-Sent Events) — отправка потока клиенту

SSE — это протокол, в котором сервер отправляет поток событий по HTTP. Каждое событие — текстовый блок с полями `event`, `data`, `id`. Библиотека `sse-starlette` предоставляет `EventSourceResponse` для ASGI-приложений:

```python
from sse_starlette.sse import EventSourceResponse

async def generate():
    yield {"event": "start", "data": "{}"}
    async for token in get_tokens():
        yield {"event": "token", "data": json.dumps({"content": token})}
    yield {"event": "end", "data": "{}"}

response = EventSourceResponse(generate())
```

**Формат yield:**

- `yield {"data": "text"}` — отправит `data: text\n\n`
- `yield {"event": "token", "data": "text"}` — отправит `event: token\ndata: text\n\n`
- `yield {"data": json.dumps(obj)}` — JSON в data

**Обработка отключения клиента:**

Если клиент закрывает соединение, `EventSourceResponse` автоматически отменяет генератор. Но если внутри генератора длительная операция (например, вызов LLM), отмена произойдёт только при следующем `yield`. Генератор должен периодически проверять состояние соединения:

```python
async def generate(is_disconnected):
    async for chunk in chain.astream(data):
        if is_disconnected():
            break
        if chunk.content:
            yield {"data": chunk.content}
```

### 7. Structured Output и Streaming — несовместимость и обходные пути

`with_structured_output()` использует tool calling API: модель генерирует JSON как аргумент tool call, который парсится целиком. Невозможно распарсить половину JSON в Pydantic-объект — `{"overall_score": 85, "summary": "The ess` — это невалидный JSON.

**Production-стратегия — два режима:**

| Режим | Метод | Для кого | Ответ |
|---|---|---|---|
| Structured | `ainvoke()` | API-to-API | Pydantic-объект (JSON) |
| Streaming | `astream()` | UI / фронтенд | Поток токенов |

Structured-режим использует `with_structured_output()` и возвращает полный объект. Streaming-режим использует **raw LLM** (без structured output) и передаёт текст токен за токеном.

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

async def generate():
    yield {"event": "start", "data": json.dumps({"status": "ok"})}
    for i in range(10):
        yield {"event": "token", "data": json.dumps({"index": i, "content": f"word_{i}"})}
    yield {"event": "end", "data": json.dumps({"total": 10})}

response = EventSourceResponse(generate(), ping=15)
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

## Практика

### Пример 1: Базовый стриминг с astream()

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor. Evaluate the text and provide detailed feedback."),
    ("human", "Assess this student work:\n\n{student_work}"),
])
chain = prompt | llm


async def main():
    full_message = None
    async for chunk in chain.astream(
        {"student_work": "Climate change is a major threat. Rising temperatures cause ice to melt."},
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        full_message = chunk if full_message is None else full_message + chunk

    print(f"\n\nFull content length: {len(full_message.content)}")
    print(f"Usage: {full_message.usage_metadata}")


asyncio.run(main())
```

Каждый `AIMessageChunk` содержит фрагмент текста. Оператор `+` объединяет chunk-и — в конце `full_message` эквивалентен результату `ainvoke()`. Поле `usage_metadata` доступно после объединения всех chunk-ов.

### Пример 2: astream_events() — детальные события chain

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-20250514").with_config({"run_name": "assessor_llm"})
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor."),
    ("human", "Assess:\n\n{student_work}"),
])
chain = prompt | llm


async def main():
    token_count = 0
    async for event in chain.astream_events(
        {"student_work": "Climate change is a major threat."},
        version="v2",
        include_names=["assessor_llm"],
    ):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print(">>> LLM started generation")
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                token_count += 1
                print(chunk.content, end="", flush=True)
        elif kind == "on_chat_model_end":
            print(f"\n>>> LLM finished. Chunks received: {token_count}")


asyncio.run(main())
```

`astream_events()` с `version="v2"` выдаёт типизированные события для каждого этапа chain. Фильтр `include_names` ограничивает поток событиями от конкретного runnable — полезно для chain с несколькими LLM-вызовами.

### Пример 3: Подсчёт токенов и стоимости через callback

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
}


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
                if msg and getattr(msg, "usage_metadata", None):
                    self.input_tokens = msg.usage_metadata.get("input_tokens", 0)
                    self.output_tokens = msg.usage_metadata.get("output_tokens", 0)
                    return


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor."),
    ("human", "Assess:\n\n{student_work}"),
])
chain = prompt | llm


async def main():
    tracker = UsageTracker()
    async for chunk in chain.astream(
        {"student_work": "Climate change is a major threat. Rising temperatures cause ice to melt."},
        config={"callbacks": [tracker]},
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    model_name = "claude-sonnet-4-20250514"
    pricing = MODEL_PRICING[model_name]
    input_cost = tracker.input_tokens * pricing["input"] / 1_000_000
    output_cost = tracker.output_tokens * pricing["output"] / 1_000_000

    print(f"\n\nInput tokens:  {tracker.input_tokens}")
    print(f"Output tokens: {tracker.output_tokens}")
    print(f"Cost (USD):    {input_cost + output_cost:.6f}")


asyncio.run(main())
```

`UsageTracker` реализует `AsyncCallbackHandler` и перехватывает `on_llm_end` — в этот момент провайдер возвращает метаданные о потреблении токенов. Callback передаётся через `config`, не влияя на логику chain.

### Пример 4: Стриминг с разными типами chain

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])


class MathAnswer(BaseModel):
    answer: int
    explanation: str


async def main():
    print("=== prompt | llm → AIMessageChunk ===")
    chain_msg = prompt | llm
    async for chunk in chain_msg.astream({"question": "What is 2+2?"}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

    print("=== prompt | llm | StrOutputParser → str ===")
    chain_str = prompt | llm | StrOutputParser()
    async for chunk in chain_str.astream({"question": "What is 2+2?"}):
        print(chunk, end="", flush=True)
    print("\n")

    print("=== with_structured_output → single chunk ===")
    structured_llm = llm.with_structured_output(MathAnswer)
    chain_structured = prompt | structured_llm
    async for chunk in chain_structured.astream({"question": "What is 2+2?"}):
        print(f"Chunk: {chunk}")


asyncio.run(main())
```

Тип chunk зависит от последнего элемента chain. `StrOutputParser` пропускает строки, `with_structured_output` собирает полный JSON — стриминг отдаёт один chunk.

**Связь примеров с теорией:**

| Пример | Концепция из теории | Что демонстрирует |
|---|---|---|
| Пример 1 | §3 astream(), AIMessageChunk | Базовый стриминг, объединение chunk-ов |
| Пример 2 | §4 astream_events(), фильтрация | Типизированные события, `include_names` |
| Пример 3 | §8 Callback handlers | Подсчёт токенов и стоимости через AsyncCallbackHandler |
| Пример 4 | §7 Structured Output + Streaming | Разница поведения стриминга для разных типов chain |

---

## Чеклист самопроверки

- [ ] Почему structured output и streaming несовместимы? Какая production-стратегия решает это?
- [ ] В чём разница между `astream()` и `astream_events()`? Когда какой использовать?
- [ ] Что такое TTFT и почему это ключевая UX-метрика для LLM-приложений?
- [ ] Зачем callback handlers, если можно логировать напрямую в chain?
- [ ] Как `AIMessageChunk` объединяются в полное сообщение?
- [ ] Почему нужен `version="v2"` в `astream_events()`?
- [ ] Как отследить потребление токенов при стриминге через callback?
- [ ] Как тип последнего элемента chain влияет на тип chunk при стриминге?

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

### 2. Пустые chunk-и при стриминге

```python
async for chunk in chain.astream(data):
    print(chunk.content, end="")
```

```python
async for chunk in chain.astream(data):
    if chunk.content:
        print(chunk.content, end="")
```

Некоторые chunk-и приходят с пустым `.content` (служебные). Без проверки код выведет `None` или пустую строку.

### 3. Блокирующий код в async-функции

```python
import time

async def process():
    time.sleep(1)
    async for chunk in chain.astream(data):
        print(chunk.content, end="")
```

```python
import asyncio

async def process():
    await asyncio.sleep(1)
    async for chunk in chain.astream(data):
        if chunk.content:
            print(chunk.content, end="")
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

В async-контексте синхронные callbacks блокируют event loop. Используйте `AsyncCallbackHandler` при работе с `astream()` и `astream_events()`.

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

# Тема 4: Streaming

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 2](topic_02_langchain_lcel.md), [Тема 3](topic_03_structured_output.md)
> **Где в проекте:** `app/api/v1/assessment.py`, `app/services/assessment.py`
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `sse-starlette`

---

## Теория

### 1. Зачем streaming

LLM генерирует текст **токен за токеном** (50-100 токенов/сек для Claude). Полный ответ на 1000 токенов занимает 10-20 секунд. Без стриминга пользователь видит **пустой экран** 20 секунд, а потом сразу весь текст.

Стриминг решает это: каждый токен отправляется клиенту **по мере генерации**. Пользователь видит текст "печатающийся" в реальном времени. Время до первого токена (TTFT — Time To First Token) — обычно 0.5-2 секунды.

### 2. stream() и astream()

LangChain Runnable поддерживает стриминг через единый интерфейс:

```python
# Синхронный стриминг
for chunk in chain.stream({"student_work": "..."}):
    print(chunk, end="", flush=True)

# Асинхронный стриминг (для FastAPI)
async for chunk in chain.astream({"student_work": "..."}):
    print(chunk.content, end="", flush=True)
```

Что приходит в chunk:
- Для ChatModel: `AIMessageChunk` — содержит `.content` (часть текста, обычно 1-3 токена).
- Для chain: зависит от последнего Runnable в цепочке. Если последний — ChatModel, то `AIMessageChunk`.
- Для structured output: **не работает** — `with_structured_output` собирает полный ответ.

### 3. astream_events() — продвинутый стриминг

`astream_events()` выдаёт **детальные события** с метаданными о каждом шаге цепочки:

```python
async for event in chain.astream_events(
    {"student_work": "..."},
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.content, end="")
    elif kind == "on_chain_end":
        print(f"\nDone! Output: {event['data']['output']}")
```

Типы событий:

| Событие | Когда | Данные |
|---------|-------|--------|
| `on_chain_start` | Начало выполнения chain | input |
| `on_chat_model_start` | Начало вызова LLM | messages |
| `on_chat_model_stream` | Каждый токен от LLM | AIMessageChunk |
| `on_chat_model_end` | LLM закончил | полный AIMessage |
| `on_chain_end` | Chain завершён | финальный output |

Зачем это нужно:
- Показать пользователю **прогресс**: "Анализирую критерий 3/5..."
- Собрать **метрики**: latency каждого шага, количество токенов.
- Логировать: полный trace вызова для дебага.

### 4. SSE (Server-Sent Events)

SSE — HTTP-протокол для **односторонней** потоковой отправки от сервера к клиенту. В отличие от WebSocket:

| | SSE | WebSocket |
|---|---|---|
| Направление | Сервер → клиент | Двунаправленное |
| Протокол | HTTP (обычный) | WS (отдельный) |
| Reconnect | Автоматический | Ручной |
| Сложность | Минимальная | Средняя |

SSE идеален для LLM стриминга: сервер отправляет токены, клиент только читает.

Формат SSE:
```
data: {"token": "The"}

data: {"token": " essay"}

data: {"token": " demonstrates"}

data: [DONE]
```

В FastAPI — через `sse-starlette`:

```python
from sse_starlette.sse import EventSourceResponse

@router.post("/stream")
async def stream_assessment(request: Request):
    async def generate():
        async for token in assess_stream(request.student_work, rubric, llm):
            yield {"data": token}
    return EventSourceResponse(generate())
```

### 5. Streaming + Structured Output = несовместимы

`with_structured_output()` использует tool calling: модель генерирует JSON, который парсится целиком. Невозможно распарсить половину JSON в Pydantic-объект.

**Стратегия в production:**
1. **Structured endpoint** (`POST /assess`) — `ainvoke()` → полный `AssessmentResponse`. Для API-to-API вызовов.
2. **Streaming endpoint** (`POST /assess/stream`) — `astream()` → текст токен за токеном. Для UI.

В проекте это уже реализовано: `assess()` использует structured chain, `assess_stream()` использует raw LLM без structured output.

### 6. Callback handlers

LangChain вызывает callback-функции на каждом этапе выполнения:

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("usage", {})
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
```

Callbacks — это **observer pattern**: не меняют поведение chain, только наблюдают. Используются для: трейсинга, подсчёта токенов, логирования, мониторинга.

---

## Ключевые концепции LangChain

### Текущая реализация стриминга в проекте

```python
# app/services/assessment.py — assess_stream()
async def assess_stream(student_work, rubric, llm):
    prompt = ChatPromptTemplate.from_messages([...])
    chain = prompt | llm  # без structured output
    async for chunk in chain.astream({"student_work": ..., "rubric": ...}):
        if chunk.content:
            yield chunk.content
```

Здесь `chain.astream()` даёт `AIMessageChunk`, из которого берётся `.content`. Это **самый простой** вариант стриминга.

### SSE endpoint в FastAPI

```python
# app/api/v1/assessment.py
@router.post("/stream")
async def assess_work_stream(request, llm, rubrics):
    rubric = _resolve_rubric(request, rubrics)
    async def event_generator():
        async for token in assess_stream(request.student_work, rubric, llm):
            yield {"data": token}
    return EventSourceResponse(event_generator())
```

---

## Практические задания

### Задание 1: astream_events

**Цель:** перейти от простого стриминга токенов к детальным событиям с метаданными.

**Файлы:** модификация `app/services/assessment.py`, `app/api/v1/assessment.py`

**Критерии успеха:**
- Streaming endpoint отправляет типизированные SSE-события: `start`, `token`, `end`
- Событие `end` содержит метаданные (total tokens, elapsed time)
- Клиент может отличить типы событий

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Refactor the streaming assessment to use astream_events() instead of astream().

1. Modify app/services/assessment.py — update assess_stream():
   - Use chain.astream_events(input, version="v2") instead of chain.astream(input)
   - Yield structured SSE events:
     - {"event": "start", "data": {"message": "Assessment started"}} on on_chain_start
     - {"event": "token", "data": {"content": chunk_text}} on on_chat_model_stream
     - {"event": "end", "data": {"message": "Assessment complete"}} on on_chain_end
   - Track and include elapsed time in the "end" event
   - Return type should be AsyncIterator[dict]

2. Modify app/api/v1/assessment.py — update assess_work_stream():
   - Adapt the event_generator to yield the structured events from assess_stream
   - Each SSE event should have "event" type and JSON "data"

3. Create experiments/t4_stream_events.py to test:
   - Call the streaming function directly (not through HTTP)
   - Print each event type and content as it arrives
   - Count total tokens received
   - Print total elapsed time

Run experiment with: python -m experiments.t4_stream_events
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the streaming changes in app/services/assessment.py and app/api/v1/assessment.py:

1. astream_events: Is version="v2" used? Is the correct event type filtered (on_chat_model_stream)?
2. EVENT TYPES: Are events clearly typed (start/token/end)?
3. METADATA: Does the "end" event include timing or token count?
4. SSE FORMAT: Is EventSourceResponse correctly yielding events?
5. EXPERIMENT: Does the test script demonstrate all event types?

Common mistakes:
- Using version="v1" instead of "v2" (v2 is more reliable)
- Not filtering events (on_chain_start fires for every sub-runnable)
- Yielding AIMessageChunk directly instead of extracting .content
- Not handling empty chunks (some chunks have no content)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Token counting

**Цель:** научиться отслеживать потребление токенов и стоимость каждого вызова.

**Файлы:** `app/callbacks/token_counter.py`, модификация streaming

**Критерии успеха:**
- Callback handler считает input и output токены
- Вычисляет стоимость по текущим ценам модели
- Статистика отправляется последним SSE-событием
- Корректно работает с async

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a token counting callback and integrate it into the streaming endpoint.

1. Create app/callbacks/__init__.py (empty)

2. Create app/callbacks/token_counter.py:
   - Class TokenCounterCallback(AsyncCallbackHandler):
     - Track input_tokens, output_tokens
     - on_llm_end: extract usage from response metadata
     - Method cost() that calculates USD cost based on model pricing:
       claude-sonnet: $3/M input, $15/M output
       claude-haiku: $0.25/M input, $1.25/M output
     - Method summary() returning dict with input_tokens, output_tokens, total_tokens, cost_usd

3. Modify app/services/assessment.py — assess_stream():
   - Create TokenCounterCallback instance
   - Pass it via config={"callbacks": [counter]} to astream or astream_events
   - After streaming completes, yield a final event with counter.summary()

4. Create experiments/t4_token_counting.py:
   - Run assessment with token counter
   - Print: input tokens, output tokens, total cost
   - Compare costs for different essay lengths (short vs long)

Run experiment with: python -m experiments.t4_token_counting
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/callbacks/token_counter.py and its integration:

1. CALLBACK: Does it extend AsyncCallbackHandler and implement on_llm_end?
2. TOKEN EXTRACTION: Does it correctly extract usage from response metadata?
3. COST CALCULATION: Are the pricing numbers correct for current Claude models?
4. INTEGRATION: Is the callback passed via config={"callbacks": [counter]}?
5. SSE: Is the cost summary sent as the final streaming event?
6. ASYNC SAFETY: Is the callback async-compatible?

Common mistakes:
- Using BaseCallbackHandler instead of AsyncCallbackHandler for async chains
- Wrong location for usage data (varies by model provider)
- Hardcoded prices without easy way to update
- Not resetting counters between calls

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: Прогресс-бар

**Цель:** дать пользователю понимание, на каком этапе оценки система находится.

**Файлы:** модификация `app/services/assessment.py`

**Критерии успеха:**
- SSE-события прогресса между основными этапами
- Клиент получает: "Анализирую работу...", "Оцениваю критерий 2/5...", "Формирую итог..."
- Прогресс-события отделены от текстовых токенов

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Add progress events to the streaming assessment.

Modify app/services/assessment.py — update assess_stream():

1. Break the streaming into logical stages with progress events:
   - Yield {"event": "progress", "data": {"stage": "analyzing", "message": "Analyzing student work..."}}
   - For each criterion in the rubric:
     Yield {"event": "progress", "data": {"stage": "scoring", "message": f"Evaluating criterion {i}/{total}: {criterion_name}..."}}
   - Yield {"event": "progress", "data": {"stage": "finalizing", "message": "Generating final assessment..."}}

2. Strategy: Since we can't know exactly when the LLM switches criteria,
   use a heuristic approach:
   - Parse the rubric to get criterion names
   - As tokens stream in, detect when a new criterion name appears
   - Send a progress event when a new criterion is detected

3. Alternative simpler approach:
   - Send "analyzing" at start
   - Monitor streamed text for criterion names from the rubric
   - Send progress update when criterion name is detected in the stream
   - Send "finalizing" when all criteria are found

4. Create experiments/t4_progress.py to test:
   - Call streaming assessment
   - Print progress events distinctly from text tokens
   - Show a simple CLI progress bar

Run experiment with: python -m experiments.t4_progress
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the progress event implementation:

1. EVENT SEPARATION: Are progress events clearly separated from token events?
2. CRITERION DETECTION: Does it detect when the model starts evaluating a new criterion?
3. RUBRIC AWARENESS: Does it use the actual rubric criteria names (not hardcoded)?
4. UX: Are progress messages user-friendly and informative?
5. ROBUSTNESS: Does it handle cases where criterion names aren't found in the stream?

Common mistakes:
- Hardcoding criterion names instead of reading from rubric
- Progress events mixed with token events (no event type separation)
- Detection logic too brittle (exact match vs fuzzy match)
- Not handling the case where LLM outputs criteria in different order

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Почему structured output и streaming несовместимы? Как обходить?
- [ ] В чём разница между `astream()` и `astream_events()`? Когда какой использовать?
- [ ] Объясни SSE: как протокол, формат сообщения, отличие от WebSocket.
- [ ] Зачем callback handlers, если можно просто логировать в chain?
- [ ] Что такое TTFT и почему это ключевая метрика UX для LLM?

---

## Частые ошибки

### 1. Стриминг structured output

```python
# Не работает: with_structured_output собирает полный ответ
async for chunk in structured_chain.astream(data):
    print(chunk)  # получишь весь AssessmentResponse целиком, одним куском

# Правильно: стримить текст, structured — через invoke
# Streaming endpoint: prompt | llm (text tokens)
# Structured endpoint: prompt | llm.with_structured_output(Schema)
```

### 2. Не обрабатывать пустые чанки

```python
# Плохо: может отправить пустые SSE-события
async for chunk in chain.astream(data):
    yield {"data": chunk.content}

# Хорошо: фильтровать пустые
async for chunk in chain.astream(data):
    if chunk.content:
        yield {"data": chunk.content}
```

### 3. Блокирующий код в async генераторе

```python
# Плохо: time.sleep блокирует event loop
async def generate():
    yield {"data": "start"}
    time.sleep(1)  # БЛОКИРУЕТ!
    async for chunk in chain.astream(data):
        yield {"data": chunk.content}

# Хорошо: asyncio.sleep для пауз
async def generate():
    yield {"data": "start"}
    await asyncio.sleep(1)
    async for chunk in chain.astream(data):
        yield {"data": chunk.content}
```

---

## Что читать дальше

- [LangChain Streaming](https://python.langchain.com/docs/concepts/streaming/) — концепция
- [How to stream responses](https://python.langchain.com/docs/how_to/streaming/) — практика
- [astream_events](https://python.langchain.com/docs/how_to/streaming/#using-stream-events) — продвинутый стриминг
- [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html) — спецификация протокола
- [sse-starlette](https://github.com/sysid/sse-starlette) — SSE для FastAPI

**Следующая тема:** [Тема 5: RAG](topic_05_rag.md) — как давать LLM контекст из документов.

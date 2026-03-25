# Тема 2: LangChain Core + LCEL

> **Пререквизиты:** [Тема 1: Промпт-инжиниринг](topic_01_prompt_engineering.md)
> **Что добавим в проект:** `app/api/v1/chains.py` — роутер с 4 эндпоинтами
> **Зависимости:** `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. Runnable protocol — единый интерфейс

LCEL (LangChain Expression Language) построен на одной абстракции: **Runnable**. Каждый компонент — промпт, модель, парсер, функция — реализует один и тот же интерфейс:

| Метод | Описание |
|-------|----------|
| `invoke(input)` | Синхронный вызов. Отправляет input, ждёт полный результат. |
| `ainvoke(input)` | Асинхронный invoke. Для FastAPI и async-кода. |
| `stream(input)` | Генератор, выдаёт результат по частям (токенам). |
| `astream(input)` | Асинхронный stream. |
| `batch(inputs)` | Параллельный вызов на списке входов. |
| `abatch(inputs)` | Асинхронный batch. |

Это аналог паттерна **Strategy** из ООП, но для data pipelines. Любой Runnable можно подставить вместо любого другого, если типы входа/выхода совпадают.

Почему это мощно:
- **Композиция.** Runnables соединяются оператором `|` в цепочки. `prompt | model | parser` — это 3 Runnables, соединённых в один.
- **Единообразие.** И ChatAnthropic, и ChatOpenAI, и ваша кастомная функция — все вызываются через `.invoke()`. Замена модели — это одна строка.
- **Бесплатный параллелизм.** `batch()` автоматически запускает вызовы параллельно (с настраиваемым concurrency).

#### Типы входов и выходов

Каждый Runnable имеет **строго типизированный** интерфейс. Можно инспектировать типы программно:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_messages([("system", "{role}"), ("human", "{question}")])
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

chain = prompt | llm

print(chain.input_schema.model_json_schema())
print(chain.output_schema.model_json_schema())
```

Это позволяет LangChain валидировать цепочки при создании, а не при вызове. Если выход одного Runnable не совместим со входом следующего, ошибка будет при построении chain, а не в runtime.

#### Config и метаданные

Каждый вызов Runnable принимает опциональный `config`:

```python
result = await chain.ainvoke(
    {"student_work": "..."},
    config={
        "max_concurrency": 5,
        "tags": ["assessment", "v1"],
        "metadata": {"user_id": "123"},
        "run_name": "essay_assessment",
    },
)
```

Config пропагируется через всю цепочку: если ты передал `tags` в `chain.ainvoke()`, каждый внутренний Runnable (prompt, model, parser) получит эти теги. Это критично для observability — в LangSmith/LangFuse ты увидишь каждый шаг с метаданными.

### 2. Pipe-оператор `|`

Оператор `|` создаёт `RunnableSequence` — цепочку, где выход левого Runnable становится входом правого:

```python
chain = prompt | model | parser
```

Что происходит внутри при `chain.invoke({"question": "..."})`:
1. `prompt.invoke({"question": "..."})` → `[SystemMessage(...), HumanMessage(...)]`
2. `model.invoke([SystemMessage(...), HumanMessage(...)])` → `AIMessage("...")`
3. `parser.invoke(AIMessage("..."))` → `{"answer": "..."}`

Каждый шаг трансформирует данные и передаёт дальше. Это **функциональная композиция**: `f(g(h(x)))` записанная как `h | g | f`.

#### Как это работает под капотом

Оператор `|` определён в базовом классе `Runnable.__or__()`. При `a | b` создаётся `RunnableSequence(first=a, last=b)`. При `a | b | c` — `RunnableSequence(first=a, middle=[b], last=c)`. Цепочки плоские, не вложенные.

Важные свойства `RunnableSequence`:
- **Потоковая передача**: `chain.stream()` вызывает `stream()` на последнем элементе, а все предыдущие обрабатываются через `invoke()`. Это означает: streaming работает только для последнего шага.
- **Async**: `chain.ainvoke()` вызывает `ainvoke()` на каждом шаге последовательно.
- **Batch**: `chain.batch(inputs)` обрабатывает каждый input через всю цепочку параллельно.

В проекте:

```python
chain = prompt | structured_llm
```

Здесь 2 Runnables: `prompt` (ChatPromptTemplate) → messages, `structured_llm` (ChatAnthropic + structured output) → AssessmentResponse.

#### Отладка цепочек

Для понимания, что происходит на каждом шаге:

```python
from langchain_core.runnables import RunnableLambda

def debug_step(data):
    print(f"Type: {type(data).__name__}, Data: {str(data)[:200]}")
    return data

chain = prompt | RunnableLambda(debug_step) | structured_llm
```

В production используй callbacks или LangSmith вместо print-debug.

### 3. RunnablePassthrough

Пробрасывает входные данные без изменений. Зачем? Чтобы **добавлять** данные, не теряя исходные:

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split()),
    timestamp=lambda _: datetime.now().isoformat(),
) | prompt | model
```

`RunnablePassthrough.assign()` берёт все входные ключи + добавляет новые. Это как spread-оператор `{...input, word_count: 42}`.

#### RunnablePassthrough() vs RunnablePassthrough.assign()

| Вариант | Поведение | Пример |
|---|---|---|
| `RunnablePassthrough()` | Пробрасывает input без изменений | Используется как identity-функция в параллельных ветках |
| `RunnablePassthrough.assign(key=fn)` | Пробрасывает input + добавляет новые ключи | Обогащение данных перед промптом |

Частый паттерн — `RunnablePassthrough.assign()` в начале chain для подготовки данных:

```python
chain = (
    RunnablePassthrough.assign(
        rubric_text=lambda x: format_rubric(x["rubric"]),
        word_count=lambda x: len(x["student_work"].split()),
    )
    | prompt
    | model
)
```

Входной dict `{"student_work": "...", "rubric": Rubric(...)}` превращается в `{"student_work": "...", "rubric": Rubric(...), "rubric_text": "...", "word_count": 150}`. Все исходные ключи сохраняются.

### 4. RunnableParallel

Запускает несколько Runnables **параллельно** на одном входе и собирает результаты в dict:

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    assessment=assessment_chain,
    metadata=metadata_chain,
)

result = chain.invoke({"student_work": "..."})
```

Под капотом: `RunnableParallel` запускает все ветки через `asyncio.gather()` (или `ThreadPoolExecutor` для синхронного кода). Реальный параллелизм, не sequential.

#### Как RunnableParallel маршрутизирует данные

Критически важно: **каждая ветка получает одинаковый input**. Если input — `{"student_work": "...", "rubric": "..."}`, то и `assessment_chain`, и `metadata_chain` получат этот же dict.

Если веткам нужны **разные** данные, используй `.partial()` на промптах внутри веток:

```python
chain_rubric_1 = prompt.partial(rubric=rubric_1_text) | structured_llm
chain_rubric_2 = prompt.partial(rubric=rubric_2_text) | structured_llm

parallel = RunnableParallel(
    assessment_1=chain_rubric_1,
    assessment_2=chain_rubric_2,
)

result = await parallel.ainvoke({"student_work": "..."})
```

#### Время выполнения

Время параллельного выполнения ≈ max(время каждой ветки), а не сумме. Если 2 ветки по 3 секунды каждая — параллельное выполнение займёт ~3 секунды, а не 6.

Ограничение: API rate limits. Если у вас лимит 5 req/s, запуск 10 параллельных веток приведёт к throttling. Используй `config={"max_concurrency": N}` для контроля.

### 5. RunnableLambda

Обёртка обычной Python-функции в Runnable:

```python
from langchain_core.runnables import RunnableLambda

def preprocess(input: dict) -> dict:
    text = input["student_work"]
    return {**input, "word_count": len(text.split())}

chain = RunnableLambda(preprocess) | prompt | model
```

Или через декоратор `@chain`:

```python
from langchain_core.runnables import chain

@chain
def preprocess(input: dict) -> dict:
    return {**input, "word_count": len(input["student_work"].split())}
```

`RunnableLambda` делает вашу функцию полноправным участником LCEL — с `invoke`, `ainvoke`, `batch` и т.д.

#### Async-функции в RunnableLambda

Если функция async — `RunnableLambda` автоматически использует её для `ainvoke`:

```python
async def fetch_rubric(input: dict) -> dict:
    rubric = await db.get_rubric(input["rubric_id"])
    return {**input, "rubric": rubric}

chain = RunnableLambda(fetch_rubric) | prompt | model
```

Можно передать обе версии (sync и async):

```python
def sync_process(x): ...
async def async_process(x): ...

step = RunnableLambda(sync_process, afunc=async_process)
```

#### Когда RunnableLambda, а когда просто функция

| Ситуация | Решение |
|---|---|
| Нужна функция в LCEL pipe `\|` | `RunnableLambda` |
| Нужна предобработка перед chain | `RunnablePassthrough.assign()` или `RunnableLambda` |
| Нужна логика вне цепочки (до/после invoke) | Обычная функция |
| Нужна функция с batch/stream поддержкой | `RunnableLambda` |

### 6. .with_fallbacks()

Если основной Runnable бросает исключение — автоматически переключается на запасной:

```python
primary = ChatAnthropic(model="claude-sonnet-4-20250514")
backup = ChatAnthropic(model="claude-haiku-4-20250414")

reliable_model = primary.with_fallbacks([backup])
```

Сценарии:
- Rate limit от основного провайдера → переключение на запасной
- Модель вернула невалидный ответ → попытка с другой моделью
- Timeout → fallback на более быструю модель

Fallback срабатывает **на любое исключение**. Можно указать конкретные: `with_fallbacks([backup], exceptions_to_handle=(RateLimitError,))`.

#### Несколько уровней fallback

Fallback может быть каскадным:

```python
reliable = sonnet.with_fallbacks([haiku, gpt4_mini])
```

Порядок: sonnet → при ошибке haiku → при ошибке gpt4_mini. Каждый следующий fallback вызывается только при ошибке предыдущего.

#### Fallback на модели vs fallback на цепочке

```python
reliable_model = primary_llm.with_fallbacks([backup_llm])
chain = prompt | reliable_model

reliable_chain = (prompt | primary_llm).with_fallbacks([prompt | backup_llm])
```

Разница: в первом случае fallback происходит только на уровне модели (промпт вызывается один раз). Во втором — fallback перезапускает всю цепочку (промпт вызывается повторно).

Для structured output **обязательно** настраивать fallback на уровне structured_llm:

```python
primary_structured = primary.with_structured_output(Schema)
backup_structured = backup.with_structured_output(Schema)
reliable = primary_structured.with_fallbacks([backup_structured])
```

### 7. .with_retry()

Повторные попытки при ошибках с exponential backoff:

```python
chain = (prompt | model).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

Полезно для transient errors (rate limits, timeouts, network issues). Не помогает при ошибках в логике (неправильный промпт не станет правильным от повтора).

#### Retry vs Fallback: когда что

| Ситуация | Retry | Fallback |
|---|---|---|
| Rate limit (429) | Да (backoff даст время) | Да (другой провайдер) |
| Timeout | Да (сеть могла восстановиться) | Да (более быстрая модель) |
| Invalid JSON output | Может помочь | Лучше (другая модель может справиться) |
| Wrong API key | Нет (ошибка постоянная) | Да (другой провайдер) |
| Model overloaded | Да (backoff) | Да (другая модель) |

На практике оба паттерна часто комбинируют:

```python
chain = (prompt | model.with_retry(stop_after_attempt=2)).with_fallbacks(
    [prompt | backup_model.with_retry(stop_after_attempt=2)]
)
```

Сначала retry на primary (2 попытки), затем fallback на backup (тоже 2 попытки). Итого максимум 4 попытки.

#### Параметры retry

```python
chain.with_retry(
    retry_if_exception_type=(RateLimitError, TimeoutError),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    wait_exponential_multiplier=1,
    wait_exponential_max=10,
)
```

`wait_exponential_jitter=True` добавляет случайный jitter к backoff. Без jitter 100 параллельных retry-запросов "проснутся" одновременно и снова перегрузят API. С jitter они размазываются по времени.

### 8. .bind()

Привязывает дополнительные аргументы к Runnable:

```python
model = ChatAnthropic(model="claude-sonnet-4-20250514")
model_with_tools = model.bind(tools=[...])
model_with_stop = model.bind(stop=["\n\n"])
```

`bind()` создаёт новый Runnable, который при вызове передаёт привязанные аргументы. Исходный Runnable не меняется (иммутабельность).

#### Типичные использования bind()

| Сценарий | Пример |
|---|---|
| Tool calling | `model.bind(tools=[tool_schema])` |
| Stop sequences | `model.bind(stop=["\n\n", "END"])` |
| Response format | `model.bind(response_format={"type": "json_object"})` |
| Function calling (OpenAI) | `model.bind(functions=[fn_schema])` |

`bind()` отличается от передачи параметров в конструктор тем, что можно привязывать параметры **после** создания модели, создавая специализированные версии одного инстанса:

```python
base_model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
model_for_analysis = base_model.bind(stop=["## Conclusion"])
model_for_scoring = base_model.bind(max_tokens=256)
```

### 9. batch() и abatch()

Параллельная обработка нескольких входов:

```python
inputs = [
    {"student_work": essay_1, "rubric": rubric_text},
    {"student_work": essay_2, "rubric": rubric_text},
    {"student_work": essay_3, "rubric": rubric_text},
]

results = await chain.abatch(inputs)
```

#### Как batch работает под капотом

`batch()` / `abatch()` не просто запускает N последовательных вызовов. Внутри:

1. Для `abatch()`: создаётся `asyncio.gather()` с N coroutines
2. Для `batch()`: создаётся `ThreadPoolExecutor` с N потоками
3. `max_concurrency` контролирует максимальное число одновременных вызовов через `asyncio.Semaphore`

```python
results = await chain.abatch(
    inputs,
    config={"max_concurrency": 5},
)
```

Без `max_concurrency` все inputs обрабатываются одновременно. Для 100 inputs — это 100 одновременных API-запросов, что почти гарантированно вызовет rate limiting.

#### batch vs RunnableParallel

| | batch/abatch | RunnableParallel |
|---|---|---|
| Входы | Список одинаковых по структуре | Один вход, разные обработчики |
| Выход | Список результатов (одинаковый тип) | Dict с именованными результатами (разные типы) |
| Пример | 10 эссе через одну chain | 1 эссе через 2 разных рубрики |
| Под капотом | asyncio.gather + semaphore | asyncio.gather |

### 10. ChatModel vs LLM

| | ChatModel (BaseChatModel) | LLM (BaseLLM) |
|---|---|---|
| Вход | Список messages | Строка текста |
| Выход | AIMessage | Строка текста |
| Примеры | ChatAnthropic, ChatOpenAI | (legacy) OpenAI completion |
| API | Chat Completions | (deprecated) Completions |

В 2024+ **все современные модели** — это ChatModel. BaseLLM — legacy для старых completion-моделей. В новом коде используй только ChatModel.

### 11. Как устроен chain в проекте

```python
def build_assessment_chain(llm: ChatAnthropic) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ASSESSMENT_SYSTEM_PROMPT),
        ("human", "Please assess:\n\n{student_work}"),
    ]).partial(few_shot_good=..., few_shot_bad=...)

    structured_llm = llm.with_structured_output(AssessmentResponse)
    return prompt | structured_llm
```

Здесь 2 Runnables в цепочке:
1. `prompt` — `ChatPromptTemplate` → принимает `dict`, возвращает `list[BaseMessage]`
2. `structured_llm` — `ChatAnthropic` + structured output → принимает `list[BaseMessage]`, возвращает `AssessmentResponse`

### 12. invoke vs ainvoke в FastAPI

FastAPI — async framework. Используй `ainvoke`:

```python
@router.post("")
async def assess_work(request: AssessmentRequest, chain: ChainDep):
    result = await chain.ainvoke({"student_work": request.student_work, "rubric": rubric_text})
    return result
```

Если использовать `invoke` в async endpoint — заблокируешь event loop. FastAPI запустит его в threadpool, но это хуже, чем нативный `ainvoke`.

Правило: в `async def` endpoint → `ainvoke` / `abatch` / `astream`. В `def` endpoint (sync) → `invoke` / `batch` / `stream`. Никогда не смешивай.

---

## Справочник API

### RunnablePassthrough

**Описание:** Identity-runnable, пробрасывающий входные данные без изменений. Основное использование — `.assign()` для обогащения данных новыми полями без потери исходных.

```python
RunnablePassthrough(
    func: Callable | None = None,
    afunc: Callable | None = None,
    input_type: type | None = None,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `func` | `Callable \| None` | `None` | Побочный эффект при прохождении данных (данные не меняются) |
| `afunc` | `Callable \| None` | `None` | Async-версия func |
| `input_type` | `type \| None` | `None` | Тип входных данных для валидации |

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `.assign(**kwargs)` | `keyword args (key=Callable)` | `RunnableAssign` | Создаёт runnable, добавляющий новые ключи к input dict |
| `.invoke(input)` | `Any` | `Any` | Возвращает input без изменений |
| `.ainvoke(input)` | `Any` | `Any` | Async-версия invoke |

**Пример использования:**

```python
from langchain_core.runnables import RunnablePassthrough

enrich = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split()),
    char_count=lambda x: len(x["text"]),
)

result = enrich.invoke({"text": "Hello world", "id": 1})
```

### RunnableParallel

**Описание:** Запускает несколько Runnables параллельно на одном входе. Результаты собираются в dict с именованными ключами. Используй для одновременной обработки данных разными chain.

```python
RunnableParallel(
    steps: Mapping[str, Runnable] | None = None,
    **kwargs: Runnable,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `steps` | `Mapping[str, Runnable] \| None` | `None` | Именованные Runnable-ветки |
| `**kwargs` | `Runnable` | — | Альтернативный способ задать ветки как keyword args |

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `.invoke(input)` | `Any` | `dict[str, Any]` | Параллельный запуск всех веток, возвращает dict результатов |
| `.ainvoke(input)` | `Any` | `dict[str, Any]` | Async-версия |
| `.batch(inputs)` | `list[Any]` | `list[dict[str, Any]]` | Batch-версия |

**Пример использования:**

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    essay_score=essay_chain,
    creative_score=creative_chain,
)

results = await parallel.ainvoke({"student_work": "..."})
```

### RunnableLambda

**Описание:** Обёртка Python-функции в Runnable. Позволяет вставлять произвольную логику в LCEL-цепочки. Функция получает полный набор методов Runnable (invoke, ainvoke, batch, stream).

```python
RunnableLambda(
    func: Callable,
    afunc: Callable | None = None,
    name: str | None = None,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `func` | `Callable` | — | Синхронная функция для invoke/batch |
| `afunc` | `Callable \| None` | `None` | Асинхронная функция для ainvoke/abatch |
| `name` | `str \| None` | `None` | Имя для отладки и tracing |

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `.invoke(input)` | `Any` | `Any` | Вызов func(input) |
| `.ainvoke(input)` | `Any` | `Any` | Вызов afunc(input) или func(input) в threadpool |
| `.batch(inputs)` | `list[Any]` | `list[Any]` | Параллельный вызов для каждого input |
| `.map()` | — | `RunnableEach` | Создаёт runnable, применяющий функцию к каждому элементу списка |

**Пример использования:**

```python
from langchain_core.runnables import RunnableLambda

def add_metadata(input: dict) -> dict:
    return {
        **input,
        "word_count": len(input["student_work"].split()),
        "has_citations": "et al" in input["student_work"],
    }

chain = RunnableLambda(add_metadata) | prompt | structured_llm
```

### RunnableSequence

**Описание:** Цепочка Runnables, где выход предыдущего становится входом следующего. Создаётся автоматически оператором `|`. Обычно не создаётся вручную.

```python
RunnableSequence(
    first: Runnable,
    *middle: Runnable,
    last: Runnable,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `first` | `Runnable` | — | Первый элемент цепочки |
| `*middle` | `Runnable` | — | Промежуточные элементы |
| `last` | `Runnable` | — | Последний элемент |

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `.invoke(input)` | `Input of first` | `Output of last` | Последовательный вызов всех шагов |
| `.ainvoke(input)` | `Input of first` | `Output of last` | Async-версия |
| `.stream(input)` | `Input of first` | `Iterator[Output of last]` | Streaming последнего шага |
| `.batch(inputs)` | `list[Input]` | `list[Output]` | Параллельный batch |

**Пример использования:**

```python
chain = prompt | llm | parser

chain.first
chain.middle
chain.last

for step in chain.steps:
    print(type(step).__name__)
```

### .with_fallbacks()

**Описание:** Метод Runnable, добавляющий запасные варианты при ошибках. Возвращает `RunnableWithFallbacks`, который при исключении в основном Runnable автоматически переключается на следующий в списке.

```python
runnable.with_fallbacks(
    fallbacks: list[Runnable],
    exceptions_to_handle: tuple[type[Exception], ...] = (Exception,),
    exception_key: str | None = None,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `fallbacks` | `list[Runnable]` | — | Список запасных Runnables в порядке приоритета |
| `exceptions_to_handle` | `tuple[type[Exception], ...]` | `(Exception,)` | Типы исключений, запускающие fallback |
| `exception_key` | `str \| None` | `None` | Ключ для сохранения исключения в output dict |

**Пример использования:**

```python
from langchain_anthropic import ChatAnthropic

primary = ChatAnthropic(model="claude-sonnet-4-20250514").with_structured_output(Schema)
backup = ChatAnthropic(model="claude-haiku-4-20250414").with_structured_output(Schema)

reliable = primary.with_fallbacks([backup])
chain = prompt | reliable
```

### .with_retry()

**Описание:** Метод Runnable, добавляющий автоматический retry с exponential backoff. Полезен для transient errors: rate limits, timeouts, временные сбои сети.

```python
runnable.with_retry(
    retry_if_exception_type: tuple[type[Exception], ...] = (Exception,),
    wait_exponential_jitter: bool = True,
    stop_after_attempt: int = 3,
    wait_exponential_multiplier: float = 1,
    wait_exponential_max: float = 60,
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `retry_if_exception_type` | `tuple` | `(Exception,)` | Типы ошибок для retry |
| `wait_exponential_jitter` | `bool` | `True` | Добавлять случайный jitter к backoff |
| `stop_after_attempt` | `int` | `3` | Максимум попыток |
| `wait_exponential_multiplier` | `float` | `1` | Множитель для backoff (секунды) |
| `wait_exponential_max` | `float` | `60` | Максимальное время ожидания между попытками |

**Пример использования:**

```python
chain = (prompt | model).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

### .bind()

**Описание:** Метод Runnable, привязывающий дополнительные kwargs. Создаёт новый Runnable, который при каждом вызове передаёт привязанные аргументы. Исходный Runnable не изменяется.

```python
runnable.bind(**kwargs: Any) -> RunnableBinding
```

| Параметр | Тип | Описание |
|---|---|---|
| `**kwargs` | `Any` | Аргументы, передаваемые при каждом вызове |

**Пример использования:**

```python
model = ChatAnthropic(model="claude-sonnet-4-20250514")

model_with_stop = model.bind(stop=["END", "\n\n"])
model_with_tools = model.bind(tools=[tool_definition])
```

### .batch() / .abatch()

**Описание:** Методы Runnable для параллельной обработки списка входов. `batch()` — синхронная версия, `abatch()` — асинхронная. Автоматически управляют concurrency.

```python
runnable.batch(
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Output]

runnable.abatch(
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Output]
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `inputs` | `list[Input]` | — | Список входов для параллельной обработки |
| `config` | `RunnableConfig \| None` | `None` | Конфиг; `max_concurrency` управляет параллелизмом |
| `return_exceptions` | `bool` | `False` | Если True, ошибки возвращаются в список вместо raise |

**Пример использования:**

```python
inputs = [
    {"student_work": essay_1, "rubric": rubric_text},
    {"student_work": essay_2, "rubric": rubric_text},
    {"student_work": essay_3, "rubric": rubric_text},
]

results = await chain.abatch(inputs, config={"max_concurrency": 3})

results_safe = await chain.abatch(
    inputs,
    return_exceptions=True,
)
for r in results_safe:
    if isinstance(r, Exception):
        print(f"Error: {r}")
    else:
        print(f"Score: {r.overall_score}")
```

---

## Практика: роутер `/api/v1/chains`

В этом разделе мы создадим FastAPI-роутер с 4 эндпоинтами, каждый из которых демонстрирует одну из LCEL-концепций.

| Эндпоинт | Концепция | Что проверяем |
|---|---|---|
| `POST /enriched` | RunnablePassthrough.assign() | Обогащение данных метаинформацией |
| `POST /parallel` | RunnableParallel | Параллельная оценка по 2 рубрикам |
| `POST /batch` | chain.abatch() | Пакетная оценка нескольких работ |
| `POST /with-fallback` | .with_fallbacks() | Отказоустойчивость с fallback-моделью |

### Шаг 1. Схемы запросов и ответов

```python
from pydantic import BaseModel, Field
from app.schemas.assessment import AssessmentResponse
from app.schemas.rubric import Rubric


class EnrichedAssessmentRequest(BaseModel):
    student_work: str
    rubric: Rubric


class AssessmentMetadata(BaseModel):
    word_count: int
    paragraph_count: int
    timestamp: str


class EnrichedAssessmentResponse(BaseModel):
    assessment: AssessmentResponse
    metadata: AssessmentMetadata


class ParallelAssessmentRequest(BaseModel):
    student_work: str
    rubric_1: Rubric
    rubric_2: Rubric


class ParallelAssessmentResponse(BaseModel):
    rubric_1_result: AssessmentResponse
    rubric_2_result: AssessmentResponse
    elapsed_seconds: float


class BatchAssessmentRequest(BaseModel):
    works: list[str] = Field(min_length=1, max_length=10)
    rubric: Rubric
    max_concurrency: int = Field(default=3, ge=1, le=10)


class BatchAssessmentResponse(BaseModel):
    results: list[AssessmentResponse]
    total_works: int
    elapsed_seconds: float


class FallbackAssessmentRequest(BaseModel):
    student_work: str
    rubric: Rubric


class FallbackAssessmentResponse(BaseModel):
    assessment: AssessmentResponse
    primary_model: str
    fallback_model: str
```

- `EnrichedAssessmentResponse` возвращает assessment + metadata, вычисленную через `RunnablePassthrough.assign()`
- `ParallelAssessmentResponse` содержит результаты обеих рубрик + время выполнения (чтобы убедиться в параллельности)
- `BatchAssessmentRequest` ограничивает размер batch (1-10 работ) и concurrency (1-10)
- `FallbackAssessmentResponse` показывает, какие модели были сконфигурированы как primary и fallback

### Шаг 2. Роутер `app/api/v1/chains.py`

Полный файл роутера. Создай `app/api/v1/chains.py`:

```python
import time
from datetime import UTC, datetime

from fastapi import APIRouter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from app.dependencies import SettingsDep
from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse
from app.schemas.rubric import Rubric

router = APIRouter(prefix="/chains", tags=["lesson-2-chains"])


class EnrichedAssessmentRequest(BaseModel):
    student_work: str
    rubric: Rubric


class AssessmentMetadata(BaseModel):
    word_count: int
    paragraph_count: int
    timestamp: str


class EnrichedAssessmentResponse(BaseModel):
    assessment: AssessmentResponse
    metadata: AssessmentMetadata


class ParallelAssessmentRequest(BaseModel):
    student_work: str
    rubric_1: Rubric
    rubric_2: Rubric


class ParallelAssessmentResponse(BaseModel):
    rubric_1_result: AssessmentResponse
    rubric_2_result: AssessmentResponse
    elapsed_seconds: float


class BatchAssessmentRequest(BaseModel):
    works: list[str] = Field(min_length=1, max_length=10)
    rubric: Rubric
    max_concurrency: int = Field(default=3, ge=1, le=10)


class BatchAssessmentResponse(BaseModel):
    results: list[AssessmentResponse]
    total_works: int
    elapsed_seconds: float


class FallbackAssessmentRequest(BaseModel):
    student_work: str
    rubric: Rubric


class FallbackAssessmentResponse(BaseModel):
    assessment: AssessmentResponse
    primary_model: str
    fallback_model: str


def format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", ASSESSMENT_SYSTEM_PROMPT),
        (
            "human",
            "Word count: {word_count}\nParagraph count: {paragraph_count}\n"
            "Timestamp: {timestamp}\n\n"
            "Please assess the following student work:\n\n{student_work}",
        ),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )


def build_simple_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", ASSESSMENT_SYSTEM_PROMPT),
        ("human", "Please assess the following student work:\n\n{student_work}"),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )


@router.post("/enriched")
async def enriched_assessment(
    request: EnrichedAssessmentRequest,
    settings: SettingsDep,
) -> EnrichedAssessmentResponse:
    rubric_text = format_rubric(request.rubric)
    llm = ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )

    enrich = RunnablePassthrough.assign(
        word_count=lambda x: len(x["student_work"].split()),
        paragraph_count=lambda x: len([p for p in x["student_work"].split("\n\n") if p.strip()]),
        timestamp=lambda _: datetime.now(UTC).isoformat(),
    )

    prompt = build_prompt()
    chain = enrich | prompt | llm.with_structured_output(AssessmentResponse)

    result = await chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
    })

    return EnrichedAssessmentResponse(
        assessment=result,
        metadata=AssessmentMetadata(
            word_count=len(request.student_work.split()),
            paragraph_count=len([p for p in request.student_work.split("\n\n") if p.strip()]),
            timestamp=datetime.now(UTC).isoformat(),
        ),
    )


@router.post("/parallel")
async def parallel_assessment(
    request: ParallelAssessmentRequest,
    settings: SettingsDep,
) -> ParallelAssessmentResponse:
    llm = ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )
    structured_llm = llm.with_structured_output(AssessmentResponse)

    rubric_1_text = format_rubric(request.rubric_1)
    rubric_2_text = format_rubric(request.rubric_2)

    prompt_1 = build_simple_prompt().partial(rubric=rubric_1_text)
    prompt_2 = build_simple_prompt().partial(rubric=rubric_2_text)

    parallel = RunnableParallel(
        rubric_1=prompt_1 | structured_llm,
        rubric_2=prompt_2 | structured_llm,
    )

    start = time.monotonic()
    results = await parallel.ainvoke({"student_work": request.student_work})
    elapsed = time.monotonic() - start

    return ParallelAssessmentResponse(
        rubric_1_result=results["rubric_1"],
        rubric_2_result=results["rubric_2"],
        elapsed_seconds=round(elapsed, 2),
    )


@router.post("/batch")
async def batch_assessment(
    request: BatchAssessmentRequest,
    settings: SettingsDep,
) -> BatchAssessmentResponse:
    rubric_text = format_rubric(request.rubric)
    llm = ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )

    prompt = build_simple_prompt()
    chain = prompt | llm.with_structured_output(AssessmentResponse)

    inputs = [{"student_work": work, "rubric": rubric_text} for work in request.works]

    start = time.monotonic()
    results = await chain.abatch(inputs, config={"max_concurrency": request.max_concurrency})
    elapsed = time.monotonic() - start

    return BatchAssessmentResponse(
        results=results,
        total_works=len(request.works),
        elapsed_seconds=round(elapsed, 2),
    )


@router.post("/with-fallback")
async def fallback_assessment(
    request: FallbackAssessmentRequest,
    settings: SettingsDep,
) -> FallbackAssessmentResponse:
    rubric_text = format_rubric(request.rubric)
    primary_model = "claude-sonnet-4-20250514"
    fallback_model = "claude-haiku-4-20250414"

    primary = ChatAnthropic(
        model=primary_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )
    fallback = ChatAnthropic(
        model=fallback_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )

    primary_structured = primary.with_structured_output(AssessmentResponse)
    fallback_structured = fallback.with_structured_output(AssessmentResponse)
    reliable_llm = primary_structured.with_fallbacks([fallback_structured])

    prompt = build_simple_prompt()
    chain = prompt | reliable_llm

    result = await chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
    })

    return FallbackAssessmentResponse(
        assessment=result,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )
```

**Как каждый эндпоинт связан с теорией:**

- **`/enriched`** → Раздел 3 (RunnablePassthrough). `RunnablePassthrough.assign()` добавляет `word_count`, `paragraph_count`, `timestamp` к input dict. Эти данные доступны в промпте через `{word_count}` и т.д. Исходные поля (`student_work`, `rubric`) сохраняются.
- **`/parallel`** → Раздел 4 (RunnableParallel). Две цепочки с разными рубриками (через `.partial()`) запускаются параллельно. `elapsed_seconds` в ответе позволяет убедиться, что время ≈ одному LLM-вызову, а не двум.
- **`/batch`** → Раздел 9 (batch/abatch). `chain.abatch()` обрабатывает N работ параллельно. `max_concurrency` из запроса управляет параллелизмом через `config`. `elapsed_seconds` показывает ускорение по сравнению с последовательной обработкой.
- **`/with-fallback`** → Раздел 6 (with_fallbacks). Primary (Sonnet) + fallback (Haiku). Оба обёрнуты в `with_structured_output()` — это критично, иначе fallback вернёт `AIMessage` вместо `AssessmentResponse`. При ошибке Sonnet автоматически используется Haiku.

### Шаг 3. Регистрация в `app/api/router.py`

```python
from fastapi import APIRouter

from app.api.v1 import assessment, chains, prompts, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(chains.router)
```

### Шаг 4. Тестирование

Запусти сервер:

```bash
uvicorn app.main:app --reload
```

**Тест 1 — Enriched assessment (RunnablePassthrough.assign):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chains/enriched \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Artificial intelligence is transforming the modern workplace.\n\nWhile automation threatens certain routine jobs, it simultaneously creates new roles in AI development and data science.\n\nStudies suggest that up to 47% of jobs may be automated within two decades.",
    "rubric": {
      "id": "essay", "name": "Essay Assessment",
      "criteria": [
        {"name": "Thesis", "description": "Clear thesis", "max_score": 25, "weight": 0.25},
        {"name": "Evidence", "description": "Use of evidence", "max_score": 25, "weight": 0.25},
        {"name": "Structure", "description": "Organization", "max_score": 20, "weight": 0.20},
        {"name": "Critical Thinking", "description": "Depth of analysis", "max_score": 20, "weight": 0.20},
        {"name": "Language", "description": "Grammar and style", "max_score": 10, "weight": 0.10}
      ]
    }
  }' | python -m json.tool
```

Ожидаемый результат: `metadata.word_count` > 0, `metadata.paragraph_count` = 3 (три абзаца через `\n\n`), `metadata.timestamp` содержит ISO-дату.

**Тест 2 — Parallel assessment (RunnableParallel):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chains/parallel \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "AI is reshaping how we work, learn, and communicate. The rapid advancement of machine learning has created both unprecedented opportunities and significant challenges for society.",
    "rubric_1": {
      "id": "essay", "name": "Academic Essay",
      "criteria": [
        {"name": "Thesis", "description": "Clear thesis with logical development", "max_score": 25, "weight": 0.25},
        {"name": "Evidence", "description": "Use of relevant evidence", "max_score": 25, "weight": 0.25},
        {"name": "Structure", "description": "Organization and flow", "max_score": 20, "weight": 0.20},
        {"name": "Critical Thinking", "description": "Depth of analysis", "max_score": 20, "weight": 0.20},
        {"name": "Language", "description": "Grammar and style", "max_score": 10, "weight": 0.10}
      ]
    },
    "rubric_2": {
      "id": "creative", "name": "Creative Writing",
      "criteria": [
        {"name": "Originality", "description": "Fresh ideas and unique perspective", "max_score": 30, "weight": 0.30},
        {"name": "Voice", "description": "Distinctive voice and style", "max_score": 25, "weight": 0.25},
        {"name": "Narrative", "description": "Engaging narrative structure", "max_score": 25, "weight": 0.25},
        {"name": "Language Craft", "description": "Skillful use of language", "max_score": 20, "weight": 0.20}
      ]
    }
  }' | python -m json.tool
```

Ожидаемый результат: `elapsed_seconds` ≈ время одного LLM-вызова (3-8 секунд), а не двух. Два набора оценок по разным критериям.

**Тест 3 — Batch assessment (abatch):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chains/batch \
  -H "Content-Type: application/json" \
  -d '{
    "works": [
      "AI is transforming the workplace through automation of routine tasks. Studies from MIT suggest 47% of jobs face automation risk. However, new roles in AI development are emerging.",
      "AI is good. It helps people. Some jobs will disappear but new ones will come. The end.",
      "The intersection of artificial intelligence and labor economics presents a nuanced challenge. While Frey and Osborne (2013) estimated 47% automation risk, subsequent analyses by Arntz et al. (2016) suggest only 9% of jobs are fully automatable. This discrepancy highlights the importance of task-level rather than occupation-level analysis."
    ],
    "rubric": {
      "id": "essay", "name": "Essay Assessment",
      "criteria": [
        {"name": "Thesis", "description": "Clear thesis", "max_score": 25, "weight": 0.25},
        {"name": "Evidence", "description": "Use of evidence", "max_score": 25, "weight": 0.25},
        {"name": "Structure", "description": "Organization", "max_score": 20, "weight": 0.20},
        {"name": "Critical Thinking", "description": "Analysis depth", "max_score": 20, "weight": 0.20},
        {"name": "Language", "description": "Grammar and style", "max_score": 10, "weight": 0.10}
      ]
    },
    "max_concurrency": 3
  }' | python -m json.tool
```

Ожидаемый результат: 3 результата в `results`. Оценки отражают качество: третья работа (с цитатами и нюансированным анализом) > первая > вторая. `elapsed_seconds` ≈ время одного вызова (благодаря `max_concurrency=3`).

**Тест 4 — Fallback assessment:**

```bash
curl -s -X POST http://localhost:8000/api/v1/chains/with-fallback \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Artificial intelligence is transforming the modern workplace in profound ways. While automation threatens certain routine jobs, it simultaneously creates new roles in AI development, data science, and human-AI collaboration.",
    "rubric": {
      "id": "essay", "name": "Essay Assessment",
      "criteria": [
        {"name": "Thesis", "description": "Clear thesis", "max_score": 25, "weight": 0.25},
        {"name": "Evidence", "description": "Use of evidence", "max_score": 25, "weight": 0.25},
        {"name": "Structure", "description": "Organization", "max_score": 20, "weight": 0.20},
        {"name": "Critical Thinking", "description": "Analysis depth", "max_score": 20, "weight": 0.20},
        {"name": "Language", "description": "Grammar and style", "max_score": 10, "weight": 0.10}
      ]
    }
  }' | python -m json.tool
```

Ожидаемый результат: `primary_model` = "claude-sonnet-4-20250514", `fallback_model` = "claude-haiku-4-20250414". В обычном режиме используется primary. Fallback активируется автоматически при ошибке primary (rate limit, timeout и т.д.).

---

## Чеклист самопроверки

Ответь на эти вопросы **своими словами**:

- [ ] Что такое Runnable protocol и зачем единый интерфейс? Назови 6 методов Runnable.
- [ ] Как оператор `|` работает под капотом? Что создаётся при `a | b | c`?
- [ ] В чём разница между `RunnablePassthrough()` и `RunnablePassthrough.assign()`?
- [ ] Когда использовать `RunnableParallel` vs просто два вызова `ainvoke`?
- [ ] Почему `RunnableLambda` нужен, если можно просто вызвать функцию? Что он даёт?
- [ ] Объясни разницу между `.with_fallbacks()` и `.with_retry()`. Когда что?
- [ ] Почему в FastAPI нужен `ainvoke`, а не `invoke`?
- [ ] Чем `batch` отличается от `RunnableParallel`? Когда какой использовать?
- [ ] Зачем `max_concurrency` в `abatch`? Что произойдёт без него при 100 inputs?
- [ ] Почему fallback-модель тоже должна быть обёрнута в `with_structured_output()`?

---

## Частые ошибки

### 1. invoke в async endpoint

```python
@router.post("")
async def assess(request: Request, chain: ChainDep):
    return chain.invoke({"student_work": request.student_work})
```

```python
@router.post("")
async def assess(request: Request, chain: ChainDep):
    return await chain.ainvoke({"student_work": request.student_work})
```

`invoke` в `async def` блокирует event loop. FastAPI запустит его в threadpool, но это менее эффективно, чем нативный `ainvoke`.

### 2. Потеря данных в цепочке

```python
preprocess = RunnableLambda(lambda x: {"student_work": x["student_work"][:1000]})
chain = preprocess | prompt | model
```

```python
preprocess = RunnableLambda(lambda x: {**x, "student_work": x["student_work"][:1000]})
chain = preprocess | prompt | model
```

Первый вариант теряет `rubric` и другие ключи. Всегда используй `{**x, ...}` для сохранения всех полей.

### 3. Fallback без structured output

```python
primary = llm.with_structured_output(Schema)
fallback_llm = ChatAnthropic(model="claude-haiku-4-20250414")
chain = primary.with_fallbacks([fallback_llm])
```

```python
primary = llm.with_structured_output(Schema)
fallback = fallback_llm.with_structured_output(Schema)
chain = primary.with_fallbacks([fallback])
```

Без `with_structured_output` на fallback, он вернёт `AIMessage` вместо Pydantic-объекта — TypeError в runtime.

### 4. batch() без ограничения concurrency

```python
results = await chain.abatch(hundred_inputs)
```

```python
results = await chain.abatch(hundred_inputs, config={"max_concurrency": 5})
```

100 параллельных запросов гарантированно вызовут rate limiting. Всегда ставь `max_concurrency` для production-кода.

### 5. RunnablePassthrough вместо RunnablePassthrough.assign

```python
chain = RunnablePassthrough(lambda x: len(x["text"].split())) | prompt | model
```

```python
chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split())
) | prompt | model
```

`RunnablePassthrough(func)` вызывает `func` как побочный эффект, но **не меняет данные**. `.assign()` — добавляет новые ключи.

### 6. Забыть partial для rubric в parallel

```python
parallel = RunnableParallel(
    result_1=prompt | structured_llm,
    result_2=prompt | structured_llm,
)
result = await parallel.ainvoke({"student_work": "...", "rubric": rubric_text})
```

```python
prompt_1 = prompt.partial(rubric=rubric_1_text)
prompt_2 = prompt.partial(rubric=rubric_2_text)

parallel = RunnableParallel(
    result_1=prompt_1 | structured_llm,
    result_2=prompt_2 | structured_llm,
)
result = await parallel.ainvoke({"student_work": "..."})
```

Без `.partial()` обе ветки получат одинаковую рубрику из input dict. Для разных рубрик — partial на каждом промпте.

---

## Что читать дальше

- [LangChain LCEL Conceptual Guide](https://python.langchain.com/docs/concepts/lcel/) — как устроен LCEL
- [LangChain Runnables How-to](https://python.langchain.com/docs/how_to/#runnables) — практические рецепты
- [RunnableParallel](https://python.langchain.com/docs/how_to/parallel/) — параллельное выполнение
- [Fallbacks](https://python.langchain.com/docs/how_to/fallbacks/) — отказоустойчивость
- [RunnablePassthrough](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) — полный API reference
- [Batch/Streaming](https://python.langchain.com/docs/concepts/runnables/#optimized-parallel-execution-batch) — batch и streaming

**Следующая тема:** [Тема 3: Structured Output](topic_03_structured_output.md) — как заставить LLM возвращать типизированные данные.

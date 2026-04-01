# Тема 2: LangChain Core + LCEL

> **Пререквизиты:** [Тема 1: Промпт-инжиниринг](topic_01_prompt_engineering.md)
> **Зависимости:** `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. Runnable protocol — единый интерфейс

LCEL (LangChain Expression Language) построен на одной абстракции: **Runnable**. Каждый компонент — промпт, модель, парсер, функция — реализует один и тот же интерфейс:

| Метод | Описание |
|-------|----------|
| `invoke(input)` | Синхронный вызов. Отправляет input, ждёт полный результат. |
| `ainvoke(input)` | Асинхронный invoke. Для async-кода. |
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

Пример:

```python
chain = prompt | structured_llm
```

Здесь 2 Runnables: `prompt` (ChatPromptTemplate) → messages, `structured_llm` (ChatAnthropic + structured output) → типизированный ответ.

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

### 11. Типовая структура chain

```python
def build_assessment_chain(llm: ChatAnthropic) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Please assess:\n\n{student_work}"),
    ]).partial(few_shot_good=..., few_shot_bad=...)

    structured_llm = llm.with_structured_output(AssessmentResponse)
    return prompt | structured_llm
```

Здесь 2 Runnables в цепочке:
1. `prompt` — `ChatPromptTemplate` → принимает `dict`, возвращает `list[BaseMessage]`
2. `structured_llm` — `ChatAnthropic` + structured output → принимает `list[BaseMessage]`, возвращает типизированный объект

### 12. invoke vs ainvoke

В async-контексте используй `ainvoke`:

```python
async def assess(text: str) -> AssessmentResponse:
    return await chain.ainvoke({"student_work": text})
```

Если использовать `invoke` внутри `async def` — заблокируешь event loop. Python запустит его в threadpool, но это менее эффективно, чем нативный `ainvoke`.

Правило: в `async def` → `ainvoke` / `abatch` / `astream`. В обычных функциях → `invoke` / `batch` / `stream`. Не смешивай sync- и async-вызовы.

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

## Практика

В этом разделе — 4 самостоятельных примера, каждый демонстрирует одну LCEL-концепцию. Код можно запускать в Jupyter-ноутбуке или как обычный Python-скрипт.

| Пример | Концепция | Что проверяем |
|---|---|---|
| 1 | RunnablePassthrough.assign() | Обогащение данных метаинформацией |
| 2 | RunnableParallel | Параллельная оценка по 2 рубрикам |
| 3 | chain.abatch() | Пакетная оценка нескольких работ |
| 4 | .with_fallbacks() | Отказоустойчивость с fallback-моделью |

### Общие зависимости

```bash
pip install langchain-core langchain-anthropic
```

### Пример 1. RunnablePassthrough.assign() — обогащение данных

`RunnablePassthrough.assign()` добавляет вычисляемые поля к входному dict, сохраняя все исходные ключи. Здесь мы добавляем `word_count`, `paragraph_count` и `timestamp` перед передачей в промпт.

```python
from datetime import UTC, datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel


class Assessment(BaseModel):
    score: int
    feedback: str


llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor. Consider the metadata when assessing."),
    (
        "human",
        "Word count: {word_count}\nParagraph count: {paragraph_count}\n"
        "Timestamp: {timestamp}\n\nAssess this text:\n\n{text}",
    ),
])

enrich = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split()),
    paragraph_count=lambda x: len([p for p in x["text"].split("\n\n") if p.strip()]),
    timestamp=lambda _: datetime.now(UTC).isoformat(),
)

chain = enrich | prompt | llm.with_structured_output(Assessment)

result = await chain.ainvoke({
    "text": (
        "Artificial intelligence is transforming the modern workplace.\n\n"
        "While automation threatens certain routine jobs, it simultaneously "
        "creates new roles in AI development and data science.\n\n"
        "Studies suggest that up to 47% of jobs may be automated within two decades."
    ),
})

print(f"Score: {result.score}")
print(f"Feedback: {result.feedback}")
```

Ожидаемый результат: `word_count`, `paragraph_count` и `timestamp` вычисляются автоматически через `assign()` и доступны в промпте как `{word_count}` и т.д. Исходное поле `text` сохраняется.

### Пример 2. RunnableParallel — параллельное выполнение

`RunnableParallel` запускает несколько цепочек одновременно на одном входе. Две цепочки с разными рубриками (через `.partial()`) работают параллельно — время выполнения ≈ одному вызову, а не сумме.

```python
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel


class Assessment(BaseModel):
    score: int
    feedback: str


llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
structured_llm = llm.with_structured_output(Assessment)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor. Use this rubric:\n{rubric}"),
    ("human", "Assess this text:\n\n{text}"),
])

prompt_academic = prompt.partial(
    rubric="Thesis (25%), Evidence (25%), Structure (20%), Critical Thinking (20%), Language (10%)"
)
prompt_creative = prompt.partial(
    rubric="Originality (30%), Voice (25%), Narrative (25%), Language Craft (20%)"
)

parallel = RunnableParallel(
    academic=prompt_academic | structured_llm,
    creative=prompt_creative | structured_llm,
)

start = time.monotonic()
results = await parallel.ainvoke({
    "text": (
        "AI is reshaping how we work, learn, and communicate. "
        "The rapid advancement of machine learning has created both "
        "unprecedented opportunities and significant challenges for society."
    ),
})
elapsed = time.monotonic() - start

print(f"Academic: score={results['academic'].score}, feedback={results['academic'].feedback}")
print(f"Creative: score={results['creative'].score}, feedback={results['creative'].feedback}")
print(f"Elapsed: {elapsed:.2f}s")
```

Ожидаемый результат: два набора оценок по разным критериям. `elapsed` ≈ время одного LLM-вызова (3-8 секунд), не двух — подтверждение параллельности.

### Пример 3. chain.abatch() — пакетная обработка

`abatch()` обрабатывает список входов параллельно. `max_concurrency` ограничивает число одновременных запросов к API.

```python
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class Assessment(BaseModel):
    score: int
    feedback: str


llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor. Assess briefly."),
    ("human", "Assess this text:\n\n{text}"),
])

chain = prompt | llm.with_structured_output(Assessment)

inputs = [
    {"text": "AI is transforming the workplace through automation of routine tasks. "
             "Studies from MIT suggest 47% of jobs face automation risk."},
    {"text": "AI is good. It helps people. The end."},
    {"text": "The intersection of artificial intelligence and labor economics presents "
             "a nuanced challenge. Frey and Osborne (2013) estimated 47% automation risk, "
             "yet Arntz et al. (2016) suggest only 9% of jobs are fully automatable."},
]

start = time.monotonic()
results = await chain.abatch(inputs, config={"max_concurrency": 3})
elapsed = time.monotonic() - start

for i, result in enumerate(results):
    print(f"Work {i + 1}: score={result.score}, feedback={result.feedback[:80]}...")

print(f"\nTotal: {len(results)} assessments in {elapsed:.2f}s")
```

Ожидаемый результат: 3 оценки с разными баллами (третья работа > первая > вторая). `elapsed` ≈ время одного вызова благодаря параллельности. Без `max_concurrency` при 100 inputs — 100 одновременных API-запросов, что гарантированно вызовет rate limiting.

### Пример 4. .with_fallbacks() — отказоустойчивость

`with_fallbacks()` автоматически переключается на запасную модель при ошибке основной. Обе модели обёрнуты в `with_structured_output()` — иначе fallback вернёт `AIMessage` вместо Pydantic-объекта.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class Assessment(BaseModel):
    score: int
    feedback: str


primary = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
backup = ChatAnthropic(model="claude-haiku-4-20250414", temperature=0)

primary_structured = primary.with_structured_output(Assessment)
backup_structured = backup.with_structured_output(Assessment)
reliable_llm = primary_structured.with_fallbacks([backup_structured])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing assessor."),
    ("human", "Assess this text:\n\n{text}"),
])

chain = prompt | reliable_llm

result = await chain.ainvoke({
    "text": (
        "Artificial intelligence is transforming the modern workplace. "
        "While automation threatens certain routine jobs, it simultaneously "
        "creates new roles in AI development and data science."
    ),
})

print(f"Score: {result.score}")
print(f"Feedback: {result.feedback}")
```

При нормальной работе используется primary (Sonnet). При ошибке (rate limit, timeout) — автоматически Haiku. Для production рекомендуется комбинировать с `.with_retry()`:

```python
chain = (
    prompt | primary_structured.with_retry(stop_after_attempt=2)
).with_fallbacks([
    prompt | backup_structured.with_retry(stop_after_attempt=2)
])
```

---

## Чеклист самопроверки

Ответь на эти вопросы **своими словами**:

- [ ] Что такое Runnable protocol и зачем единый интерфейс? Назови 6 методов Runnable.
- [ ] Как оператор `|` работает под капотом? Что создаётся при `a | b | c`?
- [ ] В чём разница между `RunnablePassthrough()` и `RunnablePassthrough.assign()`?
- [ ] Когда использовать `RunnableParallel` vs просто два вызова `ainvoke`?
- [ ] Почему `RunnableLambda` нужен, если можно просто вызвать функцию? Что он даёт?
- [ ] Объясни разницу между `.with_fallbacks()` и `.with_retry()`. Когда что?
- [ ] Почему в `async def` нужен `ainvoke`, а не `invoke`? Что произойдёт при смешивании?
- [ ] Чем `batch` отличается от `RunnableParallel`? Когда какой использовать?
- [ ] Зачем `max_concurrency` в `abatch`? Что произойдёт без него при 100 inputs?
- [ ] Почему fallback-модель тоже должна быть обёрнута в `with_structured_output()`?

---

## Частые ошибки

### 1. invoke в async-функции

```python
async def assess(text: str) -> Assessment:
    return chain.invoke({"text": text})
```

```python
async def assess(text: str) -> Assessment:
    return await chain.ainvoke({"text": text})
```

`invoke` в `async def` блокирует event loop. Python выполнит его синхронно, заморозив все остальные корутины. Используй `ainvoke` для нативной async-работы.

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

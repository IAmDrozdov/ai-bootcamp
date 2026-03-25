# Тема 2: LangChain Core + LCEL

> **Пререквизиты:** [Тема 1: Промпт-инжиниринг](topic_01_prompt_engineering.md)
> **Где в проекте:** `app/chains/assessment_chain.py`, `app/services/llm.py`, `app/dependencies.py`
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

В проекте:
```python
chain = prompt | structured_llm  # prompt → messages → model → AssessmentResponse
```

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

### 4. RunnableParallel

Запускает несколько Runnables **параллельно** на одном входе и собирает результаты в dict:

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    assessment=assessment_chain,
    metadata=metadata_chain,
)

result = chain.invoke({"student_work": "..."})
# result = {"assessment": AssessmentResponse(...), "metadata": {...}}
```

Под капотом: `RunnableParallel` запускает все ветки через `asyncio.gather()` (или `ThreadPoolExecutor` для синхронного кода). Реальный параллелизм, не sequential.

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

### 6. .with_fallback()

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

### 7. .with_retry()

Повторные попытки при ошибках с exponential backoff:

```python
chain = (prompt | model).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

Полезно для transient errors (rate limits, timeouts, network issues). Не помогает при ошибках в логике (неправильный промпт не станет правильным от повтора).

### 8. .bind()

Привязывает дополнительные аргументы к Runnable:

```python
model = ChatAnthropic(model="claude-sonnet-4-20250514")
model_with_tools = model.bind(tools=[...])
model_with_stop = model.bind(stop=["\n\n"])
```

`bind()` создаёт новый Runnable, который при вызове передаёт привязанные аргументы. Исходный Runnable не меняется (иммутабельность).

### 9. ChatModel vs LLM

| | ChatModel (BaseChatModel) | LLM (BaseLLM) |
|---|---|---|
| Вход | Список messages | Строка текста |
| Выход | AIMessage | Строка текста |
| Примеры | ChatAnthropic, ChatOpenAI | (legacy) OpenAI completion |
| API | Chat Completions | (deprecated) Completions |

В 2024+ **все современные модели** — это ChatModel. BaseLLM — legacy для старых completion-моделей. В новом коде используй только ChatModel.

---

## Ключевые концепции LangChain

### Как устроен chain в проекте

```python
# app/chains/assessment_chain.py
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

### invoke vs ainvoke в FastAPI

FastAPI — async framework. Используй `ainvoke`:

```python
@router.post("")
async def assess_work(request: AssessmentRequest, chain: ChainDep):
    result = await chain.ainvoke({"student_work": request.student_work, "rubric": rubric_text})
    return result
```

Если использовать `invoke` в async endpoint — заблокируешь event loop. FastAPI запустит его в threadpool, но это хуже, чем нативный `ainvoke`.

---

## Практические задания

### Задание 1: RunnablePassthrough

**Цель:** научиться обогащать входные данные chain без потери исходных полей.

**Файлы:** модификация `app/chains/assessment_chain.py`, новый `experiments/t2_passthrough.py`

**Критерии успеха:**
- Chain принимает `{"student_work": "...", "rubric": "..."}` и автоматически добавляет `timestamp` и `rubric_name`
- Эти метаданные доступны в промпте через `{timestamp}` и `{rubric_name}`
- Оценка работает как раньше, но теперь с метаданными

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Modify app/chains/assessment_chain.py to use RunnablePassthrough.assign().

Current chain: prompt | structured_llm
New chain: RunnablePassthrough.assign(timestamp=..., word_count=...) | prompt | structured_llm

Steps:
1. Import RunnablePassthrough from langchain_core.runnables
2. Import datetime
3. Add a RunnablePassthrough.assign() step before the prompt that:
   - Adds "timestamp": current ISO timestamp
   - Adds "word_count": number of words in student_work
4. Update the human message in ChatPromptTemplate to include: "Word count: {word_count}\nTimestamp: {timestamp}\n\nPlease assess the following student work:\n\n{student_work}"

Also create experiments/t2_passthrough.py that:
1. Imports and invokes the chain
2. Prints the input keys available to the prompt (to verify timestamp and word_count are added)
3. Runs an assessment and prints the result

Run with: python -m experiments.t2_passthrough
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the changes to app/chains/assessment_chain.py and experiments/t2_passthrough.py:

1. PASSTHROUGH: Is RunnablePassthrough.assign() used to add metadata fields?
2. COMPOSITION: Is the chain still a clean pipe: passthrough | prompt | structured_llm?
3. IMMUTABILITY: Does the original input dict still contain student_work and rubric? (assign should ADD, not replace)
4. PROMPT UPDATE: Does the ChatPromptTemplate reference the new variables {timestamp}, {word_count}?
5. EXPERIMENT: Does the experiment script demonstrate that the metadata is correctly added?

Common mistakes:
- Using RunnablePassthrough() instead of RunnablePassthrough.assign()
- Overwriting student_work or rubric instead of adding new fields
- Not updating the prompt template to use the new variables
- Lambda captures that break serialization

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: RunnableParallel

**Цель:** научиться запускать несколько chains параллельно и собирать результаты.

**Файлы:** `experiments/t2_parallel.py`

**Критерии успеха:**
- Одна работа оценивается по двум разным рубрикам параллельно
- Используется `RunnableParallel` (не два последовательных вызова)
- Результат — dict с двумя `AssessmentResponse`
- Время выполнения ≈ время одного вызова (не двух)

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t2_parallel.py that evaluates one essay against 2 rubrics in parallel.

1. Define two rubrics:
   - essay_rubric: Thesis & Argument (25), Evidence (25), Structure (20), Critical Thinking (20), Language (10)
   - creative_rubric: Originality (30), Voice & Style (25), Narrative Structure (25), Language Craft (20)

2. Build two separate chains, each with its own rubric in the prompt

3. Combine them using RunnableParallel:
   parallel_chain = RunnableParallel(
       essay_assessment=essay_chain,
       creative_assessment=creative_chain,
   )

4. Invoke the parallel chain with the same student_work

5. Measure and print execution time. Compare with sequential execution time.

6. Print both assessment results side by side.

Use existing project imports. Temperature=0 for determinism.
Run with: python -m experiments.t2_parallel
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t2_parallel.py:

1. PARALLEL EXECUTION: Is RunnableParallel actually used (not just two sequential ainvoke calls)?
2. TWO RUBRICS: Are two meaningfully different rubrics defined?
3. TIMING: Is execution time measured to demonstrate parallel speedup?
4. RESULTS: Are both assessment results printed clearly?
5. CHAIN STRUCTURE: Each sub-chain should be a complete prompt | structured_llm pipeline.

Expected: parallel execution should take roughly the time of ONE LLM call, not two. If parallel is ~2x slower than expected, check that ainvoke is used (not invoke).

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: RunnableLambda

**Цель:** научиться вставлять произвольную логику в LCEL chain.

**Файлы:** `experiments/t2_lambda.py`

**Критерии успеха:**
- Preprocessing функция: обрезает текст до N слов, считает количество слов и абзацев
- Postprocessing функция: добавляет метаданные к результату
- Обе обёрнуты в `RunnableLambda` и встроены в chain
- Chain работает корректно end-to-end

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t2_lambda.py demonstrating RunnableLambda for pre/post-processing.

1. Define a preprocessing function:
   - Takes dict with "student_work" and "rubric"
   - Truncates student_work to max 500 words
   - Adds "word_count", "paragraph_count", "is_truncated" to the dict
   - Returns the enriched dict

2. Define a postprocessing function:
   - Takes AssessmentResponse
   - Wraps it in a dict with metadata: {"assessment": result, "processed_at": timestamp, "model": model_name}

3. Build chain:
   RunnableLambda(preprocess) | prompt | structured_llm | RunnableLambda(postprocess)

4. Test with a long essay (>500 words) and a short one (<100 words)
5. Print: was text truncated? word count before/after? final result with metadata?

Use existing project imports.
Run with: python -m experiments.t2_lambda
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t2_lambda.py:

1. LAMBDA USAGE: Are RunnableLambda wrappers used for both pre and post-processing?
2. PREPROCESSING: Does it correctly truncate text, count words, count paragraphs?
3. POSTPROCESSING: Does it wrap the result with metadata?
4. CHAIN COMPOSITION: Is the full chain a single pipe expression?
5. TESTING: Is it tested with both long and short texts to verify truncation?

Common mistakes:
- Using a regular function call instead of RunnableLambda in the chain
- Preprocessing function doesn't return all required keys for the prompt
- Postprocessing expects wrong input type (should match structured_llm output)
- Not handling edge cases (empty text, single paragraph)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: Fallback

**Цель:** научиться строить отказоустойчивые chains с автоматическим переключением на запасную модель.

**Файлы:** `experiments/t2_fallback.py`

**Критерии успеха:**
- Основная модель: claude-sonnet, fallback: claude-haiku
- При ошибке основной модели автоматически вызывается haiku
- Эксперимент показывает, что fallback работает
- Логирование: какая модель ответила

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t2_fallback.py demonstrating .with_fallbacks().

1. Create two LLM instances:
   - primary: ChatAnthropic(model="claude-sonnet-4-20250514")
   - fallback: ChatAnthropic(model="claude-haiku-4-20250414")

2. Create a reliable model:
   reliable_llm = primary.with_fallbacks([fallback])

3. Build the assessment chain with reliable_llm

4. Demonstrate fallback works:
   - Normal call: should use primary (sonnet)
   - Simulate failure: create a RunnableLambda that raises an exception for the first call, then use with_fallbacks to show it falls back
   - Alternative: use a deliberately broken model config as primary

5. Add logging: print which model was actually used (check response metadata or use callbacks)

6. Compare output quality: run same assessment with sonnet-only, haiku-only, and sonnet-with-haiku-fallback

Use existing project imports. Temperature=0.
Run with: python -m experiments.t2_fallback
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t2_fallback.py:

1. FALLBACK SETUP: Is .with_fallbacks() used correctly on the LLM?
2. DEMONSTRATION: Can you see that fallback actually triggers on error?
3. COMPARISON: Is there a comparison between primary, fallback, and combined output?
4. LOGGING: Does the script indicate which model responded?
5. REALISTIC: Is the failure scenario realistic (rate limit, timeout, invalid model)?

Common mistakes:
- with_fallbacks on the chain instead of the model (both work, but different semantics)
- Not actually testing the fallback path (only testing happy path)
- Fallback model not configured with structured output
- Not comparing output quality between primary and fallback

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 5: Batch

**Цель:** научиться эффективно обрабатывать несколько входов параллельно.

**Файлы:** `experiments/t2_batch.py`

**Критерии успеха:**
- 3 разных эссе оцениваются через `chain.abatch()`
- Время batch ≈ время одного вызова (не трёх)
- Все 3 результата — валидные AssessmentResponse
- Сравнение с последовательным выполнением

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t2_batch.py demonstrating batch assessment.

1. Define 3 sample essays of different quality:
   - Strong essay (~200 words, clear thesis, citations, good structure)
   - Medium essay (~150 words, decent but lacks depth)
   - Weak essay (~100 words, no citations, vague thesis)

2. Build the assessment chain using existing project code

3. Run assessments:
   a. Sequential: 3x await chain.ainvoke() — measure total time
   b. Batch: await chain.abatch([input1, input2, input3]) — measure total time

4. Print comparison:
   - Sequential time vs batch time
   - Speedup factor
   - All 3 assessment results (scores should differ based on essay quality)

5. Try batch with max_concurrency parameter: abatch(inputs, config={"max_concurrency": 2})

Use existing project imports. Temperature=0.
Run with: python -m experiments.t2_batch
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t2_batch.py:

1. BATCH USAGE: Is chain.abatch() used (not just a loop of ainvoke)?
2. TIMING: Are both sequential and batch times measured for comparison?
3. ESSAYS: Are the 3 essays meaningfully different in quality?
4. RESULTS: Do the scores reflect essay quality (strong > medium > weak)?
5. CONCURRENCY: Is max_concurrency demonstrated?

Expected: batch should be ~3x faster than sequential (or close, depending on API rate limits). If batch is slower, check for: synchronous invoke, missing await, rate limiting.

Common mistakes:
- Using invoke instead of ainvoke in sequential comparison
- Not awaiting abatch result
- All 3 essays being identical (defeats the purpose)
- Not measuring time correctly (include only LLM call, not setup)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Что такое Runnable protocol и зачем единый интерфейс? Назови 4 метода Runnable.
- [ ] Как оператор `|` работает под капотом? Что создаётся при `a | b | c`?
- [ ] В чём разница между `RunnablePassthrough()` и `RunnablePassthrough.assign()`?
- [ ] Когда использовать `RunnableParallel` vs просто два вызова `ainvoke`?
- [ ] Почему `RunnableLambda` нужен, если можно просто вызвать функцию?
- [ ] Объясни разницу между `.with_fallback()` и `.with_retry()`. Когда что?
- [ ] Почему в FastAPI нужен `ainvoke`, а не `invoke`?

---

## Частые ошибки

### 1. invoke в async endpoint

```python
# Плохо: блокирует event loop
@router.post("")
async def assess(request: Request, chain: ChainDep):
    return chain.invoke({"student_work": request.student_work})

# Правильно: async вызов
@router.post("")
async def assess(request: Request, chain: ChainDep):
    return await chain.ainvoke({"student_work": request.student_work})
```

### 2. Потеря данных в цепочке

```python
# Плохо: RunnableLambda теряет rubric
preprocess = RunnableLambda(lambda x: {"student_work": x["student_work"][:1000]})
chain = preprocess | prompt | model  # prompt ожидает {rubric}, но его нет

# Правильно: сохранять все ключи
preprocess = RunnableLambda(lambda x: {**x, "student_work": x["student_work"][:1000]})
```

### 3. Fallback без structured output

```python
# Плохо: primary с structured output, fallback без
primary = llm.with_structured_output(Schema)
fallback_llm = ChatAnthropic(model="claude-haiku-4-20250414")
chain = primary.with_fallbacks([fallback_llm])  # fallback вернёт AIMessage, не Schema

# Правильно: structured output на обоих
fallback = fallback_llm.with_structured_output(Schema)
chain = primary.with_fallbacks([fallback])
```

### 4. batch() без ограничения concurrency

```python
# Потенциальная проблема: 100 параллельных запросов = rate limit
results = await chain.abatch(hundred_inputs)

# Лучше: ограничить параллелизм
results = await chain.abatch(hundred_inputs, config={"max_concurrency": 5})
```

---

## Что читать дальше

- [LangChain LCEL Conceptual Guide](https://python.langchain.com/docs/concepts/lcel/) — как устроен LCEL
- [LangChain Runnables How-to](https://python.langchain.com/docs/how_to/#runnables) — практические рецепты
- [RunnableParallel](https://python.langchain.com/docs/how_to/parallel/) — параллельное выполнение
- [Fallbacks](https://python.langchain.com/docs/how_to/fallbacks/) — отказоустойчивость

**Следующая тема:** [Тема 3: Structured Output](topic_03_structured_output.md) — как заставить LLM возвращать типизированные данные.

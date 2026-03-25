# Тема 3: Structured Output

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 2](topic_02_langchain_lcel.md)
> **Где в проекте:** `app/schemas/assessment.py`, `app/chains/assessment_chain.py`
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `pydantic`

---

## Теория

### 1. Проблема: LLM возвращает текст

LLM — это генератор текста. Он возвращает строку. Но твоему коду нужны **типизированные данные**: числа, списки, вложенные объекты. Между "LLM написал `score: 85`" и `result.score == 85` — пропасть парсинга, валидации и обработки ошибок.

Два подхода к решению:
1. **API-level constraint** (`with_structured_output`) — модель ограничена на уровне API, она *обязана* вернуть валидный JSON по схеме.
2. **Prompt-level instruction** (`PydanticOutputParser`) — ты просишь модель вернуть JSON и парсишь текст. Модель может ошибиться.

### 2. with_structured_output() — как это работает

Когда ты вызываешь `llm.with_structured_output(MySchema)`, LangChain под капотом:

1. **Конвертирует Pydantic-схему в JSON Schema** — каждое поле, его тип, description, constraints.
2. **Отправляет схему через tool calling API** — модель "вызывает инструмент" с аргументами, соответствующими схеме. Это не обычный text completion — API физически ограничивает вывод валидным JSON.
3. **Парсит ответ** — JSON из tool call → Pydantic model → валидация типов.

```python
structured_llm = llm.with_structured_output(AssessmentResponse)
result = await structured_llm.ainvoke(messages)
# result — это AssessmentResponse, не AIMessage
```

#### Почему это надёжнее

Tool calling API использует **constrained decoding**: модель генерирует токены, но API отбрасывает те, которые нарушают JSON-схему. Если поле `score` — int, модель физически не может написать `"excellent"`. Это работает на уровне токен-генерации, не пост-обработки.

#### Ограничения

- Не все модели поддерживают tool calling (Claude, GPT-4 — да; некоторые open-source — нет).
- Нельзя стримить structured output (нужен полный JSON для парсинга).
- Сложные вложенные схемы могут вызывать ошибки у менее мощных моделей.

### 3. PydanticOutputParser — альтернативный подход

Работает с **любой** моделью, включая те, что не поддерживают tool calling:

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=AssessmentResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return your answer in the following JSON format:\n{format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
```

Что происходит:
1. `parser.get_format_instructions()` генерирует текстовое описание JSON-формата с примером.
2. Это описание вставляется в промпт.
3. Модель генерирует текст, который (надеемся) содержит валидный JSON.
4. `parser.invoke(ai_message)` извлекает JSON из текста и валидирует через Pydantic.

#### Проблемы

- Модель может добавить текст до/после JSON ("Here is my response: ```json...```").
- Модель может вернуть невалидный JSON (пропущенная запятая, неправильный тип).
- Модель может игнорировать format instructions при длинном контексте.

### 4. Pydantic Field(description=...) — подсказки для LLM

Описания полей в Pydantic-схеме — это **не просто документация**. Они передаются модели как часть JSON Schema и напрямую влияют на качество ответа:

```python
class AssessmentResponse(BaseModel):
    overall_score: int = Field(description="Total score across all criteria")
    summary: str = Field(description="Brief overall assessment summary, 2-3 sentences")
    strengths: list[str] = Field(description="Key strengths of the work, minimum 2 items")
```

Чем точнее описания — тем лучше модель понимает, что от неё ждут:
- `description="Score"` — расплывчато, модель может интерпретировать по-разному.
- `description="Total score 0-100, sum of all criteria scores"` — модель знает диапазон и способ подсчёта.

### 5. JsonOutputParser — когда нет Pydantic-схемы

Для случаев, когда ты хочешь JSON, но не хочешь жёсткую схему:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser  # вернёт dict, не Pydantic model
```

Полезно для: прототипирования, работы с произвольными JSON-ответами, когда схема ещё не определена.

### 6. OutputFixingParser — автоматическое исправление

Если парсер получил невалидный JSON — он вызывает LLM ещё раз, передавая ошибку, и просит исправить:

```python
from langchain.output_parsers import OutputFixingParser

parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
```

Это **retry через LLM**: "Ты вернул невалидный JSON, вот ошибка: ..., исправь". Работает, но удваивает latency и стоимость при ошибках.

---

## Ключевые концепции LangChain

### with_structured_output — методы

```python
# Метод 1: Pydantic model (рекомендуется)
structured_llm = llm.with_structured_output(AssessmentResponse)

# Метод 2: JSON Schema dict
structured_llm = llm.with_structured_output({
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
})

# Метод 3: с include_raw=True (получить и сырой ответ, и распарсенный)
structured_llm = llm.with_structured_output(AssessmentResponse, include_raw=True)
result = await structured_llm.ainvoke(messages)
# result = {"raw": AIMessage(...), "parsed": AssessmentResponse(...), "parsing_error": None}
```

`include_raw=True` полезен для дебага: можно увидеть, что модель "хотела сказать" до парсинга.

### Разница подходов — сводная таблица

| | with_structured_output | PydanticOutputParser |
|---|---|---|
| Механизм | Tool calling API | Текстовые инструкции в промпте |
| Надёжность | Высокая (~99%+) | Средняя (~90-95%) |
| Совместимость | Claude, GPT-4, Gemini | Любая модель |
| Streaming | Нет (нужен полный ответ) | Нет (нужен полный текст) |
| Стоимость | Те же токены | Больше токенов (format instructions) |
| Контроль | Схема через API | Схема в тексте промпта |

---

## Практические задания

### Задание 1: Сравнение подходов

**Цель:** эмпирически измерить надёжность двух подходов к structured output.

**Файлы:** `experiments/t3_structured_comparison.py`

**Критерии успеха:**
- Одна оценка выполняется 10 раз каждым способом
- Считается: сколько из 10 парсятся без ошибок
- with_structured_output должен быть надёжнее (≥ 9/10)
- Результаты в табличном формате

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t3_structured_comparison.py comparing with_structured_output vs PydanticOutputParser.

1. Import AssessmentResponse from app.schemas.assessment and the prompt from app.prompts.templates

2. Approach A — with_structured_output:
   - chain_a = prompt | llm.with_structured_output(AssessmentResponse)
   - Run 10 times with ainvoke, count successes (no exceptions)

3. Approach B — PydanticOutputParser:
   - Create PydanticOutputParser(pydantic_object=AssessmentResponse)
   - Modify the prompt to include parser.get_format_instructions() in the system message
   - chain_b = prompt_with_instructions | llm | parser
   - Run 10 times with ainvoke, wrap in try/except, count successes

4. Print comparison table:
   Approach | Successes | Failures | Success Rate
   structured_output | 10 | 0 | 100%
   pydantic_parser    | 9  | 1 | 90%

5. For failures, print the error message to understand what went wrong

Use temperature=0.3. Same essay and rubric for all runs.
Run with: python -m experiments.t3_structured_comparison
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t3_structured_comparison.py:

1. FAIR COMPARISON: Same prompt content, same temperature, same essay for both approaches?
2. ERROR HANDLING: Does approach B (PydanticOutputParser) have proper try/except to count failures?
3. SAMPLE SIZE: At least 10 runs per approach?
4. PARSER SETUP: Does approach B correctly use parser.get_format_instructions() in the prompt?
5. RESULTS: Clear table comparing success rates?

Common mistakes:
- Not wrapping approach B in try/except (exceptions crash the script)
- Using different prompts for A and B (invalidates comparison)
- format_instructions not actually inserted into the prompt
- Not using .partial() for format_instructions

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Обогащение схемы

**Цель:** научиться проектировать Pydantic-схемы, которые направляют LLM к лучшим ответам.

**Файлы:** модификация `app/schemas/assessment.py`, `experiments/t3_enriched_schema.py`

**Критерии успеха:**
- AssessmentResponse обогащён полями: `confidence: float` (0-1), `reasoning: str`
- Field descriptions точно описывают ожидания
- Модель заполняет новые поля осмысленно
- confidence коррелирует со "сложностью" оценки

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Modify app/schemas/assessment.py and create experiments/t3_enriched_schema.py.

PART 1: Update AssessmentResponse in app/schemas/assessment.py
Add two new fields:
- confidence: float = Field(description="Confidence level 0.0-1.0 in the assessment accuracy. Lower for ambiguous works, higher for clearly strong or weak submissions.")
- reasoning: str = Field(description="Step-by-step reasoning process: for each criterion, explain what was identified, analyzed, and how the score was determined. This should show the assessment thought process.")

PART 2: Create experiments/t3_enriched_schema.py
1. Test the enriched schema with 3 essays of different clarity:
   - Clear strong essay (should get high confidence)
   - Clear weak essay (should get high confidence)
   - Ambiguous/mixed essay (should get lower confidence)
2. Print for each: overall_score, confidence, reasoning (first 300 chars)
3. Verify: confidence is 0-1, reasoning is non-empty and substantive

Run with: python -m experiments.t3_enriched_schema
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/schemas/assessment.py and experiments/t3_enriched_schema.py:

1. SCHEMA: Are confidence (float) and reasoning (str) added with descriptive Field descriptions?
2. FIELD DESCRIPTIONS: Do descriptions guide the LLM? (e.g., "lower for ambiguous" helps calibrate confidence)
3. EXPERIMENT: Are 3 different essay types tested (strong, weak, ambiguous)?
4. CORRELATION: Does confidence vary meaningfully across essays? (If all 0.9, descriptions need work)
5. REASONING: Is the reasoning field substantive (not just "I evaluated the criteria")?
6. BACKWARD COMPAT: Does the existing chain still work with the updated schema?

Common mistakes:
- confidence field without proper bounds description (model returns >1.0)
- reasoning field too vague, model writes generic text
- Not testing that existing API endpoints still work after schema change

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: Обработка ошибок

**Цель:** научиться обрабатывать ситуации, когда structured output не удаётся.

**Файлы:** `experiments/t3_error_handling.py`

**Критерии успеха:**
- Retry-логика: при ошибке парсинга — повторный запрос с уточнённым промптом
- Максимум 3 попытки
- Логирование каждой попытки
- Fallback: если все попытки провалились — возвращается сообщение об ошибке

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t3_error_handling.py demonstrating robust error handling for structured output.

1. Create a retry wrapper function:
   async def assess_with_retry(chain, input_data, max_retries=3) -> AssessmentResponse | None:
   - Try chain.ainvoke(input_data)
   - On OutputParserException or ValidationError: log the error, modify input to include error context, retry
   - On success: return result
   - After max_retries: return None with error log

2. Create a chain using PydanticOutputParser (intentionally less reliable than with_structured_output)

3. Demonstrate:
   - Normal case: first attempt succeeds
   - Edge case: very short input that might confuse the model
   - Compare: same inputs with with_structured_output vs PydanticOutputParser + retry

4. Also demonstrate OutputFixingParser:
   - Create an OutputFixingParser that wraps PydanticOutputParser
   - Show how it auto-corrects malformed JSON

5. Print: attempt number, success/failure, error details for each retry

Run with: python -m experiments.t3_error_handling
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t3_error_handling.py:

1. RETRY LOGIC: Is there a proper retry loop with max attempts?
2. ERROR TYPES: Does it handle OutputParserException and ValidationError?
3. LOGGING: Is each attempt logged with attempt number and error details?
4. FIXING PARSER: Is OutputFixingParser demonstrated as an alternative?
5. COMPARISON: Does it compare reliability of with_structured_output vs parser + retry?
6. GRACEFUL FAILURE: What happens after max retries? Does it return None/error instead of crashing?

Common mistakes:
- Catching too broad exceptions (bare except: instead of specific errors)
- Not passing error context to retry attempt
- OutputFixingParser without its own LLM instance
- No limit on retries (infinite loop on persistent errors)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Объясни разницу между with_structured_output и PydanticOutputParser. Когда выбрать какой?
- [ ] Как constrained decoding ограничивает вывод модели на уровне токенов?
- [ ] Почему Field(description=...) влияет на качество ответа LLM?
- [ ] Что делает include_raw=True и зачем это нужно?
- [ ] Как OutputFixingParser исправляет невалидный JSON?
- [ ] Почему нельзя стримить structured output?

---

## Частые ошибки

### 1. Pydantic-схема без описаний

```python
# Плохо: модель не понимает, что именно ожидается
class Result(BaseModel):
    score: int
    text: str

# Хорошо: описания направляют модель
class Result(BaseModel):
    score: int = Field(description="Overall score 0-100, sum of all criteria")
    text: str = Field(description="Detailed feedback, 3-5 sentences per criterion")
```

### 2. Слишком сложная вложенная схема

```python
# Может сломать менее мощные модели
class Deep(BaseModel):
    level1: list[dict[str, list[SubModel]]]

# Лучше: плоская структура
class Flat(BaseModel):
    items: list[SimpleItem]
```

### 3. Не обрабатывать ошибки парсинга

```python
# Плохо: при ошибке — 500 Internal Server Error
result = await chain.ainvoke(data)

# Хорошо: graceful handling
try:
    result = await chain.ainvoke(data)
except (OutputParserException, ValidationError) as e:
    logger.error(f"Parse failed: {e}")
    result = await retry_or_fallback(data)
```

---

## Что читать дальше

- [LangChain Structured Output](https://python.langchain.com/docs/concepts/structured_outputs/) — концепция
- [How to use with_structured_output](https://python.langchain.com/docs/how_to/structured_output/) — практика
- [Pydantic Field Types](https://docs.pydantic.dev/latest/concepts/fields/) — продвинутые поля
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — как tool calling работает в Claude

**Следующая тема:** [Тема 4: Streaming](topic_04_streaming.md) — как отдавать ответ в реальном времени.

# Тема 3: Structured Output — типизированные ответы LLM

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 2](topic_02_langchain_lcel.md)
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `langchain`, `pydantic`

---

## Теория

### 1. Проблема: LLM генерирует текст, а коду нужны типы

LLM — генератор последовательностей токенов. На выходе — строка. Но backend-приложению нужны типизированные данные: `int`, `float`, `list[str]`, вложенные объекты с валидацией. Между «модель написала `score: 85`» и `result.score == 85` — пропасть, которую нужно преодолеть.

Без structured output типичный pipeline выглядит так:

```python
response = await llm.ainvoke(messages)
text = response.content
import re
match = re.search(r'"score":\s*(\d+)', text)
score = int(match.group(1)) if match else None
```

Проблемы этого подхода:

1. **Хрупкий парсинг** — regex ломается, если модель добавит пробел, перенос строки или изменит формат.
2. **Нет валидации типов** — `"score": "eighty-five"` пройдёт regex, но сломает код ниже по стеку.
3. **Нет гарантии полноты** — модель может пропустить поля или добавить лишние.
4. **Невоспроизводимость** — при temperature > 0 формат может меняться от вызова к вызову.

Эмпирические данные: при наивном парсинге текстового ответа LLM без каких-либо инструкций по формату, около 60-70% ответов удаётся корректно распарсить. Остальные содержат неожиданный формат, дополнительный текст, или нарушения JSON-синтаксиса.

Два фундаментальных подхода к решению:

| Характеристика | API-level constraint | Prompt-level instruction |
|---|---|---|
| Метод | `with_structured_output()` | `PydanticOutputParser` |
| Где ограничение | На уровне API / token generation | В тексте промпта |
| Надёжность | ~99.5%+ | ~85-95% |
| Совместимость | Claude, GPT-4, Gemini | Любая модель |
| Дополнительные токены | Нет | +200-500 (format instructions) |
| Стриминг | Нет | Нет |

### 2. with_structured_output() — механика на уровне токенов

Когда вызывается `llm.with_structured_output(MySchema)`, LangChain выполняет три этапа:

**Этап 1: Конвертация Pydantic → JSON Schema**

```python
from pydantic import BaseModel, Field


class AssessmentResponse(BaseModel):
    overall_score: int = Field(description="Total score across all criteria")
    summary: str = Field(description="Brief overall assessment summary")
    strengths: list[str] = Field(description="Key strengths of the work")
    improvements: list[str] = Field(description="Suggested improvements")


print(AssessmentResponse.model_json_schema())
```

Результат — JSON Schema, который описывает каждое поле, его тип, ограничения и description:

```json
{
  "type": "object",
  "properties": {
    "overall_score": {
      "type": "integer",
      "description": "Total score across all criteria"
    },
    "summary": {
      "type": "string",
      "description": "Brief overall assessment summary"
    }
  },
  "required": ["overall_score", "summary", "..."]
}
```

**Этап 2: Отправка через Tool Calling API**

JSON Schema упаковывается в tool definition и отправляется провайдеру. Для Anthropic это выглядит так:

```json
{
  "tools": [{
    "name": "AssessmentResponse",
    "description": "Structured output schema for LLM assessment.",
    "input_schema": { "...JSON Schema..." }
  }],
  "tool_choice": {"type": "tool", "name": "AssessmentResponse"}
}
```

Параметр `tool_choice` с конкретным именем инструмента **принуждает** модель вызвать именно этот tool. Модель не может проигнорировать его и вернуть обычный текст.

**Этап 3: Constrained Decoding**

На уровне token generation API применяет constrained decoding — алгоритм, который модифицирует вероятности следующего токена на основе текущего состояния JSON:

1. Модель генерирует первый токен `{` (единственный допустимый).
2. Следующий токен — `"` (начало первого ключа по схеме).
3. Далее — `o`, `v`, `e`, `r`, `a`, `l`, `l`, `_`, `s`, `c`, `o`, `r`, `e` (имя поля из схемы).
4. После `:` — только цифры (поле `int`), например `8`, `5`.
5. Запятая, следующее поле.

На каждом шаге токены, нарушающие JSON Schema, получают вероятность 0. Если поле `score` — int, модель **физически не может** сгенерировать `"excellent"` — эти токены заблокированы mask-ой.

**Параметр method**

`with_structured_output` поддерживает два метода:

| Метод | Механизм | Когда использовать |
|---|---|---|
| `"function_calling"` (default) | Tool calling API | Рекомендуется для всех моделей с поддержкой tools |
| `"json_mode"` | Системное указание + JSON parsing | Когда tool calling недоступен |

```python
structured_llm = llm.with_structured_output(
    AssessmentResponse,
    method="function_calling",
)
```

В режиме `"json_mode"` LangChain добавляет инструкцию в system prompt и указывает API генерировать JSON. Это менее надёжно, чем tool calling, но работает с моделями без поддержки tools.

**Параметр include_raw**

При `include_raw=True` метод возвращает не Pydantic-объект, а словарь с тремя ключами:

```python
structured_llm = llm.with_structured_output(AssessmentResponse, include_raw=True)
result = await structured_llm.ainvoke(messages)

result["raw"]            # AIMessage с tool_calls
result["parsed"]         # AssessmentResponse или None
result["parsing_error"]  # Exception или None
```

Это незаменимо при отладке: видно, что модель реально вернула, как это было распарсено, и какая ошибка возникла (если она была).

### 3. PydanticOutputParser — текстовый подход

PydanticOutputParser работает на уровне промпта — он генерирует текстовое описание ожидаемого формата и вставляет его в системное сообщение:

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
instructions = parser.get_format_instructions()
```

Результат `get_format_instructions()` — блок текста вида:

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.

Here is the output schema:
{"properties": {"overall_score": {"description": "Total score across all criteria", "title": "Overall Score", "type": "integer"}, ...}, "required": [...]}
```

Этот текст вставляется в промпт через `.partial()`:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assessor.\n\n{format_instructions}"),
    ("human", "{student_work}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = await chain.ainvoke({"student_work": "..."})
```

**Как работает парсинг:**

1. Модель генерирует текст, содержащий JSON (иногда обёрнутый в markdown-блок ` ```json...``` `).
2. Парсер извлекает JSON из текста регулярным выражением.
3. JSON валидируется через `AssessmentResponse.model_validate(data)`.
4. Если валидация успешна — возвращается Pydantic-объект.

**Режимы отказа (от частых к редким):**

1. **Дополнительный текст** (~10% случаев) — модель добавляет «Here is my assessment:» перед JSON. Парсер обычно справляется с этим, извлекая JSON из текста.
2. **Невалидный JSON** (~3-5%) — пропущенная запятая, незакрытая кавычка. Особенно при длинных ответах.
3. **Неправильные типы** (~2-3%) — `"score": "85"` вместо `"score": 85`. Pydantic V2 обычно конвертирует это автоматически в lax-режиме.
4. **Пропущенные обязательные поля** (~1-2%) — модель «забывает» вернуть поле, особенно при длинном контексте.
5. **Полное игнорирование формата** (<1%) — модель возвращает обычный текст без JSON. Чаще при очень длинном контексте, когда format instructions «выталкиваются» из окна внимания.

### 4. JsonOutputParser — гибкий вариант без схемы

Для прототипирования или случаев, когда схема не определена:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser
result = await chain.ainvoke(data)
```

Результат — `dict`, не Pydantic-модель. Нет валидации типов, нет автодополнения в IDE, нет гарантий структуры.

Можно передать Pydantic-модель для генерации format instructions без валидации:

```python
parser = JsonOutputParser(pydantic_object=AssessmentResponse)
instructions = parser.get_format_instructions()
```

В этом случае format instructions будут содержать описание схемы, но парсер вернёт `dict`, а не `AssessmentResponse`.

| Критерий | PydanticOutputParser | JsonOutputParser |
|---|---|---|
| Возвращаемый тип | `BaseModel` | `dict` |
| Валидация типов | Да (Pydantic V2) | Нет |
| format_instructions | По JSON Schema модели | По JSON Schema или пустые |
| Ошибки парсинга | `OutputParserException` | `OutputParserException` |
| Когда использовать | Production, стабильная схема | Прототипы, динамические ответы |

### 5. OutputFixingParser — автоисправление через LLM

Если парсер получил невалидный JSON, OutputFixingParser автоматически вызывает LLM повторно с сообщением об ошибке:

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
```

**Внутренний механизм:**

1. `fixing_parser.parse(malformed_text)` вызывает `base_parser.parse(malformed_text)`.
2. Парсинг падает с `OutputParserException`, содержащим текст ошибки.
3. OutputFixingParser формирует новый промпт: «Вот невалидный ответ: `{malformed_text}`. Ошибка: `{error}`. Исправь.»
4. LLM генерирует исправленный JSON.
5. `base_parser.parse(fixed_text)` пытается распарсить исправление.

**Последствия:**

| Метрика | Без fixing | С fixing |
|---|---|---|
| Latency (успех) | 2-5 сек | 2-5 сек (без изменений) |
| Latency (ошибка) | Exception | 4-10 сек (double call) |
| Стоимость (ошибка) | 0 (failed) | 2x токенов |
| Success rate | ~90% | ~97% |

Используйте OutputFixingParser, когда: модель не поддерживает tool calling, ошибки формата редки но критичны, и допустимо увеличение latency при ошибках. Не используйте, если: нужна предсказуемая latency, бюджет на токены ограничен, или модель поддерживает `with_structured_output`.

### 6. Pydantic Field() — управление поведением LLM через схему

Описания полей (`description`) — не просто документация. Они передаются модели как часть JSON Schema и **напрямую влияют** на качество ответа.

**Сравнение с описаниями и без:**

```python
class BadSchema(BaseModel):
    score: int
    text: str
    items: list[str]

class GoodSchema(BaseModel):
    score: int = Field(
        description="Total score 0-100, computed as sum of all criterion scores"
    )
    text: str = Field(
        description="Detailed feedback, 3-5 sentences per criterion, referencing specific parts of the work"
    )
    items: list[str] = Field(
        description="List of 2-5 specific, actionable improvement suggestions"
    )
```

При использовании `BadSchema` модель сама интерпретирует, что значит `score` — это может быть 0-10, 0-100, или даже процент. `GoodSchema` задаёт диапазон, метод подсчёта и ожидания по формату.

**Числовые ограничения через Field:**

```python
class StrictScore(BaseModel):
    score: int = Field(ge=0, le=100, description="Score 0-100")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level")
    feedback: str = Field(min_length=50, max_length=2000, description="Detailed feedback")
```

Ограничения `ge`, `le`, `min_length`, `max_length` попадают в JSON Schema и влияют на constrained decoding. При tool calling API физически не позволит модели вернуть `score: 150` — это отклоняется на уровне schema validation.

**Параметр examples:**

```python
class Assessment(BaseModel):
    strengths: list[str] = Field(
        description="Key strengths",
        examples=[["Clear thesis statement", "Strong evidence from peer-reviewed sources"]]
    )
```

`examples` попадает в JSON Schema в поле `examples` и служит few-shot подсказкой для модели прямо в структуре схемы.

**Генерация JSON Schema:**

```python
schema = StrictScore.model_json_schema()
```

Результат:

```json
{
  "properties": {
    "score": {
      "description": "Score 0-100",
      "maximum": 100,
      "minimum": 0,
      "title": "Score",
      "type": "integer"
    },
    "confidence": {
      "description": "Confidence level",
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Confidence",
      "type": "number"
    }
  },
  "required": ["score", "confidence", "feedback"]
}
```

Всё, что вы указали в `Field()`, транслируется в JSON Schema и передаётся модели.

### 7. Конфигурация BaseModel для LLM-ответов

Pydantic V2 предоставляет `ConfigDict` для тонкой настройки поведения модели:

```python
from pydantic import BaseModel, ConfigDict

class StrictAssessment(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    score: int
    summary: str
```

| Параметр | Значение | Эффект для LLM output |
|---|---|---|
| `strict=True` | Запрет implicit coercion | `"85"` → ValidationError (не конвертируется в int) |
| `strict=False` (default) | Lax mode | `"85"` → `85` (автоконвертация) |
| `extra="forbid"` | Запрет доп. полей | Если модель вернёт лишнее поле — ошибка |
| `extra="ignore"` (default) | Игнорирование доп. полей | Лишние поля молча отбрасываются |
| `extra="allow"` | Принятие доп. полей | Лишние поля сохраняются в `model_extra` |
| `frozen=True` | Immutable instance | Нельзя изменить поля после создания |

Для LLM-ответов рекомендуется `strict=False` (default) и `extra="ignore"` — это наиболее устойчивая конфигурация, прощающая мелкие отклонения модели от схемы.

### 8. Дерево решений: выбор подхода

```
Модель поддерживает tool calling?
├── Да → with_structured_output() [рекомендуется]
│   ├── Нужен raw ответ для дебага? → include_raw=True
│   └── Нужен только parsed? → include_raw=False (default)
└── Нет
    ├── Нужна типизация? → PydanticOutputParser
    │   ├── Ошибки критичны? → + OutputFixingParser
    │   └── Ошибки допустимы? → PydanticOutputParser solo
    └── Прототип / динамика? → JsonOutputParser
```

---

## Справочник API

### with_structured_output()

**Описание:** Метод `BaseChatModel`, который оборачивает LLM для возврата типизированных данных вместо `AIMessage`. Использует tool calling API или JSON mode для constrained generation. Рекомендуемый подход для моделей с поддержкой tool calling (Claude, GPT-4, Gemini).

```python
def with_structured_output(
    schema: type[BaseModel] | dict,       # Pydantic модель или JSON Schema dict
    *,
    method: str = "function_calling",      # "function_calling" | "json_mode"
    include_raw: bool = False,             # вернуть dict с raw/parsed/parsing_error
    **kwargs,                              # доп. параметры для конкретного провайдера
) -> Runnable
```

**Основные методы возвращаемого Runnable:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `invoke(input)` | `dict \| list[BaseMessage]` | `BaseModel \| dict` | Синхронный вызов |
| `ainvoke(input)` | `dict \| list[BaseMessage]` | `BaseModel \| dict` | Асинхронный вызов |
| `batch(inputs)` | `list[dict]` | `list[BaseModel \| dict]` | Параллельное выполнение |

При `include_raw=True` выход — `dict` с ключами `"raw"`, `"parsed"`, `"parsing_error"`.

**Пример использования:**

```python
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(ge=1, le=10, description="Rating 1-10")
    pros: list[str] = Field(description="Positive aspects")
    cons: list[str] = Field(description="Negative aspects")

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("Review the movie Inception")
print(result.title)
print(result.rating)
```

---

### PydanticOutputParser

**Описание:** Парсер, который генерирует текстовые инструкции по формату (format instructions), вставляет их в промпт, а затем парсит текстовый ответ модели в Pydantic-объект. Работает с любой моделью, включая те, что не поддерживают tool calling.

```python
PydanticOutputParser(
    pydantic_object: type[BaseModel],     # целевой Pydantic класс для парсинга
)
```

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `get_format_instructions()` | — | `str` | Текстовое описание ожидаемого JSON-формата |
| `parse(text)` | `str` | `BaseModel` | Извлечение JSON из текста и валидация |
| `invoke(input)` | `str \| AIMessage` | `BaseModel` | LCEL-совместимый вызов (вызывает parse) |

**Пример использования:**

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    label: str = Field(description="positive, negative, or neutral")
    score: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")

parser = PydanticOutputParser(pydantic_object=Sentiment)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze sentiment.\n\n{format_instructions}"),
    ("human", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = await chain.ainvoke({"text": "This product is amazing!"})
print(result.label, result.score)
```

---

### JsonOutputParser

**Описание:** Парсер для получения `dict` из текстового ответа LLM. Не выполняет Pydantic-валидацию, но может генерировать format instructions на основе схемы. Подходит для прототипирования и работы с произвольным JSON.

```python
JsonOutputParser(
    pydantic_object: type[BaseModel] | None = None,  # опционально, для format instructions
)
```

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `get_format_instructions()` | — | `str` | Описание формата (если передана модель) |
| `parse(text)` | `str` | `dict` | Извлечение JSON из текста |
| `invoke(input)` | `str \| AIMessage` | `dict` | LCEL-совместимый вызов |

**Пример использования:**

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

chain = prompt | llm | parser
result = await chain.ainvoke({"query": "List 3 programming languages"})
print(type(result))
print(result)
```

---

### OutputFixingParser

**Описание:** Обёртка над другим парсером, которая при ошибке парсинга автоматически вызывает LLM для исправления. Реализует двухпроходную стратегию: parse → fail → fix → parse. Увеличивает надёжность за счёт дополнительной latency и стоимости.

```python
OutputFixingParser.from_llm(
    parser: BaseOutputParser,         # базовый парсер (PydanticOutputParser и т.д.)
    llm: BaseChatModel,               # LLM для исправления ошибок
    max_retries: int = 1,             # максимум попыток исправления
)
```

**Основные методы:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `parse(text)` | `str` | `BaseModel \| dict` | Парсинг с автоисправлением |
| `invoke(input)` | `str \| AIMessage` | `BaseModel \| dict` | LCEL-совместимый вызов |

**Пример использования:**

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

base_parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

malformed = '{"overall_score": 85, summary: "Good work"}'
result = fixing_parser.invoke(malformed)
print(type(result))
```

---

### Pydantic Field()

**Описание:** Функция для конфигурации полей Pydantic-модели. В контексте LLM — основной инструмент управления тем, что модель генерирует: `description` передаётся как часть JSON Schema, числовые ограничения ограничивают constrained decoding.

```python
Field(
    default: Any = PydanticUndefined,     # значение по умолчанию
    default_factory: Callable = None,      # фабрика для default (для mutable типов)
    alias: str | None = None,              # альтернативное имя поля при десериализации
    title: str | None = None,              # человекочитаемое имя (→ JSON Schema title)
    description: str | None = None,        # описание поля (→ JSON Schema description, читается LLM)
    examples: list[Any] | None = None,     # примеры значений (→ JSON Schema examples)
    gt: float | None = None,               # > (exclusive minimum)
    ge: float | None = None,               # >= (minimum)
    lt: float | None = None,               # < (exclusive maximum)
    le: float | None = None,               # <= (maximum)
    min_length: int | None = None,         # минимальная длина строки/списка
    max_length: int | None = None,         # максимальная длина строки/списка
    pattern: str | None = None,            # regex-паттерн для строки
    json_schema_extra: dict | None = None, # произвольные доп. поля в JSON Schema
)
```

**Пример использования:**

```python
from pydantic import BaseModel, Field

class DetailedScore(BaseModel):
    criterion: str = Field(
        description="Name of the evaluated criterion",
        min_length=1,
        max_length=100,
    )
    score: int = Field(
        ge=0,
        le=25,
        description="Score for this criterion, 0 to max_score",
    )
    feedback: str = Field(
        description="Specific feedback with references to the student's work",
        min_length=20,
        examples=["The thesis is clearly stated in paragraph 1, but lacks..."],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this score: 0.0=uncertain, 1.0=certain",
    )
```

---

### Конфигурация BaseModel

**Описание:** Pydantic V2 `ConfigDict` позволяет настроить поведение валидации, сериализации и десериализации. Для LLM-ответов ключевые параметры: `strict`, `extra`, `frozen`.

```python
from pydantic import ConfigDict

ConfigDict(
    strict: bool = False,                  # True: запрет implicit coercion
    extra: str = "ignore",                 # "allow" | "ignore" | "forbid"
    frozen: bool = False,                  # True: immutable instances
    populate_by_name: bool = False,        # True: принимать и alias, и имя поля
    str_strip_whitespace: bool = False,    # True: strip пробелов в строках
    str_min_length: int = 0,              # глобальный минимум длины строк
)
```

**Основные методы BaseModel:**

| Метод | Вход | Выход | Описание |
|---|---|---|---|
| `model_validate(data)` | `dict \| Any` | `BaseModel` | Создание и валидация из данных |
| `model_dump()` | — | `dict` | Сериализация в dict |
| `model_dump_json()` | — | `str` | Сериализация в JSON-строку |
| `model_json_schema()` | — | `dict` | JSON Schema для модели (class method) |

**Пример использования:**

```python
from pydantic import BaseModel, ConfigDict, Field

class LLMResponse(BaseModel):
    model_config = ConfigDict(strict=False, extra="ignore")

    answer: str = Field(description="The answer")
    score: int = Field(ge=0, le=100)

data = {"answer": "Good", "score": "85", "extra_field": True}
result = LLMResponse.model_validate(data)
print(result.score)
print(result.model_dump())
```

---

## Практика

### Пример 1: with_structured_output() — типизированный ответ

Демонстрирует основной подход: Pydantic-модель с `Field(description=..., ge=..., le=...)`, constrained decoding через tool calling API. Результат — типизированный объект.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    criterion_name: str = Field(description="Name of the evaluated criterion")
    score: int = Field(ge=0, le=25, description="Score for this criterion, 0 to 25")
    feedback: str = Field(description="Detailed feedback explaining the score")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this score: 0.0=very uncertain, 1.0=fully certain",
    )


class AssessmentResponse(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Total score across all criteria")
    criterion_scores: list[CriterionScore] = Field(description="Per-criterion breakdown")
    summary: str = Field(description="Brief overall assessment summary, 2-3 sentences")
    strengths: list[str] = Field(description="Key strengths, minimum 2 items")
    improvements: list[str] = Field(description="Improvement suggestions, minimum 2 items")


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(AssessmentResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an essay assessor. Evaluate student work against these criteria:\n"
     "- Thesis (max 25): clarity and strength of the main argument\n"
     "- Evidence (max 25): quality and relevance of supporting evidence\n"
     "- Structure (max 25): organization and logical flow\n"
     "- Language (max 25): grammar, vocabulary, academic tone"),
    ("human", "Assess this essay:\n\n{student_work}"),
])

chain = prompt | structured_llm

result = chain.invoke({
    "student_work": (
        "Climate change is a major threat. Rising temperatures cause ice to melt, "
        "leading to higher sea levels. Governments should implement carbon taxes "
        "and invest in renewable energy. Without action, future generations will suffer."
    ),
})

print(f"Overall score: {result.overall_score}")
print(f"Summary: {result.summary}")
for cs in result.criterion_scores:
    print(f"  {cs.criterion_name}: {cs.score}/25 (confidence: {cs.confidence:.2f})")
print(f"Strengths: {result.strengths}")
print(f"Improvements: {result.improvements}")
print(f"Return type: {type(result).__name__}")
```

`Field(description=...)` с деталями (`confidence`, `feedback`) направляет модель генерировать калиброванные оценки. Ограничения `ge=0, le=25` гарантируют корректный диапазон через constrained decoding.

`include_raw=True` позволяет инспектировать сырой ответ модели — полезно при отладке:

```python
structured_llm_raw = llm.with_structured_output(AssessmentResponse, include_raw=True)
chain_raw = prompt | structured_llm_raw

raw_result = chain_raw.invoke({
    "student_work": "Climate change is a major threat.",
})

print(f"Raw content: {raw_result['raw'].content!r}")
print(f"Tool calls: {[tc['name'] for tc in raw_result['raw'].tool_calls]}")
print(f"Parsed: {type(raw_result['parsed']).__name__}")
print(f"Parsing error: {raw_result['parsing_error']}")
```

При tool calling `raw.content` будет пустым, а `raw.tool_calls` содержит данные, из которых парсится `parsed`.

### Пример 2: PydanticOutputParser — промпт-подход

Альтернативный подход: format instructions вставляются в промпт как текст, модель генерирует JSON, парсер валидирует через Pydantic.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AssessmentResponse(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Total score across all criteria")
    summary: str = Field(description="Brief overall assessment summary")
    strengths: list[str] = Field(description="Key strengths of the work")
    improvements: list[str] = Field(description="Improvement suggestions")


parser = PydanticOutputParser(pydantic_object=AssessmentResponse)

print("=== Format Instructions (первые 200 символов) ===")
print(parser.get_format_instructions()[:200] + "...")

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an essay assessor.\n\n{format_instructions}"),
    ("human", "Assess this essay:\n\n{student_work}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

result = chain.invoke({
    "student_work": (
        "Climate change is a major threat. Rising temperatures cause ice to melt, "
        "leading to higher sea levels. Governments should implement carbon taxes "
        "and invest in renewable energy. Without action, future generations will suffer."
    ),
})

print(f"\nScore: {result.overall_score}")
print(f"Summary: {result.summary}")
print(f"Return type: {type(result).__name__}")
```

Ключевое отличие от `with_structured_output`: модель получает текстовую инструкцию по формату, а не constraint на уровне token generation. Надёжность ниже (~90% vs ~99.5%), но работает с любой моделью.

### Пример 3: Сравнение двух подходов

Запускаем оба метода на одном тексте и сравниваем результаты:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AssessmentResponse(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Total score across all criteria")
    summary: str = Field(description="Brief overall assessment summary")
    strengths: list[str] = Field(description="Key strengths of the work")
    improvements: list[str] = Field(description="Improvement suggestions")


STUDENT_WORK = (
    "Climate change is a major threat. Rising temperatures cause ice to melt, "
    "leading to higher sea levels. Governments should implement carbon taxes "
    "and invest in renewable energy. Without action, future generations will suffer."
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

prompt_a = ChatPromptTemplate.from_messages([
    ("system", "You are an essay assessor. Evaluate on 0-100 scale."),
    ("human", "Assess:\n\n{student_work}"),
])
chain_a = prompt_a | llm.with_structured_output(AssessmentResponse)

try:
    result_a = chain_a.invoke({"student_work": STUDENT_WORK})
    print(f"with_structured_output: score={result_a.overall_score}, success=True")
except Exception as e:
    result_a = None
    print(f"with_structured_output: error={e}")

parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
prompt_b = ChatPromptTemplate.from_messages([
    ("system", "You are an essay assessor.\n\n{format_instructions}"),
    ("human", "Assess:\n\n{student_work}"),
]).partial(format_instructions=parser.get_format_instructions())
chain_b = prompt_b | llm | parser

try:
    result_b = chain_b.invoke({"student_work": STUDENT_WORK})
    print(f"PydanticOutputParser:   score={result_b.overall_score}, success=True")
except Exception as e:
    result_b = None
    print(f"PydanticOutputParser:   error={e}")

if result_a and result_b:
    diff = abs(result_a.overall_score - result_b.overall_score)
    print(f"\nScore difference: {diff} points")
    print(f"Scores match (within 5): {diff <= 5}")
```

Оба метода возвращают Pydantic-объект одного типа. Ожидаемый результат: оценки отличаются на ≤5 баллов. Разница объясняется стохастичностью генерации и различием промптов (format instructions добавляют ~300 токенов контекста).

### Пример 4: Обработка ошибок и retry-логика

`PydanticOutputParser` может получить невалидный JSON (~5-10% случаев). Retry-цикл передаёт ошибку в контекст промпта, давая модели шанс исправиться:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AssessmentResponse(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Total score across all criteria")
    summary: str = Field(description="Brief overall assessment summary")
    strengths: list[str] = Field(description="Key strengths of the work")
    improvements: list[str] = Field(description="Improvement suggestions")


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=AssessmentResponse)

first_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an essay assessor.\n\n{format_instructions}"),
    ("human", "Assess:\n\n{student_work}"),
]).partial(format_instructions=parser.get_format_instructions())

retry_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an essay assessor.\n\n{format_instructions}\n\n"
     "Your previous response failed to parse. Error: {error_context}\n"
     "Return ONLY valid JSON matching the schema. No additional text."),
    ("human", "Assess:\n\n{student_work}"),
]).partial(format_instructions=parser.get_format_instructions())

student_work = (
    "Climate change is a major threat. Rising temperatures cause ice to melt, "
    "leading to higher sea levels. Governments should implement carbon taxes."
)
errors: list[str] = []
max_attempts = 3
result = None

for attempt in range(1, max_attempts + 1):
    try:
        if attempt == 1:
            chain = first_prompt | llm | parser
            result = chain.invoke({"student_work": student_work})
        else:
            chain = retry_prompt | llm | parser
            result = chain.invoke({
                "student_work": student_work,
                "error_context": errors[-1],
            })
        print(f"Success on attempt {attempt}")
        print(f"Score: {result.overall_score}")
        print(f"Summary: {result.summary}")
        break
    except Exception as e:
        errors.append(str(e))
        print(f"Attempt {attempt} failed: {e}")

if result is None:
    print(f"\nAll {max_attempts} attempts failed")
    for i, err in enumerate(errors, 1):
        print(f"  Attempt {i}: {err}")
```

Альтернатива ручному retry — `OutputFixingParser`, который автоматизирует этот процесс:

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

malformed = '{"overall_score": 85, summary: "Good work", "strengths": ["clear"], "improvements": ["add sources"]}'

try:
    result = fixing_parser.invoke(malformed)
    print(f"Fixed successfully: score={result.overall_score}")
    print(f"Type: {type(result).__name__}")
except Exception as e:
    print(f"Fixing failed: {e}")
```

`OutputFixingParser` вызывает LLM повторно с сообщением об ошибке — это удваивает latency и стоимость при ошибках, но повышает success rate с ~90% до ~97%.

**Связь примеров с теорией:**

| Пример | Концепция из теории | Что демонстрирует |
|---|---|---|
| Пример 1 | §2, §6 | `with_structured_output`, Field constraints, `include_raw` |
| Пример 2 | §3 | `PydanticOutputParser`, format instructions в промпте |
| Пример 3 | §2 vs §3 | Эмпирическое сравнение двух подходов |
| Пример 4 | §3, §5 | Retry-логика и `OutputFixingParser` |

---

## Чеклист самопроверки

- [ ] Объясни разницу между `with_structured_output` и `PydanticOutputParser`. Когда выбрать какой?
- [ ] Как constrained decoding ограничивает вывод модели на уровне токенов?
- [ ] Что происходит «под капотом», когда `with_structured_output` конвертирует Pydantic-схему в tool definition?
- [ ] Почему `Field(description=...)` влияет на качество ответа LLM?
- [ ] Что делает `include_raw=True` и зачем это нужно для отладки?
- [ ] Как работает `OutputFixingParser` и какова цена его использования?
- [ ] Почему нельзя стримить structured output через tool calling?
- [ ] В чём разница между `ConfigDict(strict=True)` и `strict=False` для LLM-ответов?
- [ ] Как работает retry-логика с передачей ошибки в контекст промпта?
- [ ] Какой подход выбрать для модели без поддержки tool calling?

---

## Частые ошибки

### 1. Pydantic-схема без описаний полей

```python
class Result(BaseModel):
    score: int
    text: str
```

```python
class Result(BaseModel):
    score: int = Field(description="Overall score 0-100, sum of all criteria")
    text: str = Field(description="Detailed feedback, 3-5 sentences per criterion")
```

Без description модель угадывает, что ожидается. С description — следует точным инструкциям.

### 2. Слишком глубокая вложенность

```python
class Deep(BaseModel):
    level1: list[dict[str, list[SubModel]]]
```

```python
class Flat(BaseModel):
    items: list[SimpleItem]
```

Сложные вложенные структуры увеличивают вероятность ошибок, особенно у менее мощных моделей. Плоские схемы надёжнее.

### 3. Отсутствие обработки ошибок парсинга

```python
result = await chain.ainvoke(data)
```

```python
try:
    result = await chain.ainvoke(data)
except (OutputParserException, ValidationError) as e:
    logger.error(f"Parse failed: {e}")
    result = await fallback_chain.ainvoke(data)
```

Даже `with_structured_output` может изредка упасть — всегда обрабатывайте исключения.

### 4. format_instructions не вставлены в промпт

```python
parser = PydanticOutputParser(pydantic_object=MyModel)
chain = prompt | llm | parser
```

```python
parser = PydanticOutputParser(pydantic_object=MyModel)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in JSON.\n\n{format_instructions}"),
    ("human", "{query}"),
]).partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
```

Без вставки `format_instructions` в промпт модель не знает ожидаемый формат.

### 5. Использование strict=True для LLM-ответов

```python
class Response(BaseModel):
    model_config = ConfigDict(strict=True)
    score: int
```

```python
class Response(BaseModel):
    score: int
```

`strict=True` запрещает coercion (`"85"` → `85`). LLM иногда возвращают числа как строки — lax mode (default) обрабатывает это автоматически.

---

## Что читать дальше

- [LangChain Structured Output](https://python.langchain.com/docs/concepts/structured_outputs/) — концепция
- [How to use with_structured_output](https://python.langchain.com/docs/how_to/structured_output/) — практика
- [Pydantic Fields](https://docs.pydantic.dev/latest/concepts/fields/) — все параметры Field()
- [Pydantic ConfigDict](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict) — конфигурация
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — как tool calling работает в Claude
- [JSON Schema specification](https://json-schema.org/understanding-json-schema/) — формат, используемый для описания схемы

**Следующая тема:** [Тема 4: Streaming](topic_04_streaming.md) — как отдавать ответ LLM в реальном времени.

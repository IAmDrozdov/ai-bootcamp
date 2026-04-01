# Тема 11: Tool Use / Function Calling — глубокое погружение

> **Пререквизиты:** [Тема 6: LangGraph + Agents](topic_06_langgraph_agents.md)
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `langgraph`, `pydantic`, `httpx`

---

## Теория

### 1. От @tool к BaseTool — спектр абстракций

LangChain предоставляет три уровня абстракции для создания инструментов, которые LLM может вызывать. Выбор уровня зависит от сложности задачи и степени контроля, которая нужна разработчику.

**Уровень 1: декоратор `@tool`**

Самый простой способ — обернуть обычную Python-функцию декоратором `@tool`. LangChain автоматически извлекает имя из имени функции, описание — из docstring, а схему аргументов — из type hints.

```python
from langchain_core.tools import tool

@tool
def count_words(text: str) -> dict:
    """Count words, sentences and paragraphs in the given text."""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
    }
```

Декоратор создаёт экземпляр `StructuredTool` под капотом. Для большинства инструментов в проекте этого достаточно.

**Уровень 2: `StructuredTool.from_function`**

Когда нужно указать имя, отличное от имени функции, задать кастомную `args_schema`, или настроить обработку ошибок — используется фабрика `StructuredTool.from_function`.

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class WordCountInput(BaseModel):
    text: str = Field(description="The text to analyze for word count statistics")

def _count_words(text: str) -> dict:
    words = text.split()
    return {"word_count": len(words)}

count_words_tool = StructuredTool.from_function(
    func=_count_words,
    name="word_counter",
    description="Count words in the provided text",
    args_schema=WordCountInput,
    return_direct=False,
    handle_tool_error=True,
)
```

Этот подход даёт контроль над каждым аспектом tool definition без наследования.

**Уровень 3: наследование от `BaseTool`**

Полный контроль: кастомная валидация, сложная логика в `_run`/`_arun`, состояние между вызовами, custom callbacks.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class EssayAnalysisInput(BaseModel):
    text: str = Field(description="Essay text to analyze")
    criteria: list[str] = Field(description="List of criteria to check")

class EssayAnalysisTool(BaseTool):
    name: str = "essay_analysis"
    description: str = "Deep analysis of essay against specified criteria"
    args_schema: type[BaseModel] = EssayAnalysisInput
    rubric_db: dict = {}

    def _run(self, text: str, criteria: list[str]) -> dict:
        results = {}
        for criterion in criteria:
            results[criterion] = self._evaluate_criterion(text, criterion)
        return results

    async def _arun(self, text: str, criteria: list[str]) -> dict:
        return self._run(text, criteria)

    def _evaluate_criterion(self, text: str, criterion: str) -> str:
        return f"Evaluated '{criterion}' for text of {len(text)} chars"
```

**Таблица выбора:**

| Критерий | `@tool` | `StructuredTool.from_function` | `BaseTool` |
|---|---|---|---|
| Сложность | Минимальная | Средняя | Максимальная |
| Кастомное имя | Нет (имя = имя функции) | Да | Да |
| args_schema | Автоматически из type hints | Явная Pydantic-модель | Явная Pydantic-модель |
| Внутреннее состояние | Нет | Нет | Да (поля класса) |
| Custom validation | Нет | Нет | Да (_run может валидировать) |
| async | Через `async def` | Через `coroutine` параметр | Через `_arun` |
| handle_tool_error | Через параметр декоратора | Через параметр | Через переопределение |
| Типичное применение | 80% случаев | 15% случаев | 5% случаев |

Правило: начинай с `@tool`. Переходи на `StructuredTool.from_function`, когда нужен контроль над именем или args_schema. Используй `BaseTool` только когда tool требует состояния или сложной логики инициализации.

### 2. args_schema — типизация аргументов через Pydantic

Когда LLM получает описание tool, она видит JSON Schema — формальное описание каждого аргумента с типами и описаниями. Качество этой схемы напрямую влияет на то, насколько корректно модель будет вызывать инструмент.

**Как работает маппинг:**

LangChain конвертирует Pydantic-модель в JSON Schema, которая отправляется провайдеру. Каждое поле Pydantic → свойство в JSON Schema. `Field(description=...)` → `"description"` в schema.

```python
from pydantic import BaseModel, Field
from enum import Enum

class CitationStyle(str, Enum):
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"

class CheckCitationsInput(BaseModel):
    text: str = Field(
        description="Full text of the student essay to check for citations"
    )
    style: CitationStyle = Field(
        description="Citation style to validate against: apa, mla, or chicago"
    )
    min_citations: int = Field(
        default=3,
        description="Minimum number of citations expected in the essay",
        ge=1,
        le=50,
    )
```

Эта модель генерирует следующий JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Full text of the student essay to check for citations"
    },
    "style": {
      "type": "string",
      "enum": ["apa", "mla", "chicago"],
      "description": "Citation style to validate against: apa, mla, or chicago"
    },
    "min_citations": {
      "type": "integer",
      "default": 3,
      "minimum": 1,
      "maximum": 50,
      "description": "Minimum number of citations expected in the essay"
    }
  },
  "required": ["text", "style"]
}
```

**Почему description критичен:**

LLM использует `description` каждого поля как инструкцию. Без description модель угадывает семантику по имени поля. Сравним:

```python
class Bad(BaseModel):
    text: str
    n: int

class Good(BaseModel):
    text: str = Field(description="Student essay text, including all paragraphs")
    max_results: int = Field(
        description="Maximum number of issues to return, 1-20",
        ge=1,
        le=20,
    )
```

С `Bad` модель может передать в `n` что угодно — количество слов, оценку, номер страницы. С `Good` — точно знает, что `max_results` это лимит на количество результатов в диапазоне 1-20.

**Вложенные модели:**

Pydantic поддерживает вложенные модели, и LangChain корректно конвертирует их в nested JSON Schema.

```python
class RubricCriterion(BaseModel):
    name: str = Field(description="Name of the assessment criterion")
    max_score: int = Field(description="Maximum points for this criterion")
    weight: float = Field(description="Weight of this criterion, 0.0-1.0")

class RubricInput(BaseModel):
    criteria: list[RubricCriterion] = Field(
        description="List of assessment criteria with scores and weights"
    )
    passing_threshold: float = Field(
        default=0.6,
        description="Minimum percentage to pass, 0.0-1.0",
    )
```

Однако глубоко вложенные структуры (>2 уровней) снижают точность tool calling. Модели лучше работают с плоскими или слегка вложенными схемами.

**Optional поля и default:**

```python
class AnalyzeInput(BaseModel):
    text: str = Field(description="Text to analyze")
    language: str = Field(
        default="en",
        description="Language code: en, ru, es, etc.",
    )
    include_stats: bool = Field(
        default=True,
        description="Whether to include statistical analysis",
    )
```

Optional поля с default — LLM может их пропустить, и tool получит значение по умолчанию. Это уменьшает количество аргументов, которые модели нужно генерировать.

### 3. bind_tools() — как LLM узнаёт об инструментах

Метод `bind_tools()` принимает список инструментов и возвращает новый экземпляр модели, который при каждом вызове отправляет описания tools провайдеру.

**Базовый flow:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def count_words(text: str) -> dict:
    """Count the number of words in the text."""
    return {"word_count": len(text.split())}

@tool
def check_structure(text: str) -> dict:
    """Check if essay has introduction, body, and conclusion."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {
        "has_intro": len(paragraphs) >= 1,
        "has_body": len(paragraphs) >= 2,
        "has_conclusion": len(paragraphs) >= 3,
        "paragraph_count": len(paragraphs),
    }

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools([count_words, check_structure])
```

Под капотом `bind_tools` конвертирует каждый tool в JSON Schema и добавляет их в параметры запроса к API. Для Anthropic это массив `tools` в теле запроса. Для OpenAI — массив `functions` (legacy) или `tools`.

**Что видит модель:**

При вызове `llm_with_tools.invoke(messages)` провайдеру отправляется:

```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [...],
  "tools": [
    {
      "name": "count_words",
      "description": "Count the number of words in the text.",
      "input_schema": {
        "type": "object",
        "properties": {
          "text": {"type": "string"}
        },
        "required": ["text"]
      }
    },
    {
      "name": "check_structure",
      "description": "Check if essay has introduction, body, and conclusion.",
      "input_schema": {
        "type": "object",
        "properties": {
          "text": {"type": "string"}
        },
        "required": ["text"]
      }
    }
  ]
}
```

**tool_choice — управление поведением:**

Параметр `tool_choice` определяет, как модель решает, вызывать ли tool:

| Значение | Поведение |
|---|---|
| `"auto"` (default) | Модель сама решает: вызвать tool, ответить текстом, или сделать и то и другое |
| `"any"` | Модель обязана вызвать хотя бы один tool (но выбирает какой) |
| `"none"` | Tools отправляются, но модель не может их вызвать (для контекста) |
| `{"type": "tool", "name": "count_words"}` | Модель обязана вызвать конкретный tool |

```python
llm_forced = llm.bind_tools(
    [count_words, check_structure],
    tool_choice={"type": "tool", "name": "count_words"},
)
```

С `tool_choice="any"` модель гарантированно вызовет tool, но может выбрать любой. Это полезно, когда текстовый ответ не нужен — только результат инструмента.

**Разница провайдеров:**

| Аспект | Anthropic (Claude) | OpenAI (GPT) |
|---|---|---|
| Формат запроса | `tools` + `tool_choice` | `tools` + `tool_choice` |
| Формат ответа tool_call | `content[].type = "tool_use"` | `tool_calls[]` в message |
| Параллельные вызовы | Да, несколько `tool_use` блоков | Да, несколько `tool_calls` |
| Forced choice | `tool_choice: {"type": "tool", "name": "..."}` | `tool_choice: {"type": "function", "function": {"name": "..."}}` |
| Описание в description | До 1024 символов рекомендуется | До 1024 символов рекомендуется |

LangChain абстрагирует эти различия: `bind_tools()` и `tool_calls` работают одинаково для обоих провайдеров.

### 4. Tool calling flow — полный цикл

Tool calling — это многошаговый процесс. Модель не выполняет инструмент сама. Она **запрашивает** вызов, клиент его выполняет, и результат возвращается модели для финального ответа.

**Шаг 1: Пользователь → LLM**

```python
from langchain_core.messages import HumanMessage

messages = [
    HumanMessage(content="Analyze this essay: 'AI is transforming education...'")
]
response = await llm_with_tools.ainvoke(messages)
```

**Шаг 2: LLM → AIMessage с tool_calls**

Модель возвращает `AIMessage`, у которого поле `content` может быть пустым (или содержать рассуждения), а поле `tool_calls` содержит список запросов на вызов инструментов:

```python
response.tool_calls
```

```python
[
    {
        "name": "count_words",
        "args": {"text": "AI is transforming education..."},
        "id": "toolu_01ABC123",
        "type": "tool_call",
    }
]
```

Каждый tool call содержит:
- `name` — имя инструмента
- `args` — аргументы в виде словаря (уже распарсенные из JSON)
- `id` — уникальный идентификатор вызова (нужен для ToolMessage)
- `type` — всегда `"tool_call"`

**Шаг 3: Выполнение tool + ToolMessage**

Клиент находит инструмент по имени и вызывает его:

```python
from langchain_core.messages import ToolMessage

tool_call = response.tool_calls[0]
tool_result = count_words.invoke(tool_call["args"])

tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call["id"],
)
```

`tool_call_id` — связывает результат с конкретным запросом. Без корректного ID модель не поймёт, на какой запрос пришёл ответ.

**Шаг 4: ToolMessage → LLM → финальный ответ**

```python
messages = [
    HumanMessage(content="Analyze this essay: 'AI is transforming education...'"),
    response,
    tool_message,
]
final_response = await llm_with_tools.ainvoke(messages)
print(final_response.content)
```

Модель видит результат инструмента и формирует финальный ответ на основе этих данных.

**Полный цикл в одной функции:**

```python
from langchain_core.messages import HumanMessage, ToolMessage

async def run_tool_loop(llm_with_tools, tools_map, user_message):
    messages = [HumanMessage(content=user_message)]

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_fn = tools_map[tool_call["name"]]
            result = await tool_fn.ainvoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )
```

Цикл `while True` нужен, потому что после получения результатов модель может решить вызвать ещё один tool (или тот же tool с другими аргументами). Цикл завершается, когда `tool_calls` пуст — модель готова дать финальный текстовый ответ.

**Параллельный tool calling:**

Модель может запросить несколько инструментов за один ход:

```python
response = await llm_with_tools.ainvoke([
    HumanMessage(content="Count words and check structure of this essay: '...'")
])
print(len(response.tool_calls))
```

Если модель вернула 2 tool calls, нужно вернуть 2 ToolMessage — по одному на каждый `tool_call_id`. Порядок ToolMessage не важен, но каждый должен ссылаться на правильный ID.

```python
import asyncio

async def execute_parallel(response, tools_map):
    tasks = []
    for tc in response.tool_calls:
        tool_fn = tools_map[tc["name"]]
        tasks.append(tool_fn.ainvoke(tc["args"]))
    results = await asyncio.gather(*tasks)

    tool_messages = []
    for tc, result in zip(response.tool_calls, results):
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return tool_messages
```

### 5. Error handling в tools

Инструменты работают с внешними системами — базами данных, API, файлами. Ошибки неизбежны. LangChain предоставляет несколько механизмов обработки ошибок в tools.

**ToolException — специальное исключение:**

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def get_rubric(rubric_name: str) -> dict:
    """Fetch assessment rubric by name."""
    rubrics = {"essay": {"criteria": ["thesis", "evidence"]}}
    if rubric_name not in rubrics:
        raise ToolException(
            f"Rubric '{rubric_name}' not found. "
            f"Available rubrics: {list(rubrics.keys())}"
        )
    return rubrics[rubric_name]
```

Когда `handle_tool_error=True` и функция бросает `ToolException`, LangChain перехватывает её и возвращает сообщение об ошибке как ToolMessage. Модель видит ошибку и может попробовать другой подход — например, запросить другое имя рубрики.

Обычные исключения (`ValueError`, `KeyError`) при `handle_tool_error=True` тоже перехватываются. Разница в том, что `ToolException` позволяет контролировать текст ошибки, который увидит модель.

**Custom error handler:**

```python
def rubric_error_handler(error: ToolException) -> str:
    return (
        f"Tool execution failed: {error}. "
        f"Please try a different rubric name or ask the user to specify one."
    )

@tool(handle_tool_error=rubric_error_handler)
def get_rubric_v2(rubric_name: str) -> dict:
    """Fetch assessment rubric by name."""
    raise ToolException(f"Rubric '{rubric_name}' not found")
```

Кастомный handler позволяет добавить инструкции для модели — как исправить ситуацию, что попробовать, какие альтернативы доступны.

**Retry внутри tool:**

Для нестабильных внешних API полезен retry:

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import tool

@tool
async def check_plagiarism(text: str) -> dict:
    """Check text for potential plagiarism using external API."""
    return await _call_plagiarism_api(text)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def _call_plagiarism_api(text: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.example.com/plagiarism",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()
```

Retry вынесен в отдельную функцию, чтобы не загромождать tool. Стратегия `wait_exponential` — 1с, 2с, 4с — снижает нагрузку на API при повторных вызовах.

**Fallback tools:**

Если основной tool недоступен, можно переключиться на fallback:

```python
from langchain_core.tools import tool, ToolException

@tool
def check_plagiarism_primary(text: str) -> dict:
    """Check text for plagiarism using primary service."""
    raise ToolException("Primary service unavailable")

@tool
def check_plagiarism_fallback(text: str) -> dict:
    """Check text for plagiarism using simple heuristic analysis."""
    sentences = text.split('.')
    unique_starts = len(set(s.strip().split()[0] for s in sentences if s.strip()))
    diversity = unique_starts / max(len(sentences), 1)
    return {
        "method": "heuristic",
        "diversity_score": round(diversity, 2),
        "warning": "Heuristic analysis only, not a full plagiarism check",
    }
```

Fallback можно реализовать как отдельный tool (модель выберет его после ошибки primary) или как логику внутри одного tool с try/except.

### 6. Async tools

В асинхронном приложении все I/O-операции должны быть асинхронными. Синхронный HTTP-вызов внутри tool блокирует event loop, что снижает пропускную способность.

**Создание async tool с `@tool`:**

```python
import httpx
from langchain_core.tools import tool

@tool
async def check_plagiarism(text: str) -> dict:
    """Check text for potential plagiarism using external API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.plagiarism-checker.example.com/check",
            json={"text": text[:5000]},
        )
        data = response.json()
        return {
            "originality_score": data.get("score", 0),
            "matched_sources": data.get("sources", []),
            "is_original": data.get("score", 0) > 0.85,
        }
```

При использовании `@tool` с `async def` LangChain автоматически создаёт tool с `coroutine`. При вызове через `ainvoke` используется async-версия, через `invoke` — LangChain оборачивает в `asyncio.run`.

**Async tool через BaseTool:**

```python
import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class PlagiarismInput(BaseModel):
    text: str = Field(description="Text to check for plagiarism")

class PlagiarismTool(BaseTool):
    name: str = "check_plagiarism"
    description: str = "Check text for potential plagiarism via external API"
    args_schema: type[BaseModel] = PlagiarismInput
    api_url: str = "https://api.example.com/check"

    def _run(self, text: str) -> dict:
        raise NotImplementedError("Use async version")

    async def _arun(self, text: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.api_url,
                json={"text": text[:5000]},
            )
            return response.json()
```

В `BaseTool` можно реализовать только `_arun` и бросить `NotImplementedError` в `_run`, если sync-вызов не поддерживается.

**Смешивание sync и async tools:**

LangChain позволяет комбинировать sync и async tools в одном наборе:

```python
@tool
def count_words(text: str) -> dict:
    """Count words in text."""
    return {"word_count": len(text.split())}

@tool
async def check_external_api(text: str) -> dict:
    """Check text against external API."""
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://api.example.com/check", json={"text": text})
        return resp.json()

tools = [count_words, check_external_api]
llm_with_tools = llm.bind_tools(tools)
```

При вызове через `ainvoke` sync-tool оборачивается в `run_in_executor`, а async-tool вызывается напрямую. Это безопасно, но sync-tool всё равно блокирует поток из thread pool.

Рекомендация: делай все tools, которые выполняют I/O (HTTP, DB, файлы), асинхронными. Sync оставляй только для чистых вычислений (подсчёт слов, regex, математика).

### 7. Dynamic tool selection

Не все tools нужны для каждой задачи. Модель, загруженная 30 инструментами, работает хуже, чем модель с 5 релевантными. Dynamic tool selection — подход, при котором набор tools адаптируется к контексту запроса.

**Наборы tools для разных типов работ:**

```python
from langchain_core.tools import tool

@tool
def check_essay_structure(text: str) -> dict:
    """Check if essay has introduction, body paragraphs, and conclusion."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {"paragraph_count": len(paragraphs), "has_structure": len(paragraphs) >= 3}

@tool
def check_citations(text: str, style: str) -> dict:
    """Check citations in the text for a given citation style (apa/mla)."""
    import re
    apa_pattern = r'\([A-Z][a-z]+,\s*\d{4}\)'
    mla_pattern = r'\([A-Z][a-z]+\s+\d+\)'
    pattern = apa_pattern if style == "apa" else mla_pattern
    citations = re.findall(pattern, text)
    return {"citation_count": len(citations), "citations": citations}

@tool
def run_code_tests(code: str) -> dict:
    """Run basic syntax and style checks on Python code."""
    issues = []
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            issues.append(f"Line {i}: exceeds 120 characters")
    return {"issues": issues, "passed": len(issues) == 0}

@tool
def check_math_solution(expression: str) -> dict:
    """Verify a mathematical expression or equation."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": result, "valid": True}
    except Exception:
        return {"expression": expression, "valid": False}

ESSAY_TOOLS = [check_essay_structure, check_citations]
CODE_TOOLS = [run_code_tests]
MATH_TOOLS = [check_math_solution]

TOOL_SETS = {
    "essay": ESSAY_TOOLS,
    "code": CODE_TOOLS,
    "math": MATH_TOOLS,
}
```

**Выбор набора по типу задачи:**

```python
def get_tools_for_task(task_type: str) -> list:
    base_tools = [count_words]
    specific_tools = TOOL_SETS.get(task_type, [])
    return base_tools + specific_tools
```

При запросе на оценку эссе модель получает `count_words` + `check_essay_structure` + `check_citations`. При оценке кода — `count_words` + `run_code_tests`. Модель видит только релевантные инструменты и не тратит токены на парсинг описаний ненужных.

**Конфигурация tools через LangGraph config:**

В LangGraph можно передавать набор tools через `configurable`:

```python
from langgraph.prebuilt import create_react_agent

def create_agent_with_tools(llm, task_type: str):
    tools = get_tools_for_task(task_type)
    return create_react_agent(llm, tools)
```

**Рекомендации по количеству tools:**

| Количество tools | Поведение модели |
|---|---|
| 1-5 | Отлично — модель точно выбирает нужный |
| 6-10 | Хорошо — иногда путает похожие tools |
| 11-15 | Приемлемо — нужны чёткие descriptions |
| 16-20 | Рискованно — модель может выбрать неоптимальный tool |
| >20 | Не рекомендуется — разбей на подмножества |

### 8. InjectedToolArg — скрытые аргументы

Часто tool нуждается в данных, которые не должна выбирать LLM: сессия базы данных, ID пользователя, конфигурация, API-ключи. `InjectedToolArg` позволяет скрыть аргумент из JSON Schema, которую видит модель, но передать его при вызове.

**Проблема:**

```python
@tool
def get_rubric(rubric_name: str, db_session: Session) -> dict:
    """Fetch rubric from database."""
    ...
```

Здесь `db_session` попадёт в JSON Schema — LLM попытается сгенерировать для него значение, что бессмысленно и вызовет ошибку.

**Решение с InjectedToolArg:**

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def get_rubric(
    rubric_name: str,
    db_session: Annotated[dict, InjectedToolArg],
) -> dict:
    """Fetch assessment rubric by name from the database."""
    rubrics = db_session.get("rubrics", {})
    if rubric_name not in rubrics:
        return {"error": f"Rubric '{rubric_name}' not found"}
    return rubrics[rubric_name]
```

`Annotated[dict, InjectedToolArg]` говорит LangChain: «Это поле не включать в JSON Schema для LLM. Оно будет передано при вызове tool извне».

Модель видит только:

```json
{
  "name": "get_rubric",
  "input_schema": {
    "properties": {
      "rubric_name": {"type": "string"}
    },
    "required": ["rubric_name"]
  }
}
```

Аргумент `db_session` не виден. При вызове tool инжектируем его вручную:

```python
tool_call = response.tool_calls[0]
result = get_rubric.invoke({
    **tool_call["args"],
    "db_session": {"rubrics": {"essay": {"criteria": ["thesis"]}}},
})
```

**Несколько скрытых аргументов:**

```python
@tool
def save_assessment(
    student_id: str,
    score: int,
    feedback: str,
    db_session: Annotated[dict, InjectedToolArg],
    current_user_id: Annotated[str, InjectedToolArg],
    config: Annotated[dict, InjectedToolArg],
) -> dict:
    """Save assessment result for a student."""
    return {
        "saved": True,
        "student_id": student_id,
        "assessor": current_user_id,
    }
```

LLM видит только `student_id`, `score`, `feedback`. Остальное инжектится.

**InjectedToolArg в ToolNode (LangGraph):**

При использовании `ToolNode` из LangGraph скрытые аргументы можно передать через `state`:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_rubric])
```

В этом случае при выполнении через граф аргументы с `InjectedToolArg` передаются из state или config графа, а не от LLM.

---

## Справочник API

### `@tool`

Декоратор для превращения функции в LangChain tool.

**Импорт:**

```python
from langchain_core.tools import tool
```

**Параметры декоратора:**

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `name_or_callable` | `str \| Callable` | имя функции | Имя tool (если передана строка) или сама функция |
| `return_direct` | `bool` | `False` | Если `True`, результат tool возвращается пользователю напрямую, без передачи обратно в LLM |
| `args_schema` | `type[BaseModel]` | автоматически | Pydantic-модель для валидации аргументов. Если не указана, генерируется из type hints |
| `infer_schema` | `bool` | `True` | Автоматически создавать args_schema из type hints |
| `response_format` | `str` | `"content"` | `"content"` — строка, `"content_and_artifact"` — tuple (content, artifact) |
| `parse_docstring` | `bool` | `False` | Парсить docstring для извлечения описаний аргументов |
| `error_on_invalid_docstring` | `bool` | `True` | Бросать ошибку при невалидном docstring (если `parse_docstring=True`) |
| `handle_tool_error` | `bool \| str \| Callable` | `False` | Обработка ошибок: `True` — автоматическая, `str` — фиксированное сообщение, `Callable` — custom handler |

**Пример с основными параметрами:**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WordCountInput(BaseModel):
    text: str = Field(description="Text to count words in")

@tool("word_counter", args_schema=WordCountInput, return_direct=False)
def count_words(text: str) -> dict:
    """Count the number of words in the provided text."""
    return {"word_count": len(text.split())}
```

**Пример с `parse_docstring`:**

```python
@tool(parse_docstring=True)
def analyze_text(text: str, max_length: int) -> dict:
    """Analyze the given text.

    Args:
        text: The text content to analyze.
        max_length: Maximum length of text to process.
    """
    return {"length": min(len(text), max_length)}
```

### `StructuredTool`

Класс для создания tools с полным контролем над конфигурацией.

**Импорт:**

```python
from langchain_core.tools import StructuredTool
```

**`StructuredTool.from_function()` параметры:**

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `func` | `Callable` | — | Синхронная функция для выполнения |
| `coroutine` | `Callable` | `None` | Асинхронная функция для выполнения |
| `name` | `str` | имя функции | Имя tool |
| `description` | `str` | docstring | Описание для LLM |
| `args_schema` | `type[BaseModel]` | автоматически | Pydantic-модель аргументов |
| `return_direct` | `bool` | `False` | Возвращать результат напрямую |
| `handle_tool_error` | `bool \| str \| Callable` | `False` | Обработка ошибок |
| `metadata` | `dict` | `None` | Дополнительные метаданные |
| `tags` | `list[str]` | `None` | Теги для фильтрации |

**Пример с sync и async:**

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str = Field(description="Text content to process")

def _analyze_sync(text: str) -> dict:
    return {"length": len(text), "words": len(text.split())}

async def _analyze_async(text: str) -> dict:
    return {"length": len(text), "words": len(text.split())}

analyze_tool = StructuredTool.from_function(
    func=_analyze_sync,
    coroutine=_analyze_async,
    name="text_analyzer",
    description="Analyze text and return statistics",
    args_schema=TextInput,
    handle_tool_error=True,
)
```

### `BaseTool`

Абстрактный базовый класс для tools с полным контролем.

**Импорт:**

```python
from langchain_core.tools import BaseTool
```

**Атрибуты для переопределения:**

| Атрибут / Метод | Тип | Описание |
|---|---|---|
| `name` | `str` | Имя tool (обязательно) |
| `description` | `str` | Описание для LLM (обязательно) |
| `args_schema` | `type[BaseModel]` | Pydantic-модель аргументов |
| `return_direct` | `bool` | Возвращать результат напрямую |
| `handle_tool_error` | `bool \| str \| Callable` | Обработка ошибок |
| `_run(**kwargs)` | Метод | Синхронная логика (обязательно) |
| `_arun(**kwargs)` | Метод | Асинхронная логика (опционально) |

**Пример:**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class AssessInput(BaseModel):
    text: str = Field(description="Student work to assess")
    rubric_id: str = Field(description="ID of the rubric to use")

class AssessmentTool(BaseTool):
    name: str = "assess_work"
    description: str = "Assess student work against a rubric"
    args_schema: type[BaseModel] = AssessInput
    rubric_store: dict = {}

    def _run(self, text: str, rubric_id: str) -> dict:
        rubric = self.rubric_store.get(rubric_id)
        if not rubric:
            return {"error": f"Rubric {rubric_id} not found"}
        return {"score": 85, "rubric": rubric_id}

    async def _arun(self, text: str, rubric_id: str) -> dict:
        return self._run(text, rubric_id)
```

### `bind_tools()`

Метод ChatModel, привязывающий tools к модели.

**Импорт:**

```python
from langchain_anthropic import ChatAnthropic
```

**Параметры:**

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `tools` | `list` | — | Список tools (объекты tool, Pydantic-классы, или dict) |
| `tool_choice` | `str \| dict` | `"auto"` | Стратегия выбора: `"auto"`, `"any"`, `"none"`, или dict с конкретным tool |

**Варианты `tool_choice`:**

```python
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

llm_auto = llm.bind_tools(tools, tool_choice="auto")

llm_any = llm.bind_tools(tools, tool_choice="any")

llm_none = llm.bind_tools(tools, tool_choice="none")

llm_forced = llm.bind_tools(
    tools,
    tool_choice={"type": "tool", "name": "count_words"},
)
```

**Возвращаемое значение:**

Новый экземпляр ChatModel с привязанными tools. Оригинальный `llm` не изменяется.

### `ToolMessage`

Сообщение с результатом выполнения tool, возвращаемое модели.

**Импорт:**

```python
from langchain_core.messages import ToolMessage
```

**Параметры:**

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `content` | `str \| list` | — | Результат выполнения tool (строка или structured content) |
| `tool_call_id` | `str` | — | ID tool call из `AIMessage.tool_calls[].id` |
| `name` | `str` | `None` | Имя tool (опционально, для отладки) |
| `artifact` | `Any` | `None` | Артефакт — данные, которые не отправляются модели |
| `status` | `str` | `"success"` | Статус: `"success"` или `"error"` |

**Пример:**

```python
from langchain_core.messages import ToolMessage

tool_msg = ToolMessage(
    content='{"word_count": 150, "sentence_count": 12}',
    tool_call_id="toolu_01ABC123",
    name="count_words",
)
```

**Важно:** `content` должен быть строкой. Если tool возвращает dict, его нужно сериализовать: `content=json.dumps(result)` или `content=str(result)`.

### `ToolNode`

Компонент LangGraph для автоматического выполнения tool calls.

**Импорт:**

```python
from langgraph.prebuilt import ToolNode
```

**Параметры:**

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `tools` | `list` | — | Список tools для выполнения |
| `name` | `str` | `"tools"` | Имя ноды в графе |
| `handle_tool_errors` | `bool \| str \| Callable` | `True` | Обработка ошибок в tools |

**Пример в графе:**

```python
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
tools = [count_words, check_structure]

tool_node = ToolNode(tools)

agent = create_react_agent(llm, tools)
result = await agent.ainvoke({"messages": [("human", "Count words in: hello world")]})
```

`ToolNode` автоматически:
1. Извлекает `tool_calls` из последнего AIMessage
2. Находит соответствующий tool по имени
3. Вызывает tool с аргументами
4. Создаёт ToolMessage с результатом и правильным `tool_call_id`
5. Добавляет ToolMessage в state

### `InjectedToolArg`

Маркер для скрытия аргумента tool от LLM schema.

**Импорт:**

```python
from langchain_core.tools import InjectedToolArg
```

**Использование:**

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def search_db(
    query: str,
    db_connection: Annotated[dict, InjectedToolArg],
) -> dict:
    """Search database for matching records."""
    return {"results": []}
```

`db_connection` не появляется в JSON Schema для LLM. При вызове передаётся вручную:

```python
result = search_db.invoke({"query": "test", "db_connection": db})
```

### `ToolException`

Исключение для корректной обработки ошибок в tools.

**Импорт:**

```python
from langchain_core.tools import ToolException
```

**Использование:**

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def divide(a: float, b: float) -> dict:
    """Divide a by b."""
    if b == 0:
        raise ToolException("Cannot divide by zero. Please provide a non-zero divisor.")
    return {"result": a / b}
```

При `handle_tool_error=True` ToolException перехватывается, и её сообщение возвращается как ToolMessage. Модель видит ошибку и может скорректировать аргументы.

| `handle_tool_error` | Поведение при ToolException |
|---|---|
| `False` | Исключение пробрасывается вверх — необработанная ошибка |
| `True` | Сообщение ToolException → ToolMessage |
| `str` | Фиксированная строка → ToolMessage |
| `Callable` | `handler(exception)` → ToolMessage |

---

## Практика

### Пример 1. @tool — базовый декоратор

```python
from langchain_core.tools import tool


@tool
def count_words(text: str) -> dict:
    """Count words, sentences, and paragraphs in the given text."""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": max(len(paragraphs), 1),
    }


@tool
def check_essay_structure(text: str) -> dict:
    """Check if essay has introduction, body paragraphs, and conclusion."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    intro_signals = ["this essay", "in this paper", "the purpose", "introduction"]
    conclusion_signals = ["in conclusion", "to summarize", "in summary", "therefore"]

    first_para = paragraphs[0].lower() if paragraphs else ""
    last_para = paragraphs[-1].lower() if paragraphs else ""

    has_intro = any(s in first_para for s in intro_signals)
    has_conclusion = any(s in last_para for s in conclusion_signals)
    has_body = len(paragraphs) >= 3

    return {
        "has_introduction": has_intro,
        "has_body_paragraphs": has_body,
        "has_conclusion": has_conclusion,
        "structure_score": sum([has_intro, has_body, has_conclusion]),
    }


result = count_words.invoke({"text": "AI is transforming education. Students learn faster."})
print("count_words:", result)

result = check_essay_structure.invoke({
    "text": (
        "This essay examines AI in education.\n\n"
        "AI tools help personalize learning for each student.\n\n"
        "In conclusion, AI will transform how we teach."
    )
})
print("check_essay_structure:", result)

print()
print("Tool name:", count_words.name)
print("Tool description:", count_words.description)
print("Tool schema:", count_words.args_schema.model_json_schema())
```

### Пример 2. StructuredTool и BaseTool с Pydantic-схемой

```python
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field


class CheckCitationsInput(BaseModel):
    text: str = Field(description="Essay text to check for citations")
    style: str = Field(description="Citation style: 'apa' or 'mla'")


def _check_citations(text: str, style: str) -> dict:
    import re
    patterns = {
        "apa": r'\([A-Z][a-z]+,\s*\d{4}\)',
        "mla": r'\([A-Z][a-z]+\s+\d+\)',
    }
    pattern = patterns.get(style.lower(), patterns["apa"])
    found = re.findall(pattern, text)
    return {"style": style, "citation_count": len(found), "citations": found}


check_citations = StructuredTool.from_function(
    func=_check_citations,
    name="check_citations",
    description="Find and validate citations in essay text",
    args_schema=CheckCitationsInput,
    handle_tool_error=True,
)

text_with_citations = "According to research (Smith, 2023), AI improves outcomes (Johnson, 2022)."
print("StructuredTool:", check_citations.invoke({"text": text_with_citations, "style": "apa"}))


class VocabularyInput(BaseModel):
    text: str = Field(description="Text to analyze for vocabulary richness")


class VocabularyTool(BaseTool):
    name: str = "analyze_vocabulary"
    description: str = "Analyze vocabulary richness and complexity of the text"
    args_schema: type[BaseModel] = VocabularyInput
    academic_markers: list[str] = [
        "however", "therefore", "furthermore", "moreover",
        "consequently", "demonstrates", "indicates", "analysis",
    ]

    def _run(self, text: str) -> dict:
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        unique = set(words)
        academic_found = [w for w in self.academic_markers if w in unique]
        return {
            "total_words": len(words),
            "unique_words": len(unique),
            "uniqueness_ratio": round(len(unique) / max(len(words), 1), 3),
            "academic_words": academic_found,
            "level": "advanced" if len(academic_found) >= 3 else "basic",
        }


vocab_tool = VocabularyTool()
print("BaseTool:", vocab_tool.invoke({
    "text": "Furthermore, this analysis demonstrates significant results. "
            "However, the methodology indicates limitations."
}))
```

### Пример 3. bind_tools() и цикл tool calling

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def count_words(text: str) -> dict:
    """Count words in the given text. Use for basic text statistics."""
    words = text.split()
    return {"word_count": len(words), "char_count": len(text)}


@tool
def check_structure(text: str) -> dict:
    """Check if essay has proper structure with intro, body, conclusion."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {"paragraph_count": len(paragraphs), "has_structure": len(paragraphs) >= 3}


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
tools = [count_words, check_structure]
llm_with_tools = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}


async def run_tool_loop(user_text: str) -> str:
    messages = [HumanMessage(content=user_text)]

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            print(f"  Tool call: {tc['name']}({tc['args']})")
            tool_fn = tools_map[tc["name"]]
            result = await tool_fn.ainvoke(tc["args"])
            print(f"  Result: {result}")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )


essay = (
    "This essay examines AI.\n\n"
    "AI helps students learn better.\n\n"
    "In conclusion, AI is useful."
)

print("=== Tool calling loop ===")
answer = asyncio.run(run_tool_loop(f"Analyze this essay:\n\n{essay}"))
print(f"\nFinal answer:\n{answer}")
```

### Пример 4. Параллельное выполнение tools

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def count_words(text: str) -> dict:
    """Count words in text."""
    return {"word_count": len(text.split())}


@tool
def check_structure(text: str) -> dict:
    """Check essay structure."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {"paragraphs": len(paragraphs), "has_structure": len(paragraphs) >= 3}


@tool
def analyze_vocabulary(text: str) -> dict:
    """Analyze vocabulary richness."""
    import re
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    unique = set(words)
    return {"unique_ratio": round(len(unique) / max(len(words), 1), 3)}


@tool
def check_citations(text: str, style: str) -> dict:
    """Find citations in APA or MLA style."""
    import re
    pattern = r'\([A-Z][a-z]+,\s*\d{4}\)' if style == "apa" else r'\([A-Z][a-z]+\s+\d+\)'
    return {"citations": re.findall(pattern, text)}


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
tools = [count_words, check_structure, analyze_vocabulary, check_citations]
llm_with_tools = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}


async def execute_parallel(response, tools_map):
    tasks = []
    for tc in response.tool_calls:
        tool_fn = tools_map[tc["name"]]
        tasks.append(tool_fn.ainvoke(tc["args"]))
    results = await asyncio.gather(*tasks)

    tool_messages = []
    for tc, result in zip(response.tool_calls, results):
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return tool_messages


async def main():
    essay = (
        "This essay examines AI in education (Smith, 2023). "
        "Furthermore, recent studies demonstrate significant improvements (Lee, 2022).\n\n"
        "The methodology indicates promising results. However, challenges remain.\n\n"
        "In conclusion, AI transforms education."
    )

    prompt = (
        "Analyze this essay comprehensively — check word count, structure, "
        "vocabulary, and citations (apa style). Use ALL available tools.\n\n"
        + essay
    )

    messages = [HumanMessage(content=prompt)]
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    print(f"Model requested {len(response.tool_calls)} tool calls:")
    for tc in response.tool_calls:
        print(f"  - {tc['name']}")

    tool_messages = await execute_parallel(response, tools_map)
    messages.extend(tool_messages)

    for msg in tool_messages:
        print(f"  Result: {msg.content[:100]}...")

    final = await llm_with_tools.ainvoke(messages)
    print(f"\nFinal answer:\n{final.content[:500]}...")


asyncio.run(main())
```

### Пример 5. Обработка ошибок и InjectedToolArg

```python
from typing import Annotated
from langchain_core.tools import tool, ToolException, InjectedToolArg


RUBRICS_DB = {
    "essay_basic": {"criteria": ["thesis", "evidence", "structure"], "max_score": 100},
}


@tool(handle_tool_error=True)
def get_rubric(rubric_name: str) -> dict:
    """Fetch assessment rubric by name. Available rubrics: essay_basic."""
    if rubric_name not in RUBRICS_DB:
        raise ToolException(
            f"Rubric '{rubric_name}' not found. "
            f"Available: {list(RUBRICS_DB.keys())}"
        )
    return RUBRICS_DB[rubric_name]


print("=== Successful call ===")
print(get_rubric.invoke({"rubric_name": "essay_basic"}))

print("\n=== Error handled ===")
print(get_rubric.invoke({"rubric_name": "nonexistent"}))


def custom_error_handler(error: ToolException) -> str:
    return f"Tool failed: {error}. Try 'essay_basic' instead."


@tool(handle_tool_error=custom_error_handler)
def get_rubric_v2(rubric_name: str) -> dict:
    """Fetch rubric with custom error handling."""
    if rubric_name not in RUBRICS_DB:
        raise ToolException(f"'{rubric_name}' not found")
    return RUBRICS_DB[rubric_name]


print("\n=== Custom error handler ===")
print(get_rubric_v2.invoke({"rubric_name": "wrong_name"}))


@tool
def save_assessment(
    student_id: str,
    score: int,
    db_session: Annotated[dict, InjectedToolArg],
) -> dict:
    """Save assessment result for a student."""
    db_session.setdefault("assessments", []).append(
        {"student_id": student_id, "score": score}
    )
    return {"saved": True, "student_id": student_id}


print("\n=== InjectedToolArg ===")
print("Schema (db_session hidden):", save_assessment.args_schema.model_json_schema())

fake_db = {"assessments": []}
result = save_assessment.invoke({
    "student_id": "s-001",
    "score": 85,
    "db_session": fake_db,
})
print("Result:", result)
print("DB state:", fake_db)
```

### Пример 6. Динамический выбор tools и tool_choice

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def count_words(text: str) -> dict:
    """Count words in text."""
    return {"word_count": len(text.split())}


@tool
def check_essay_structure(text: str) -> dict:
    """Check essay structure (intro, body, conclusion)."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {"paragraphs": len(paragraphs)}


@tool
def run_code_tests(code: str) -> dict:
    """Run basic syntax checks on Python code."""
    issues = []
    for i, line in enumerate(code.split('\n'), 1):
        if len(line) > 120:
            issues.append(f"Line {i}: too long")
    return {"issues": issues, "passed": len(issues) == 0}


@tool
def check_math_solution(expression: str) -> dict:
    """Verify a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result, "valid": True}
    except Exception:
        return {"valid": False}


TOOL_SETS = {
    "essay": [count_words, check_essay_structure],
    "code": [count_words, run_code_tests],
    "math": [count_words, check_math_solution],
}


def get_tools_for_task(task_type: str) -> list:
    return TOOL_SETS.get(task_type, [count_words])


llm = ChatAnthropic(model="claude-sonnet-4-20250514")


async def main():
    for task_type in ["essay", "code", "math"]:
        tools = get_tools_for_task(task_type)
        print(f"Task type '{task_type}': tools = {[t.name for t in tools]}")

    tools = get_tools_for_task("essay")

    llm_forced = llm.bind_tools(
        tools,
        tool_choice={"type": "tool", "name": "count_words"},
    )
    response = await llm_forced.ainvoke([
        HumanMessage(content="Tell me about AI in education.")
    ])
    print(f"\nForced tool_choice: {[tc['name'] for tc in response.tool_calls]}")

    llm_any = llm.bind_tools(tools, tool_choice="any")
    response = await llm_any.ainvoke([
        HumanMessage(content="Analyze this short essay about AI.")
    ])
    print(f"tool_choice='any': {[tc['name'] for tc in response.tool_calls]}")


asyncio.run(main())
```

### Связь с теорией

| Пример | Концепция из теории | Что демонстрирует |
|---|---|---|
| Пример 1 | §1 `@tool` | Создание tools через декоратор, автоматическая генерация schema из type hints |
| Пример 2 | §1 `StructuredTool` / `BaseTool`, §2 args_schema | Кастомная Pydantic-схема, внутреннее состояние в BaseTool |
| Пример 3 | §3 `bind_tools`, §4 Tool calling flow | Полный цикл: LLM запрашивает tool → клиент выполняет → результат обратно в LLM |
| Пример 4 | §4 Параллельный tool calling | asyncio.gather для одновременного выполнения нескольких tool calls |
| Пример 5 | §5 ToolException, handle_tool_error, §8 InjectedToolArg | Graceful error handling + скрытые аргументы через InjectedToolArg |
| Пример 6 | §7 Dynamic tool selection, §3 tool_choice | Выбор набора tools по контексту + принудительный вызов конкретного tool |

---

## Чеклист самопроверки

- [ ] Объясни разницу между `@tool`, `StructuredTool.from_function` и `BaseTool`. Когда использовать каждый вариант?
- [ ] Почему `Field(description=...)` в `args_schema` критичен для качества tool calling? Что видит модель без description?
- [ ] Как работает `tool_choice`? В чём разница между `"auto"`, `"any"` и конкретным tool?
- [ ] Нарисуй полный цикл tool calling: какие сообщения и в каком порядке проходят между клиентом и LLM?
- [ ] Что такое `tool_call_id` в `ToolMessage` и почему он обязателен?
- [ ] Как `ToolException` и `handle_tool_error` работают вместе? Что видит модель при ошибке?
- [ ] Почему важно делать I/O-bound tools асинхронными? Что произойдёт с sync HTTP-вызовом в async-приложении?
- [ ] Как `InjectedToolArg` скрывает аргумент от LLM? Где этот аргумент появляется при вызове tool?
- [ ] Почему больше 15 tools одновременно — плохая идея? Какие стратегии помогают?
- [ ] Как `ToolNode` из LangGraph автоматизирует цикл tool calling?

---

## Частые ошибки

### 1. Tool без docstring — LLM не знает, когда вызывать

```python
@tool
def analyze(text: str) -> dict:
    words = text.split()
    return {"count": len(words)}
```

LangChain использует docstring как description для LLM. Без docstring модель видит пустое описание и не знает, когда и зачем вызывать этот инструмент. Результат — модель игнорирует tool или вызывает его в неподходящий момент.

```python
@tool
def analyze(text: str) -> dict:
    """Count words in the given text. Use this tool when you need to know the text length and basic statistics."""
    words = text.split()
    return {"count": len(words)}
```

Docstring должен отвечать на два вопроса: что делает tool + когда его использовать.

### 2. Слишком много tools — модель путается

```python
llm_with_tools = llm.bind_tools([
    tool_1, tool_2, tool_3, tool_4, tool_5,
    tool_6, tool_7, tool_8, tool_9, tool_10,
    tool_11, tool_12, tool_13, tool_14, tool_15,
    tool_16, tool_17, tool_18, tool_19, tool_20,
])
```

При >15 tools модель начинает путать похожие инструменты, выбирать неоптимальный tool, или вообще зацикливаться между двумя tools. Каждый tool — это дополнительные токены в контексте.

```python
def get_tools_for_context(task_type: str) -> list:
    base = [count_words]
    specific = {
        "essay": [check_structure, check_citations],
        "code": [run_tests, check_style],
    }
    return base + specific.get(task_type, [])

tools = get_tools_for_context(request.task_type)
llm_with_tools = llm.bind_tools(tools)
```

Подбирай 3-7 tools под конкретную задачу, а не загружай все 20.

### 3. args_schema без Field descriptions

```python
class CheckInput(BaseModel):
    text: str
    style: str
    count: int
```

Модель видит JSON Schema с типами, но без описаний. Она угадывает: `style` — это стиль цитирования? стиль письма? CSS-стиль? `count` — количество чего?

```python
class CheckInput(BaseModel):
    text: str = Field(description="Full essay text to check for citations")
    style: str = Field(description="Citation style: 'apa' or 'mla'")
    count: int = Field(description="Minimum expected number of citations", ge=1, le=50)
```

Описания полей — это инструкции для LLM. Чем точнее описания, тем корректнее аргументы.

### 4. Синхронный HTTP-вызов в tool блокирует event loop

```python
import requests
from langchain_core.tools import tool

@tool
def check_api(text: str) -> dict:
    """Check text via external API."""
    response = requests.post("https://api.example.com/check", json={"text": text})
    return response.json()
```

`requests.post` — синхронный вызов. В async-приложении это блокирует event loop: пока ждём ответа от внешнего API, остальные задачи не выполняются.

```python
import httpx
from langchain_core.tools import tool

@tool
async def check_api(text: str) -> dict:
    """Check text via external API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.example.com/check",
            json={"text": text},
        )
        return response.json()
```

Используй `httpx.AsyncClient` вместо `requests` для HTTP-вызовов в async tools.

### 5. Отсутствие error handling в tool

```python
@tool
def get_rubric(name: str) -> dict:
    """Get rubric by name."""
    return rubrics_db[name]
```

Если `name` нет в `rubrics_db`, KeyError вылетает из tool и крашит весь chain. LLM не получает шанса обработать ошибку и попробовать другой подход.

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def get_rubric(name: str) -> dict:
    """Get rubric by name."""
    if name not in rubrics_db:
        raise ToolException(
            f"Rubric '{name}' not found. Available: {list(rubrics_db.keys())}"
        )
    return rubrics_db[name]
```

С `handle_tool_error=True` ошибка превращается в ToolMessage, модель видит список доступных рубрик и может повторить вызов с правильным именем.

### 6. return_direct без понимания последствий

```python
@tool(return_direct=True)
def count_words(text: str) -> dict:
    """Count words in text."""
    return {"word_count": len(text.split())}
```

С `return_direct=True` результат tool возвращается пользователю напрямую, **минуя LLM**. Модель не видит результат и не может его интерпретировать, добавить контекст или сформировать человекочитаемый ответ. Пользователь получает сырой `{"word_count": 42}` вместо "В вашем эссе 42 слова, что недостаточно для полноценного анализа".

```python
@tool(return_direct=False)
def count_words(text: str) -> dict:
    """Count words in text."""
    return {"word_count": len(text.split())}
```

Используй `return_direct=True` только когда tool возвращает финальный, user-facing результат, не требующий интерпретации (например, сгенерированное изображение или готовый отчёт).

---

## Что читать дальше

- [LangChain Tool Calling](https://python.langchain.com/docs/concepts/tool_calling/) — концепция tool calling в LangChain
- [How to create tools](https://python.langchain.com/docs/how_to/custom_tools/) — практическое руководство по созданию tools
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — как tool calling работает в Claude
- [LangGraph ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode) — автоматическое выполнение tools в графе
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — tool calling в OpenAI API для сравнения подходов

**Следующая тема:** [Тема 12: Multimodal AI](topic_12_multimodal.md) — работа с изображениями и аудио.

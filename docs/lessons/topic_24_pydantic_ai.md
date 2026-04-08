# Тема 24: PydanticAI — type-safe агенты на Pydantic

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 3](topic_03_structured_output.md), [Тема 11](topic_11_tool_use.md)
> **Зависимости:** `pydantic-ai`, `pydantic`, `logfire`

---

## Теория

### 1. Зачем PydanticAI

PydanticAI — фреймворк от команды Pydantic для построения AI-агентов с полной типизацией. Если LangChain — это "Spring для LLM" (много абстракций, гибкость, магия), то PydanticAI — это "FastAPI для LLM" (минимализм, type safety, явные контракты).

Ключевые принципы:

- **Type safety.** Результат агента — Pydantic-модель с валидацией. Не `str`, не `dict`, а типизированный объект. Ошибки ловятся до runtime.
- **Dependency injection.** Зависимости агента (БД, HTTP-клиенты, конфигурация) передаются явно через типизированный контейнер. Как `Depends()` в FastAPI.
- **Model-agnostic.** Единый API для OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama. Смена модели — одна строка.
- **Streaming нативно.** Structured streaming — стриминг частично заполненного Pydantic-объекта. Не "текст по токену", а "JSON по полю".
- **Logfire интеграция.** Observability из коробки через Pydantic Logfire — трейсинг каждого вызова, tool call, retry.

Сравнение подходов:

| Аспект | LangChain | PydanticAI |
|---|---|---|
| Философия | Гибкость, экосистема | Простота, type safety |
| Типизация | Частичная (TypedDict) | Полная (Pydantic models) |
| DI | Нет (глобальный state) | Встроенный (`deps_type`) |
| Structured output | `with_structured_output()` | `result_type` — нативно |
| Streaming | Текстовый или events | Structured streaming |
| Observability | Callbacks, внешние инструменты | Logfire из коробки |
| Кривая обучения | Высокая | Низкая (если знаешь Pydantic) |

### 2. Agent — центральная абстракция

`Agent` — основной класс PydanticAI. Определяет: какую модель использовать, какой system prompt, какие tools доступны, какой тип результата.

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class AssessmentResult(BaseModel):
    score: int
    feedback: str
    strengths: list[str]
    weaknesses: list[str]

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=AssessmentResult,
    system_prompt=(
        "Ты — эксперт-оценщик студенческих эссе. "
        "Оцени работу от 1 до 10 по критериям: тезис, аргументация, "
        "доказательная база, структура."
    ),
)

result = agent.run_sync("Оцени: AI is transforming society in profound ways...")
print(result.data)
print(result.data.score)
print(result.data.feedback)
```

`result.data` — это `AssessmentResult`, не строка. IDE подсказывает поля, mypy проверяет типы, невалидный JSON автоматически retry'ится.

### 3. Dependency Injection

Зависимости агента типизируются через `deps_type`. Это решает проблему "откуда agent берёт данные" — не из глобальных переменных, а из явно переданного контейнера.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
import httpx

@dataclass
class AssessmentDeps:
    http_client: httpx.AsyncClient
    rubric_url: str
    max_score: int = 10

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=AssessmentDeps,
    result_type=AssessmentResult,
)

@agent.system_prompt
async def build_prompt(ctx: RunContext[AssessmentDeps]) -> str:
    response = await ctx.deps.http_client.get(ctx.deps.rubric_url)
    rubric = response.text
    return f"Оцени работу от 1 до {ctx.deps.max_score}. Рубрика:\n{rubric}"
```

`RunContext[AssessmentDeps]` — типизированный доступ к зависимостям внутри system prompt, tools и result validators. IDE знает, что `ctx.deps` — это `AssessmentDeps`, подсказывает поля.

Преимущества DI:
- **Тестируемость.** В тестах передаём mock-зависимости: `TestAssessmentDeps(http_client=mock_client)`.
- **Конфигурируемость.** Разные окружения — разные зависимости (dev/staging/prod).
- **Явность.** Нет скрытых зависимостей от глобального состояния.

### 4. Tools — инструменты агента

Tools в PydanticAI — обычные Python-функции с декоратором `@agent.tool`. Типы параметров автоматически преобразуются в JSON Schema для LLM.

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

class EssayAnalysis(BaseModel):
    word_count: int
    has_citations: bool
    citation_count: int
    has_introduction: bool
    has_conclusion: bool

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=AssessmentDeps,
    result_type=AssessmentResult,
)

@agent.tool
async def analyze_structure(ctx: RunContext[AssessmentDeps], essay: str) -> EssayAnalysis:
    """Анализирует структуру эссе: количество слов, цитирования, наличие введения и заключения."""
    import re
    words = essay.split()
    citations = re.findall(r'\([\w\s]+,\s*\d{4}\)', essay)
    paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
    return EssayAnalysis(
        word_count=len(words),
        has_citations=bool(citations),
        citation_count=len(citations),
        has_introduction=len(paragraphs) >= 3,
        has_conclusion=len(paragraphs) >= 3,
    )

@agent.tool
async def get_similar_essays(ctx: RunContext[AssessmentDeps], topic: str) -> list[str]:
    """Находит ранее оцененные эссе на похожую тему для калибровки."""
    response = await ctx.deps.http_client.get(
        f"{ctx.deps.rubric_url}/similar",
        params={"topic": topic, "limit": 3},
    )
    return response.json()
```

Docstring функции становится описанием tool для LLM. Типы параметров (`essay: str`, `topic: str`) автоматически валидируются.

### 5. Result Validators

Result validators проверяют и корректируют ответ LLM до возврата пользователю. Если валидация не прошла — LLM получает ошибку и retry'ит.

```python
from pydantic_ai import Agent, RunContext, ModelRetry

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=AssessmentResult,
    retries=3,
)

@agent.result_validator
async def validate_assessment(ctx: RunContext, result: AssessmentResult) -> AssessmentResult:
    if not 1 <= result.score <= 10:
        raise ModelRetry(f"Score must be 1-10, got {result.score}")
    if len(result.feedback) < 50:
        raise ModelRetry("Feedback is too short, provide at least 2-3 sentences")
    if not result.strengths:
        raise ModelRetry("Must list at least one strength")
    return result
```

`ModelRetry` — не исключение для пользователя, а инструкция для LLM: "ты ошибся, попробуй ещё раз с учётом этого фидбека". PydanticAI автоматически отправляет ошибку обратно в LLM и повторяет запрос (до `retries` раз).

### 6. Structured Streaming

Обычный streaming — поток токенов текста. Structured streaming в PydanticAI — поток частично заполненного Pydantic-объекта.

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class DetailedAssessment(BaseModel):
    summary: str
    score: int
    criteria_scores: dict[str, int]
    feedback: str
    recommendations: list[str]

agent = Agent("anthropic:claude-sonnet-4-20250514", result_type=DetailedAssessment)

async def stream_assessment(essay: str):
    async with agent.run_stream(f"Оцени:\n\n{essay}") as result:
        async for partial in result.stream_structured(debounce_by=0.1):
            print(f"Partial: summary={partial.get('summary', '...')}, score={partial.get('score', '?')}")
        
        final = await result.get_data()
        print(f"Final: {final}")
```

Клиент получает обновления по мере заполнения полей: сначала `summary`, потом `score`, потом `criteria_scores` и т.д. Это лучше, чем ждать весь JSON — UI может показывать прогресс.

### 7. Multi-agent workflows

PydanticAI поддерживает композицию агентов. Один агент может вызывать другого через tool или напрямую.

```python
analyzer_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=EssayAnalysis,
    system_prompt="Проанализируй структуру и качество эссе.",
)

scorer_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=AssessmentResult,
    system_prompt="На основе анализа выставь оценку от 1 до 10.",
)

async def assess_essay(essay: str) -> AssessmentResult:
    analysis = await analyzer_agent.run(f"Проанализируй:\n\n{essay}")
    scoring_prompt = (
        f"Эссе:\n{essay}\n\n"
        f"Анализ:\n{analysis.data.model_dump_json(indent=2)}\n\n"
        f"Выставь оценку на основе анализа."
    )
    result = await scorer_agent.run(scoring_prompt)
    return result.data
```

Каждый агент специализирован: analyzer разбирает структуру, scorer выставляет оценку. Это разделение улучшает качество — у каждого агента узкая задача и оптимизированный промпт.

### 8. Observability через Logfire

Pydantic Logfire — observability-платформа от команды Pydantic. PydanticAI интегрирован нативно:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

После этого каждый `agent.run()` автоматически создаёт trace с:
- System prompt (финальный, после DI)
- Все сообщения (user, assistant, tool calls, tool results)
- Retries и validation errors
- Токены, latency, стоимость
- Structured result

Logfire dashboard показывает waterfall каждого вызова: промпт → LLM → tool call → tool result → LLM → validation → result.

---

## Справочник API

### Agent

```python
from pydantic_ai import Agent

agent = Agent(
    model: str | Model,
    result_type: type[T] = str,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[D] = NoneType,
    retries: int = 1,
    result_retries: int | None = None,
    tools: Sequence[Tool] = (),
    model_settings: ModelSettings | None = None,
    end_strategy: EndStrategy = "early",
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `model` | `str \| Model` | Модель: `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`, `"gemini-1.5-pro"` |
| `result_type` | `type[T]` | Pydantic model или примитив (`str`, `int`, `bool`) |
| `system_prompt` | `str \| Sequence[str]` | Статический system prompt (или несколько — конкатенируются) |
| `deps_type` | `type[D]` | Тип контейнера зависимостей |
| `retries` | `int` | Макс. retries для tool calls |
| `result_retries` | `int \| None` | Макс. retries для result validation (`None` = retries) |
| `end_strategy` | `EndStrategy` | `"early"` — первый валидный result, `"exhaustive"` — все tools |

### Agent.run / run_sync / run_stream

```python
result = await agent.run(
    user_prompt: str,
    message_history: list[ModelMessage] | None = None,
    model: str | Model | None = None,
    deps: D = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    infer_name: bool = True,
) -> RunResult[T]

result = agent.run_sync(...)  # синхронная версия

async with agent.run_stream(...) as result:  # streaming
    async for chunk in result.stream():
        ...
```

| Параметр | Тип | Описание |
|---|---|---|
| `user_prompt` | `str` | Сообщение пользователя |
| `message_history` | `list \| None` | История сообщений для multi-turn |
| `model` | `str \| None` | Override модели для этого вызова |
| `deps` | `D` | Экземпляр зависимостей |
| `usage_limits` | `UsageLimits \| None` | Лимиты: max_tokens, max_requests |

### RunResult

```python
result.data          # T — типизированный результат
result.all_messages()  # list[ModelMessage] — полная история
result.usage()       # Usage — токены, запросы, стоимость
result.new_messages()  # только новые сообщения
```

### Декораторы

```python
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[D]) -> str: ...

@agent.tool
async def my_tool(ctx: RunContext[D], param: str) -> str: ...

@agent.tool(retries=3)
async def my_tool_with_retries(ctx: RunContext[D], param: str) -> str: ...

@agent.result_validator
async def validate(ctx: RunContext[D], result: T) -> T: ...
```

### ModelSettings

```python
from pydantic_ai.settings import ModelSettings

settings = ModelSettings(
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    timeout: float | None = None,
)
```

### UsageLimits

```python
from pydantic_ai.settings import UsageLimits

limits = UsageLimits(
    request_limit: int | None = None,
    request_tokens_limit: int | None = None,
    response_tokens_limit: int | None = None,
    total_tokens_limit: int | None = None,
)
```

---

## Практика

### Пример 1: Базовый агент с structured output

Минимальный агент для оценки эссе: typed result, system prompt, одна модель.

```python
from pydantic_ai import Agent
from pydantic import BaseModel, Field

class AssessmentResult(BaseModel):
    score: int = Field(ge=1, le=10, description="Оценка от 1 до 10")
    feedback: str = Field(min_length=50, description="Развёрнутый фидбек")
    strengths: list[str] = Field(min_length=1, description="Сильные стороны работы")
    weaknesses: list[str] = Field(description="Слабые стороны работы")

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=AssessmentResult,
    system_prompt=(
        "Ты — опытный преподаватель. Оцени студенческое эссе от 1 до 10. "
        "Критерии: тезис и аргументация, доказательная база, структура, язык."
    ),
    retries=3,
)

result = agent.run_sync(
    "Оцени это эссе:\n\n"
    "The rapid advancement of AI poses fundamental questions about the nature "
    "of work. Brynjolfsson and McAfee (2014) argue for a 'race against the "
    "machine', while Autor (2015) suggests complementarity between human "
    "skills and AI capabilities."
)

print(f"Score: {result.data.score}/10")
print(f"Feedback: {result.data.feedback}")
print(f"Strengths: {result.data.strengths}")
print(f"Weaknesses: {result.data.weaknesses}")
print(f"Tokens: {result.usage()}")
```

### Пример 2: Dependency Injection

Агент с внешними зависимостями: HTTP-клиент для загрузки рубрики, конфигурация.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
import httpx

@dataclass
class Deps:
    http_client: httpx.AsyncClient
    rubric_api: str
    subject: str

class AssessmentResult(BaseModel):
    score: int = Field(ge=1, le=10)
    feedback: str
    criteria_scores: dict[str, int]

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=Deps,
    result_type=AssessmentResult,
)

@agent.system_prompt
async def load_rubric(ctx: RunContext[Deps]) -> str:
    response = await ctx.deps.http_client.get(
        f"{ctx.deps.rubric_api}/rubrics/{ctx.deps.subject}"
    )
    rubric = response.json()
    criteria = "\n".join(f"- {c['name']}: 0-{c['max_score']}" for c in rubric["criteria"])
    return (
        f"Ты — оценщик по предмету {ctx.deps.subject}.\n"
        f"Критерии:\n{criteria}\n"
        f"Оцени работу и выстави баллы по каждому критерию."
    )

async def main():
    async with httpx.AsyncClient() as client:
        deps = Deps(
            http_client=client,
            rubric_api="https://api.example.com",
            subject="computer_science",
        )
        result = await agent.run("Оцени: AI is transforming...", deps=deps)
        print(result.data.criteria_scores)
```

### Пример 3: Tools

Агент с инструментами: подсчёт слов, проверка цитирований, поиск похожих работ.

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import re

class DetailedAssessment(BaseModel):
    score: int
    word_count: int
    citation_count: int
    feedback: str
    recommendations: list[str]

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=DetailedAssessment,
    system_prompt=(
        "Ты — оценщик эссе. Используй инструменты для анализа, "
        "затем выстави оценку от 1 до 10."
    ),
)

@agent.tool_plain
def count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте."""
    return len(text.split())

@agent.tool_plain
def find_citations(text: str) -> dict[str, list[str]]:
    """Находит академические цитирования в тексте (формат: Author, Year)."""
    citations = re.findall(r'\(([^)]+,\s*\d{4})\)', text)
    return {"citations": citations, "count": len(citations)}

@agent.tool_plain
def check_structure(text: str) -> dict[str, bool]:
    """Проверяет наличие введения, основной части и заключения."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return {
        "has_multiple_paragraphs": len(paragraphs) >= 3,
        "paragraph_count": len(paragraphs),
        "has_sufficient_length": len(text.split()) >= 100,
    }

result = agent.run_sync(
    "Проанализируй и оцени:\n\n"
    "The intersection of artificial intelligence and labor economics "
    "presents a nuanced challenge. Frey and Osborne (2013) estimated "
    "47% automation risk, yet Arntz et al. (2016) suggest only 9% of "
    "jobs are fully automatable.\n\n"
    "Furthermore, Autor (2015) argues that technology creates new tasks "
    "even as it automates existing ones. This complementarity effect "
    "suggests a more optimistic outlook for human workers."
)

print(f"Score: {result.data.score}")
print(f"Words: {result.data.word_count}")
print(f"Citations: {result.data.citation_count}")
print(f"Recommendations: {result.data.recommendations}")
```

### Пример 4: Result Validation с ModelRetry

Валидация ответа LLM с автоматическим retry при ошибках.

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

class GradedAssessment(BaseModel):
    thesis_score: int = Field(ge=0, le=3)
    evidence_score: int = Field(ge=0, le=3)
    structure_score: int = Field(ge=0, le=2)
    language_score: int = Field(ge=0, le=2)
    total_score: int = Field(ge=0, le=10)
    grade: str
    feedback: str

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=GradedAssessment,
    retries=3,
)

@agent.result_validator
async def validate_scores(ctx: RunContext, result: GradedAssessment) -> GradedAssessment:
    calculated_total = (
        result.thesis_score + result.evidence_score
        + result.structure_score + result.language_score
    )
    if result.total_score != calculated_total:
        raise ModelRetry(
            f"total_score ({result.total_score}) must equal sum of criteria "
            f"({calculated_total}). Recalculate."
        )

    valid_grades = {"A", "B", "C", "D", "F"}
    if result.grade not in valid_grades:
        raise ModelRetry(f"grade must be one of {valid_grades}, got '{result.grade}'")

    grade_map = {"A": 8, "B": 6, "C": 4, "D": 2, "F": 0}
    if abs(result.total_score - grade_map.get(result.grade, 0)) > 3:
        raise ModelRetry(
            f"Grade '{result.grade}' inconsistent with total_score {result.total_score}"
        )

    return result

result = agent.run_sync("Оцени эссе: AI is changing the world in profound ways...")
print(f"Total: {result.data.total_score}/10 ({result.data.grade})")
print(f"  Thesis: {result.data.thesis_score}/3")
print(f"  Evidence: {result.data.evidence_score}/3")
print(f"  Structure: {result.data.structure_score}/2")
print(f"  Language: {result.data.language_score}/2")
```

### Пример 5: Structured Streaming

Streaming частично заполненного результата — UI обновляется по мере генерации.

```python
import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel

class StreamedAssessment(BaseModel):
    summary: str
    score: int
    detailed_feedback: str
    action_items: list[str]

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=StreamedAssessment,
    system_prompt="Оцени эссе. Заполни все поля последовательно.",
)

async def stream_demo():
    async with agent.run_stream("Оцени: AI is transforming society...") as result:
        async for partial in result.stream_structured(debounce_by=0.05):
            print(f"\r[streaming] {partial}", end="", flush=True)

        final = await result.get_data()
        print(f"\n\nFinal result:")
        print(f"  Score: {final.score}")
        print(f"  Summary: {final.summary}")
        print(f"  Actions: {final.action_items}")
        print(f"  Usage: {result.usage()}")

asyncio.run(stream_demo())
```

### Пример 6: Multi-agent pipeline

Два специализированных агента: analyzer + scorer, композиция через Python.

```python
import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel

class Analysis(BaseModel):
    topic: str
    argument_quality: str
    evidence_summary: str
    structural_issues: list[str]
    overall_impression: str

class FinalScore(BaseModel):
    score: int
    grade: str
    feedback: str
    based_on: str

analyzer = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=Analysis,
    system_prompt="Проанализируй эссе. Определи тему, оцени качество аргументации, суммируй доказательства.",
)

scorer = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=FinalScore,
    system_prompt=(
        "На основе предоставленного анализа выстави оценку от 1 до 10. "
        "Ты не видишь оригинальное эссе — только анализ."
    ),
)

async def assess_pipeline(essay: str) -> FinalScore:
    analysis_result = await analyzer.run(f"Проанализируй:\n\n{essay}")
    analysis = analysis_result.data

    score_result = await scorer.run(
        f"Анализ эссе:\n{analysis.model_dump_json(indent=2)}\n\nВыстави оценку."
    )
    return score_result.data

result = asyncio.run(assess_pipeline(
    "The rapid advancement of AI poses fundamental questions about work. "
    "Brynjolfsson and McAfee (2014) argue for a 'race against the machine'."
))

print(f"{result.grade} ({result.score}/10): {result.feedback}")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | Agent + result_type: typed structured output |
| 2 | Dependency injection через deps_type |
| 3 | Tools с автоматической JSON Schema генерацией |
| 4 | Result validators + ModelRetry |
| 5 | Structured streaming |
| 6 | Multi-agent composition |

---

## Чеклист самопроверки

- [ ] В чём отличие `result_type` в PydanticAI от `with_structured_output()` в LangChain?
- [ ] Что такое `RunContext[D]` и зачем он нужен? Чем отличается от глобальных переменных?
- [ ] Как работает `ModelRetry`? Что происходит при raise внутри result validator?
- [ ] В чём разница между `@agent.tool` и `@agent.tool_plain`?
- [ ] Что такое structured streaming? Чем отличается от обычного text streaming?
- [ ] Как PydanticAI обеспечивает type safety? Где именно работает валидация?
- [ ] Зачем нужен `end_strategy="exhaustive"`? Когда он полезен?
- [ ] Как тестировать PydanticAI-агента? Какую роль играет DI?
- [ ] Как организовать multi-agent workflow в PydanticAI?
- [ ] Зачем нужен Logfire? Что он показывает для PydanticAI?

---

## Частые ошибки

### 1. result_type=dict вместо Pydantic model

```python
agent = Agent("openai:gpt-4o", result_type=dict)
```

```python
from pydantic import BaseModel

class MyResult(BaseModel):
    score: int
    feedback: str

agent = Agent("openai:gpt-4o", result_type=MyResult)
```

`dict` не даёт валидации, не даёт autocompletion, не даёт описаний полей для LLM. Pydantic model с `Field(description=...)` — и валидация, и подсказки для модели.

### 2. Sync вызов в async контексте

```python
async def handler():
    result = agent.run_sync("...")
```

```python
async def handler():
    result = await agent.run("...")
```

`run_sync()` блокирует event loop. В async-коде (FastAPI, aiohttp) используй `agent.run()` с `await`.

### 3. Tool без docstring

```python
@agent.tool_plain
def count_words(text: str) -> int:
    return len(text.split())
```

```python
@agent.tool_plain
def count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте. Используй для оценки объёма работы."""
    return len(text.split())
```

Docstring — это описание tool для LLM. Без него модель не понимает, когда и зачем вызывать инструмент. Описывай и назначение, и когда использовать.

### 4. Забыли передать deps

```python
agent = Agent("openai:gpt-4o", deps_type=MyDeps, result_type=str)
result = await agent.run("Hello")
```

```python
deps = MyDeps(api_key="...", db=connection)
result = await agent.run("Hello", deps=deps)
```

Если `deps_type` указан, `deps` обязателен при вызове. Без него system prompt и tools не получат зависимости и упадут с ошибкой.

### 5. Нет retries для result validation

```python
agent = Agent("openai:gpt-4o", result_type=StrictModel)

@agent.result_validator
async def validate(ctx, result):
    if result.score > 10:
        raise ModelRetry("Score must be <= 10")
    return result
```

```python
agent = Agent("openai:gpt-4o", result_type=StrictModel, retries=3)

@agent.result_validator
async def validate(ctx, result):
    if result.score > 10:
        raise ModelRetry("Score must be <= 10")
    return result
```

По умолчанию `retries=1` — одна попытка. Если result validator часто срабатывает, увеличь `retries` до 3-5. Каждый retry — дополнительный API-вызов и стоимость.

---

## Что читать дальше

- [PydanticAI Documentation](https://ai.pydantic.dev/) — полная документация
- [PydanticAI Agents](https://ai.pydantic.dev/agents/) — Agent API reference
- [PydanticAI Tools](https://ai.pydantic.dev/tools/) — tools и function calling
- [PydanticAI Dependencies](https://ai.pydantic.dev/dependencies/) — dependency injection
- [PydanticAI Results](https://ai.pydantic.dev/results/) — result types и streaming
- [PydanticAI + Logfire](https://ai.pydantic.dev/logfire/) — observability
- [Pydantic Logfire](https://logfire.pydantic.dev/) — платформа observability
- [PydanticAI GitHub](https://github.com/pydantic/pydantic-ai) — исходный код

**Предыдущая тема:** [Тема 23: Databricks](topic_23_databricks.md) — MLflow, Model Serving, Vector Search.
**Следующая тема:** [Тема 25: DSPy](topic_25_dspy.md) — программирование LLM-пайплайнов.

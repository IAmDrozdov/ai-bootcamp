# Тема 6: LangGraph + Agents

> **Пререквизиты:** темы 1–4 (LCEL chains, structured output), рекомендуется тема 5 (RAG)
> **Что добавляем в проект:** `app/api/v1/graph.py`, `app/graph/assessment_graph.py`, `app/graph/tools.py`, `app/schemas/graph.py`
> **Зависимости:** `langgraph` (группа `agents`)

---

## Теория

### 1. От цепочек к графам

LCEL chains из темы 2 — это **линейные** пайплайны: `prompt | llm | parser`. Каждый шаг выполняется ровно один раз, в строгом порядке, без возможности вернуться назад. Это идеально для простых задач, но реальные workflow'ы часто требуют:

- **Циклов** — повторить шаг, пока результат не удовлетворит условию (например, LLM вызывает tools в цикле, пока не соберёт достаточно информации)
- **Условных переходов** — выбрать разную ветку обработки в зависимости от входных данных (короткое эссе → быстрая модель, длинное → мощная)
- **Общего состояния** — данные, доступные всем шагам и накапливающиеся по мере выполнения (анализ → оценка → ревью, каждый шаг видит результаты предыдущих)
- **Пауз** — остановка для человеческого подтверждения перед критическим действием

**Когда переходить от LCEL к LangGraph:**

| Сценарий | LCEL | LangGraph |
|----------|------|-----------|
| prompt → LLM → parse | Идеально | Излишне |
| A → B → C (линейно) | Хорошо | Можно, но проще LCEL |
| A → (условие) → B или C | Неудобно | Удобно |
| Цикл: LLM → tool → LLM → ... | Невозможно | Нативно |
| Человек подтверждает промежуточный результат | Нельзя | Встроено |
| Нужно сохранять и восстанавливать состояние | Вручную | Checkpointing |

LangGraph — это **фреймворк для построения stateful, multi-step applications** на базе LLM. Он моделирует workflow как направленный граф, где узлы — это функции, рёбра — переходы между ними, а состояние — типизированный словарь, доступный всем узлам.

В нашем проекте LangGraph позволяет строить многошаговые пайплайны оценки: анализ работы → выбор модели → оценка → ревью — где каждый шаг может принимать решения на основе результатов предыдущих.

### 2. StateGraph — основная абстракция

StateGraph — направленный граф с тремя компонентами:

- **State** — типизированный словарь (`TypedDict`), определяющий форму данных, которые проходят через граф
- **Nodes** — именованные функции, которые читают state и возвращают обновления
- **Edges** — переходы между узлами (обычные и условные)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class AssessmentState(TypedDict):
    student_work: str
    rubric: str
    analysis: str
    scores: dict
    final_result: dict


graph = StateGraph(AssessmentState)

graph.add_node("analyze", analyze_node)
graph.add_node("score", score_node)
graph.add_node("format", format_node)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "score")
graph.add_edge("score", "format")
graph.add_edge("format", END)

app = graph.compile()
```

`START` и `END` — специальные константы, обозначающие точку входа и выхода графа. `START` не является узлом — это маркер, указывающий, с какого узла начать выполнение. Аналогично `END` сигнализирует завершение.

**Компиляция** (`graph.compile()`) валидирует граф: проверяет, что все узлы достижимы из START, все пути ведут к END, и нет висячих рёбер. Если граф невалиден, ты получишь ошибку на этапе компиляции, а не в runtime.

Скомпилированный граф реализует Runnable-интерфейс — его можно вызывать через `invoke`, `ainvoke`, `stream`, `astream`, как любую LCEL-цепочку.

### 3. Nodes — функции-обработчики

Node — обычная Python-функция (sync или async), принимающая текущий state и возвращающая **словарь с обновлениями**:

```python
async def analyze_node(state: AssessmentState) -> dict:
    work = state["student_work"]
    word_count = len(work.split())
    paragraphs = len([p for p in work.split("\n\n") if p.strip()])
    return {"analysis": f"Words: {word_count}, Paragraphs: {paragraphs}"}
```

Критически важно: node возвращает **только изменённые поля**, а не весь state. LangGraph автоматически мержит возвращённый dict с текущим состоянием. Если вернуть весь state, все поля перезапишутся, включая те, которые ты не хотел менять.

```python
async def bad_node(state):
    return {**state, "analysis": "done"}

async def good_node(state):
    return {"analysis": "done"}
```

**Sync vs async:** для node'ов, вызывающих LLM или другие I/O-операции, используй `async def`. Для чистых вычислений (подсчёт слов, форматирование) подойдёт `def`. LangGraph корректно обрабатывает оба варианта.

**Порядок выполнения:** в линейном графе (A → B → C) узлы выполняются строго последовательно. Каждый узел видит state, обновлённый всеми предыдущими. В графе с параллельными ветками (A → B, A → C, B → D, C → D) узлы B и C могут выполняться параллельно, а D получит мерж их результатов.

### 4. State reducers

По умолчанию каждое обновление **заменяет** предыдущее значение поля. Если node A записал `{"scores": {"thesis": 20}}`, а node B записал `{"scores": {"evidence": 18}}`, итоговое значение `scores` будет `{"evidence": 18}` — результат A потерян.

Для списков, которые должны **накапливаться** (а не перезаписываться), используются **reducers** через `Annotated`:

```python
from typing import Annotated
from operator import add


class State(TypedDict):
    messages: Annotated[list, add]
    final_answer: str
```

С `Annotated[list, add]`: если текущее значение `messages` — `[msg1, msg2]`, а node вернёт `{"messages": [msg3]}`, результат будет `[msg1, msg2, msg3]` (append). Без `Annotated` результат был бы `[msg3]` (replace).

Модуль `operator` предоставляет готовые reducers:
- `operator.add` — конкатенация для списков и строк, сложение для чисел
- Можно написать **кастомный reducer** — любая функция `(old, new) -> merged`:

```python
def keep_max(old: int, new: int) -> int:
    return max(old, new)


class State(TypedDict):
    best_score: Annotated[int, keep_max]
    messages: Annotated[list, add]
    current_step: str
```

**Когда использовать reducers:**
- `messages` — почти всегда с `add` (чат-история накапливается)
- `tool_results` — с `add` (результаты tools добавляются)
- `final_answer`, `assessment`, `analysis` — без reducer (replace, нужно последнее значение)
- Счётчики, max/min значения — кастомный reducer

### 5. Conditional edges — ветвление

Условные рёбра позволяют выбирать следующий node на основе текущего state:

```python
def route_by_complexity(state: AssessmentState) -> str:
    word_count = len(state["student_work"].split())
    if word_count < 200:
        return "quick_assess"
    return "full_assess"


graph.add_conditional_edges(
    "analyze",
    route_by_complexity,
    {"quick_assess": "quick_assess", "full_assess": "full_assess"},
)
```

`add_conditional_edges` принимает три аргумента:
1. **source** — имя узла, из которого выходят условные рёбра
2. **path** — функция-роутер, принимающая state и возвращающая строку-ключ
3. **path_map** — словарь `{ключ: имя_узла}`. Если ключ совпадает с именем узла, path_map можно опустить

Router-функция — это **чистая функция**, не меняющая state. Она только анализирует текущее состояние и возвращает строку — имя следующего узла. Это разделение логики принятия решения (router) и обработки данных (node) делает граф прозрачным и тестируемым.

**Множественное ветвление:**

```python
def route_by_quality(state) -> str:
    score = state.get("preliminary_score", 0)
    if score > 80:
        return "quick_review"
    elif score > 50:
        return "standard_review"
    return "detailed_review"


graph.add_conditional_edges(
    "pre_assess",
    route_by_quality,
    {
        "quick_review": "quick_review",
        "standard_review": "standard_review",
        "detailed_review": "detailed_review",
    },
)
```

После ветвления пути могут **сходиться** к одному узлу. Например, и `quick_assess`, и `full_assess` могут вести к `format_result` → `END`. Это позволяет иметь общую пост-обработку независимо от выбранного пути.

### 6. Tool calling — LLM вызывает инструменты

Tools — это функции с описанием, доступные LLM. Модель **сама решает**, когда и какой tool вызвать, основываясь на docstring и параметрах:

```python
from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())


@tool
def check_citations(text: str) -> dict:
    """Check if the text contains academic citations in APA format."""
    import re
    citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text)
    return {"count": len(citations), "citations": citations}
```

LLM видит описание каждого tool (name + docstring + параметры с типами) и на каждом шаге решает:
1. Нужно ли вызвать tool? (может ответить текстом без tool)
2. Какой tool вызвать? (из доступного набора)
3. С какими аргументами? (извлекает из контекста)

Привязка tools к LLM:

```python
llm_with_tools = llm.bind_tools([count_words, check_citations])
```

`bind_tools` не вызывает tools — он «рассказывает» модели о доступных инструментах. Модель возвращает `AIMessage` с полем `tool_calls` — списком tool-вызовов, которые она хочет сделать. Фактическое выполнение tool'ов — ответственность приложения.

**Tool-calling loop в LangGraph** — стандартный паттерн (ReAct):

```
START → agent → (есть tool_calls?) → tools → agent → ... → (нет tool_calls?) → END
```

Узел `agent` вызывает LLM с привязанными tools. Если LLM решает вызвать tool, ответ содержит `tool_calls`. Conditional edge направляет к узлу `tools`, который выполняет вызовы и возвращает результаты в state. Затем снова вызывается `agent` — теперь LLM видит результаты tools и может решить: вызвать ещё один tool, или ответить текстом.

`ToolNode` из `langgraph.prebuilt` автоматизирует выполнение tools:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([count_words, check_citations])
```

`ToolNode` читает последнее сообщение из `state["messages"]`, извлекает `tool_calls`, выполняет каждый tool и возвращает `ToolMessage` с результатами. Для работы ToolNode state должен содержать поле `messages: Annotated[list, add]`.

**Правила проектирования tools:**
- Docstring — это **промпт для LLM**: он определяет, когда и как модель вызовет tool. Пиши ясные, конкретные описания.
- Типизируй параметры — LLM использует типы для генерации аргументов.
- Возвращай структурированные данные (dict), а не строки — LLM лучше интерпретирует структурированный output.
- Избегай side effects — tool не должен записывать в базу или отправлять email без явного подтверждения.

### 7. Human-in-the-loop

`interrupt_before` останавливает граф **перед** указанным узлом, давая человеку возможность проверить промежуточный результат:

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["publish_assessment"],
)
```

Поток выполнения:
1. Граф выполняется до узла `publish_assessment` → **останавливается**
2. Приложение показывает промежуточный результат (draft оценки) человеку
3. Человек проверяет и подтверждает (или отклоняет)
4. Приложение возобновляет граф: `await app.ainvoke(None, config)`
5. Граф продолжает с `publish_assessment` → END

`interrupt_after` аналогичен, но останавливает **после** выполнения узла — полезно, когда нужно показать результат узла и дождаться подтверждения перед переходом к следующему.

**Обязательное требование:** interrupt работает только с checkpointer. Без checkpointer'а состояние графа теряется при остановке. Checkpointer сохраняет state после каждого узла, позволяя возобновить выполнение позже.

**Модификация state при interrupt:** человек может не только подтвердить, но и изменить state перед продолжением:

```python
current_state = await app.aget_state(config)
await app.aupdate_state(config, {"assessment": corrected_assessment})
result = await app.ainvoke(None, config)
```

Это позволяет реализовать паттерн «approve / edit / reject»:
- **Approve** — продолжить без изменений
- **Edit** — изменить state и продолжить
- **Reject** — откатиться к предыдущему узлу и перезапустить

### 8. Checkpointing — сохранение состояния

Checkpointer сохраняет полный snapshot state после выполнения каждого узла. Это даёт:

- **Восстановление после ошибки** — если узел упал, можно перезапустить с последнего сохранённого state, а не с начала
- **Human-in-the-loop** — пауза и возобновление (как описано выше)
- **Time travel** — «перемотка» графа к любому предыдущему шагу для отладки или повторного выполнения
- **Многопользовательские сессии** — каждый `thread_id` имеет свою историю state'ов

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "assessment-123"}}
result = await app.ainvoke(initial_state, config)

history = [s async for s in app.aget_state_history(config)]
```

`MemorySaver` хранит state в памяти процесса — подходит для разработки и тестирования. Для production используют persistent checkpointer'ы:
- `SqliteSaver` — файловое хранилище
- `PostgresSaver` — для distributed deployments
- Кастомные реализации через `BaseCheckpointSaver`

**thread_id** — идентификатор сессии. Разные thread_id — разные цепочки state'ов. Это позволяет параллельно обрабатывать множество пользователей, каждый со своим состоянием графа.

---

## Справочник API

### `StateGraph`

**Импорт:** `from langgraph.graph import StateGraph`

Конструктор графа с типизированным состоянием. Определяет узлы, рёбра и компилируется в исполняемый граф.

**Конструктор:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `state_schema` | `type` | TypedDict-класс, определяющий форму state |

**Методы:**

| Метод | Параметры | Описание |
|-------|-----------|----------|
| `add_node(name, node)` | `name: str`, `node: Callable \| Runnable` | Добавить узел с именем и обработчиком |
| `add_edge(source, target)` | `source: str`, `target: str` | Добавить безусловное ребро между узлами |
| `add_conditional_edges(source, path, path_map)` | `source: str`, `path: Callable[[State], str]`, `path_map: dict \| None` | Добавить условные рёбра; `path` — роутер-функция |
| `compile(checkpointer, interrupt_before, interrupt_after)` | см. ниже | Скомпилировать в исполняемый граф |

**Параметры `compile()`:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `checkpointer` | `BaseCheckpointSaver \| None` | `None` | Checkpointer для сохранения состояния |
| `interrupt_before` | `list[str] \| None` | `None` | Имена узлов, перед которыми ставить паузу |
| `interrupt_after` | `list[str] \| None` | `None` | Имена узлов, после которых ставить паузу |

**Пример:**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class MyState(TypedDict):
    input: str
    output: str


graph = StateGraph(MyState)
graph.add_node("process", lambda state: {"output": state["input"].upper()})
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

result = app.invoke({"input": "hello", "output": ""})
```

### `START` / `END`

**Импорт:** `from langgraph.graph import START, END`

Специальные константы-маркеры для определения входа и выхода графа.

| Константа | Описание |
|-----------|----------|
| `START` | Точка входа графа. `add_edge(START, "first_node")` определяет начальный узел |
| `END` | Точка выхода. `add_edge("last_node", END)` завершает выполнение и возвращает финальный state |

`START` и `END` не являются узлами — через них не выполняются функции. Это маркеры топологии графа.

### `MemorySaver`

**Импорт:** `from langgraph.checkpoint.memory import MemorySaver`

In-memory checkpointer для разработки и тестирования. Хранит snapshots состояний в памяти процесса.

**Конструктор:** `MemorySaver()` — без параметров.

**Пример:**

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}
result = await app.ainvoke({"input": "test"}, config)
```

Для production замени на `SqliteSaver` или `PostgresSaver`.

### `@tool`

**Импорт:** `from langchain_core.tools import tool`

Декоратор, создающий LangChain tool из обычной функции. Docstring функции становится описанием tool для LLM.

**Параметры декоратора (все опциональные):**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `name` | `str \| None` | Имя tool (по умолчанию — имя функции) |
| `description` | `str \| None` | Описание (по умолчанию — docstring) |
| `args_schema` | `type[BaseModel] \| None` | Pydantic-модель для валидации аргументов |
| `return_direct` | `bool` | Если `True`, результат tool возвращается пользователю напрямую |

**Пример:**

```python
from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())


print(count_words.name)
print(count_words.description)
print(count_words.args)
```

Типы параметров функции автоматически конвертируются в JSON Schema для LLM. Используй простые типы: `str`, `int`, `float`, `bool`, `list`, `dict`.

### `CompiledGraph`

Результат `graph.compile()`. Реализует Runnable-интерфейс LangChain.

**Методы выполнения:**

| Метод | Описание |
|-------|----------|
| `invoke(input, config) -> dict` | Синхронное выполнение; возвращает финальный state |
| `ainvoke(input, config) -> dict` | Асинхронное выполнение |
| `stream(input, config, stream_mode) -> Iterator` | Синхронный стриминг |
| `astream(input, config, stream_mode) -> AsyncIterator` | Асинхронный стриминг |

**Методы для работы с состоянием (требуют checkpointer):**

| Метод | Описание |
|-------|----------|
| `get_state(config) -> StateSnapshot` | Получить текущий state для thread_id |
| `aget_state(config) -> StateSnapshot` | Async-версия |
| `update_state(config, values) -> dict` | Обновить state (для human-in-the-loop) |
| `get_state_history(config) -> Iterator` | История state'ов |

**stream_mode:**

| Значение | Описание |
|----------|----------|
| `"values"` | Полный state после каждого узла |
| `"updates"` | Только обновления от каждого узла |
| `"debug"` | Детальная информация (input, output, metadata для каждого шага) |

**Пример:**

```python
config = {"configurable": {"thread_id": "demo-1"}}

async for event in app.astream(initial_state, config, stream_mode="updates"):
    print(event)
```

### `TypedDict` с Annotated reducers

**Импорт:**

```python
from typing import TypedDict, Annotated
from operator import add
```

По умолчанию LangGraph **заменяет** значение поля при обновлении. `Annotated` с reducer-функцией меняет это поведение:

| Паттерн | Поведение | Когда использовать |
|---------|-----------|-------------------|
| `field: str` | Replace — новое значение заменяет старое | Единичные значения: answer, analysis |
| `field: Annotated[list, add]` | Append — `operator.add` конкатенирует списки | Накапливающиеся данные: messages, tool_results |
| `field: Annotated[int, custom_fn]` | Custom — произвольная логика мержа | Счётчики, агрегаты |

**Пример:**

```python
from typing import TypedDict, Annotated
from operator import add


class ChatState(TypedDict):
    messages: Annotated[list, add]
    final_answer: str
    iteration_count: int
```

### `interrupt_before` / `interrupt_after`

**Используются в:** `graph.compile(interrupt_before=[...], interrupt_after=[...])`

Параметры для human-in-the-loop. Принимают список имён узлов, перед/после которых граф приостанавливается.

| Параметр | Когда останавливается | Типичное применение |
|----------|-----------------------|--------------------|
| `interrupt_before` | **До** выполнения узла | Подтверждение перед отправкой, публикацией |
| `interrupt_after` | **После** выполнения узла | Ревью результата, подтверждение перед следующим шагом |

**Обязательное условие:** checkpointer **должен** быть задан. Без него `interrupt` вызовет ошибку.

**Пример:**

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["publish"],
)

config = {"configurable": {"thread_id": "review-1"}}

result = await app.ainvoke(initial_state, config)

state = await app.aget_state(config)
print("Draft:", state.values["assessment"])

result = await app.ainvoke(None, config)
```

---

## Практика: роутер `/api/v1/graph`

Мы создадим четыре эндпоинта, демонстрирующих разные возможности LangGraph:

| Эндпоинт | Что делает | Концепт |
|----------|------------|---------|
| `POST /assess` | Линейный граф: prepare → assess → format | StateGraph, nodes, edges |
| `POST /assess/routed` | Conditional routing по длине работы | Conditional edges |
| `POST /assess/tools` | LLM вызывает tools для анализа | Tool calling, ToolNode, цикл |
| `POST /assess/reviewed` | Draft → review (двойная проверка) | Multi-step pipeline |

### Шаг 1. Схемы данных — `app/schemas/graph.py`

```python
from pydantic import BaseModel

from app.schemas.assessment import AssessmentResponse


class GraphAssessRequest(BaseModel):
    student_work: str
    rubric_id: str | None = "essay_default"


class GraphAssessResponse(BaseModel):
    assessment: AssessmentResponse
    nodes_executed: list[str]


class RoutedAssessResponse(BaseModel):
    assessment: AssessmentResponse
    model_used: str
    word_count: int
    route: str


class ToolCallInfo(BaseModel):
    tool_name: str
    tool_input: str
    tool_output: str


class ToolsAssessResponse(BaseModel):
    assessment: AssessmentResponse
    tool_calls: list[ToolCallInfo]


class ReviewedAssessResponse(BaseModel):
    draft_assessment: AssessmentResponse
    final_assessment: AssessmentResponse
    review_notes: str
```

Каждый эндпоинт возвращает свой response-тип с дополнительной метаинформацией: какие узлы выполнились, какая модель использовалась, какие tools были вызваны.

### Шаг 2. Tools — `app/graph/tools.py`

```python
import re

from langchain_core.tools import tool


@tool
def count_words(text: str) -> dict:
    """Count words, sentences, and paragraphs in the given text."""
    words = len(text.split())
    sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    return {
        "words": words,
        "sentences": max(sentences, 1),
        "paragraphs": max(paragraphs, 1),
    }


@tool
def check_structure(text: str) -> dict:
    """Check essay structure: introduction, conclusion, and transition words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    has_intro = len(paragraphs) > 0 and len(paragraphs[0].split()) > 20
    has_conclusion = len(paragraphs) > 1 and len(paragraphs[-1].split()) > 20
    transitions = [
        "however", "moreover", "furthermore", "in addition",
        "therefore", "consequently", "nevertheless", "in contrast",
    ]
    found = sum(1 for t in transitions if t.lower() in text.lower())
    return {
        "paragraph_count": len(paragraphs),
        "has_introduction": has_intro,
        "has_conclusion": has_conclusion,
        "transitions_found": found,
    }


@tool
def check_citations(text: str) -> dict:
    """Find academic citations in APA format, e.g. (Smith, 2023) or (Johnson et al., 2024)."""
    pattern = r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)"
    citations = re.findall(pattern, text)
    return {"count": len(citations), "citations": citations}
```

Каждый tool — чистая функция с понятным docstring. LLM использует docstring для принятия решения, когда вызывать tool. Обрати внимание: возвращаем `dict`, а не строку — структурированные данные LLM интерпретирует точнее.

### Шаг 3. Графы — `app/graph/assessment_graph.py`

```python
import json
from operator import add
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.graph.tools import check_citations, check_structure, count_words
from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse


class SimpleState(TypedDict):
    student_work: str
    rubric: str
    analysis: str
    assessment: dict | None


class RoutedState(TypedDict):
    student_work: str
    rubric: str
    word_count: int
    route: str
    model_used: str
    assessment: dict | None


class ToolsState(TypedDict):
    student_work: str
    rubric: str
    messages: Annotated[list, add]
    assessment: dict | None


class ReviewedState(TypedDict):
    student_work: str
    rubric: str
    analysis: str
    draft_assessment: dict | None
    review_notes: str
    final_assessment: dict | None


def _build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", ASSESSMENT_SYSTEM_PROMPT),
        ("human", "Please assess the following student work:\n\n{student_work}"),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )


def build_simple_graph(llm: ChatAnthropic):
    async def prepare(state: SimpleState) -> dict:
        work = state["student_work"]
        word_count = len(work.split())
        paragraphs = len([p for p in work.split("\n\n") if p.strip()])
        return {"analysis": f"Words: {word_count}, Paragraphs: {paragraphs}"}

    async def assess(state: SimpleState) -> dict:
        prompt = _build_prompt()
        structured_llm = llm.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "rubric": state["rubric"],
        })
        return {"assessment": result.model_dump()}

    async def format_result(state: SimpleState) -> dict:
        assessment = dict(state["assessment"])
        assessment["_analysis"] = state["analysis"]
        return {"assessment": assessment}

    graph = StateGraph(SimpleState)
    graph.add_node("prepare", prepare)
    graph.add_node("assess", assess)
    graph.add_node("format_result", format_result)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "assess")
    graph.add_edge("assess", "format_result")
    graph.add_edge("format_result", END)
    return graph.compile()


def build_routed_graph(llm_fast: ChatAnthropic, llm_full: ChatAnthropic):
    async def analyze(state: RoutedState) -> dict:
        return {"word_count": len(state["student_work"].split())}

    def route_by_length(state: RoutedState) -> str:
        if state["word_count"] < 200:
            return "quick_assess"
        return "full_assess"

    async def quick_assess(state: RoutedState) -> dict:
        prompt = _build_prompt()
        structured_llm = llm_fast.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "rubric": state["rubric"],
        })
        return {
            "assessment": result.model_dump(),
            "route": "quick",
            "model_used": llm_fast.model,
        }

    async def full_assess(state: RoutedState) -> dict:
        prompt = _build_prompt()
        structured_llm = llm_full.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "rubric": state["rubric"],
        })
        return {
            "assessment": result.model_dump(),
            "route": "full",
            "model_used": llm_full.model,
        }

    graph = StateGraph(RoutedState)
    graph.add_node("analyze", analyze)
    graph.add_node("quick_assess", quick_assess)
    graph.add_node("full_assess", full_assess)
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_by_length,
        {"quick_assess": "quick_assess", "full_assess": "full_assess"},
    )
    graph.add_edge("quick_assess", END)
    graph.add_edge("full_assess", END)
    return graph.compile()


def build_tools_graph(llm: ChatAnthropic):
    tools = [count_words, check_structure, check_citations]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent(state: ToolsState) -> dict:
        if not state["messages"]:
            msg = HumanMessage(
                content=(
                    "Analyze this student work using the available tools, "
                    "then summarize your findings.\n\n"
                    f"Student work:\n{state['student_work']}"
                )
            )
            response = await llm_with_tools.ainvoke([msg])
            return {"messages": [msg, response]}
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: ToolsState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "assess"

    async def assess(state: ToolsState) -> dict:
        tool_parts = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_parts.append(f"{msg.name}: {msg.content}")
        tool_summary = "\n".join(tool_parts)

        system = ASSESSMENT_SYSTEM_PROMPT + "\n\n## Tool Analysis Results\n{tool_results}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Please assess the following student work:\n\n{student_work}"),
        ]).partial(
            few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
            few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
        )
        structured_llm = llm.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "rubric": state["rubric"],
            "tool_results": tool_summary,
        })
        return {"assessment": result.model_dump()}

    graph = StateGraph(ToolsState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("assess", assess)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "assess": "assess"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("assess", END)
    return graph.compile()


def build_reviewed_graph(llm: ChatAnthropic):
    async def analyze(state: ReviewedState) -> dict:
        work = state["student_work"]
        word_count = len(work.split())
        paragraphs = len([p for p in work.split("\n\n") if p.strip()])
        return {"analysis": f"Words: {word_count}, Paragraphs: {paragraphs}"}

    async def draft_assess(state: ReviewedState) -> dict:
        prompt = _build_prompt()
        structured_llm = llm.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "rubric": state["rubric"],
        })
        return {"draft_assessment": result.model_dump()}

    async def review(state: ReviewedState) -> dict:
        review_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior academic reviewer. Check this assessment for "
                "fairness, consistency between scores and feedback, and "
                "constructiveness of suggestions. If adjustments are needed, "
                "provide the corrected assessment.",
            ),
            (
                "human",
                "Student work:\n{student_work}\n\n"
                "Analysis: {analysis}\n\n"
                "Draft assessment to review:\n{draft}\n\n"
                "Provide your reviewed final assessment.",
            ),
        ])
        structured_llm = llm.with_structured_output(AssessmentResponse)
        chain = review_prompt | structured_llm
        result = await chain.ainvoke({
            "student_work": state["student_work"],
            "analysis": state["analysis"],
            "draft": json.dumps(state["draft_assessment"], indent=2),
        })
        return {
            "final_assessment": result.model_dump(),
            "review_notes": "Reviewed for fairness, score-feedback alignment, and constructiveness",
        }

    graph = StateGraph(ReviewedState)
    graph.add_node("analyze", analyze)
    graph.add_node("draft_assess", draft_assess)
    graph.add_node("review", review)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "draft_assess")
    graph.add_edge("draft_assess", "review")
    graph.add_edge("review", END)
    return graph.compile()
```

Файл содержит четыре builder-функции, по одной на каждый эндпоинт:

- **`build_simple_graph`** — линейный граф из трёх узлов. Демонстрирует базовые StateGraph, add_node, add_edge.
- **`build_routed_graph`** — conditional edge на основе word count. Принимает две LLM: быструю (haiku) и мощную (sonnet).
- **`build_tools_graph`** — ReAct-паттерн с ToolNode. Цикл `agent → tools → agent` до тех пор, пока LLM не перестанет вызывать tools.
- **`build_reviewed_graph`** — линейный, но с двойным LLM-вызовом: draft-оценка и ревью. Ревьюер получает draft и может скорректировать баллы.

### Шаг 4. Роутер — `app/api/v1/graph.py`

```python
from fastapi import APIRouter, HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage

from app.dependencies import LLMDep, RubricStoreDep, SettingsDep
from app.graph.assessment_graph import (
    build_reviewed_graph,
    build_routed_graph,
    build_simple_graph,
    build_tools_graph,
)
from app.schemas.assessment import AssessmentResponse
from app.schemas.graph import (
    GraphAssessRequest,
    GraphAssessResponse,
    ReviewedAssessResponse,
    RoutedAssessResponse,
    ToolCallInfo,
    ToolsAssessResponse,
)

router = APIRouter(prefix="/graph", tags=["lesson-6-graph"])


@router.post("/assess")
async def graph_assess(
    request: GraphAssessRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> GraphAssessResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    graph = build_simple_graph(llm)
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "analysis": "",
        "assessment": None,
    })

    assessment = AssessmentResponse(**result["assessment"])
    return GraphAssessResponse(
        assessment=assessment,
        nodes_executed=["prepare", "assess", "format_result"],
    )


@router.post("/assess/routed")
async def routed_assess(
    request: GraphAssessRequest,
    rubrics: RubricStoreDep,
    settings: SettingsDep,
) -> RoutedAssessResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    llm_fast = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )
    llm_full = ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
    )

    graph = build_routed_graph(llm_fast, llm_full)
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "word_count": 0,
        "route": "",
        "model_used": "",
        "assessment": None,
    })

    assessment = AssessmentResponse(**result["assessment"])
    return RoutedAssessResponse(
        assessment=assessment,
        model_used=result["model_used"],
        word_count=result["word_count"],
        route=result["route"],
    )


@router.post("/assess/tools")
async def tools_assess(
    request: GraphAssessRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> ToolsAssessResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    graph = build_tools_graph(llm)
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "messages": [],
        "assessment": None,
    })

    tool_calls = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            tool_calls.append(ToolCallInfo(
                tool_name=msg.name,
                tool_input="",
                tool_output=str(msg.content),
            ))

    assessment = AssessmentResponse(**result["assessment"])
    return ToolsAssessResponse(assessment=assessment, tool_calls=tool_calls)


@router.post("/assess/reviewed")
async def reviewed_assess(
    request: GraphAssessRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> ReviewedAssessResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    graph = build_reviewed_graph(llm)
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "analysis": "",
        "draft_assessment": None,
        "review_notes": "",
        "final_assessment": None,
    })

    draft = AssessmentResponse(**result["draft_assessment"])
    final = AssessmentResponse(**result["final_assessment"])
    return ReviewedAssessResponse(
        draft_assessment=draft,
        final_assessment=final,
        review_notes=result["review_notes"],
    )


def _resolve_rubric(rubric_id, rubrics):
    rid = rubric_id or "essay_default"
    if rid not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rid}' not found")
    return rubrics[rid]


def _format_rubric(rubric):
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)
```

Каждый эндпоинт:
1. Резолвит рубрику через `RubricStoreDep`
2. Строит граф через соответствующую builder-функцию
3. Запускает граф через `ainvoke`
4. Маппит результат в response-схему

Для `/assess/routed` используется `SettingsDep` для доступа к API-ключу и конфигурации — два LLM-экземпляра создаются с разными моделями.

### Шаг 5. Регистрация в `app/api/router.py`

```python
from fastapi import APIRouter

from app.api.v1 import assessment, graph, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(graph.router)
```

### Шаг 6. Тестирование

Установи зависимости и запусти сервер:

```bash
uv pip install -e ".[agents]"
uvicorn app.main:app --reload
```

**Линейный граф (prepare → assess → format):**

```bash
curl -s -X POST http://localhost:8000/api/v1/graph/assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The impact of artificial intelligence on modern education is both profound and multifaceted. AI-powered tutoring systems, as demonstrated by Smith (2023), can provide personalized learning experiences that adapt to individual student needs. This essay examines three key areas where AI transforms education: adaptive learning, automated assessment, and accessibility improvements.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

**Conditional routing (short → haiku, long → sonnet):**

Короткая работа (<200 слов) — маршрутизируется на быструю модель:

```bash
curl -s -X POST http://localhost:8000/api/v1/graph/assess/routed \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "AI is changing education. Computers help students learn. This is a short essay about technology.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

Длинная работа (>200 слов) — маршрутизируется на мощную модель:

```bash
curl -s -X POST http://localhost:8000/api/v1/graph/assess/routed \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The integration of artificial intelligence into educational systems represents a paradigm shift in how knowledge is transmitted and assessed. According to recent meta-analyses by Johnson et al. (2024), AI-powered adaptive learning platforms have demonstrated consistent improvements in student outcomes across diverse educational contexts. This essay examines the transformative potential of AI in three critical areas: personalized learning pathways, automated formative assessment, and equitable access to quality education. The first area, personalized learning, leverages machine learning algorithms to create individualized educational experiences. Unlike traditional one-size-fits-all approaches, AI systems can analyze student performance data in real-time and adjust content difficulty, pacing, and presentation style accordingly (Williams, 2023). The second area concerns automated assessment, where natural language processing enables immediate, detailed feedback on student work. The third area addresses accessibility, where AI translation and text-to-speech tools break down barriers for students with diverse needs.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

Обрати внимание на поля `route` и `model_used` в ответе — они показывают, какой путь был выбран.

**Tool calling (LLM решает, какие tools вызвать):**

```bash
curl -s -X POST http://localhost:8000/api/v1/graph/assess/tools \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "According to Smith (2023) and Johnson et al. (2024), artificial intelligence is transforming education. However, concerns about academic integrity remain valid (Brown, 2023).\n\nThe first major impact is personalized learning. AI tutoring systems adapt to individual student needs.\n\nIn conclusion, while AI presents challenges, its potential to improve educational outcomes is significant.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

В ответе `tool_calls` покажет, какие tools LLM решила вызвать: `count_words` для подсчёта слов, `check_structure` для проверки структуры, `check_citations` для поиска цитирований.

**Reviewed assessment (draft → review):**

```bash
curl -s -X POST http://localhost:8000/api/v1/graph/assess/reviewed \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is the defining crisis of our generation. The Intergovernmental Panel on Climate Change (IPCC, 2023) warns that without immediate action, global temperatures will rise by 2.5 degrees Celsius by 2100. This essay argues that a combination of carbon pricing, renewable energy investment, and reforestation offers the most viable path to mitigation.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

Сравни `draft_assessment` и `final_assessment` — ревьюер может скорректировать баллы, если считает их несправедливыми или непоследовательными.

### Связь с теорией

| Эндпоинт | Теоретический концепт | Что демонстрирует |
|----------|----------------------|-------------------|
| `POST /assess` | StateGraph, nodes, edges (разделы 2, 3) | Базовый линейный граф: define state → add nodes → add edges → compile → invoke |
| `POST /assess/routed` | Conditional edges (раздел 5) | Router-функция анализирует state и выбирает ветку обработки |
| `POST /assess/tools` | Tool calling, ToolNode, reducers (разделы 4, 6) | ReAct-цикл: agent → tools → agent; `messages: Annotated[list, add]` накапливает историю |
| `POST /assess/reviewed` | Multi-node pipeline (раздел 3) | Двойной LLM-вызов с разными промптами; draft-node и review-node обмениваются данными через state |

Граф-код в `assessment_graph.py` показывает четыре паттерна LangGraph от простого к сложному: линейный → условный → циклический → многошаговый. Каждый следующий эндпоинт добавляет один новый концепт.

---

## Чеклист самопроверки

- [ ] В чём разница между LCEL chain и LangGraph? Назови три ограничения LCEL, которые решает LangGraph.
- [ ] Что такое state reducer? Чем `Annotated[list, add]` (append) отличается от обычного `list` (replace)?
- [ ] Как работают conditional edges? Что принимает и возвращает router-функция?
- [ ] Объясни цикл tool calling: agent → (tool_calls?) → tools → agent → ... Когда цикл завершается?
- [ ] Зачем checkpointer? Почему interrupt не работает без него?
- [ ] Как `interrupt_before` реализует human-in-the-loop? Как возобновить граф после паузы?
- [ ] Почему node должен возвращать только изменённые поля, а не весь state?
- [ ] Запусти `/assess/routed` с коротким и длинным эссе — какие модели были выбраны?
- [ ] Запусти `/assess/tools` — какие tools вызвала LLM и почему?

---

## Частые ошибки

### 1. Возврат полного state из node

```python
async def my_node(state):
    return {**state, "analysis": "done"}
```

Это перезаписывает **все** поля, включая те, которые обновлялись другими node'ами. Особенно опасно с reducers — `messages: Annotated[list, add]` при возврате полного state может задвоить все сообщения.

```python
async def my_node(state):
    return {"analysis": "done"}
```

### 2. Забыть checkpointer для interrupt

```python
app = graph.compile(interrupt_before=["publish"])
```

Без checkpointer'а граф не может сохранить state при остановке и возобновить позже. Это вызовет `ValueError` при компиляции.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["publish"],
)
```

### 3. `list` без reducer для накапливающихся данных

```python
class State(TypedDict):
    messages: list
```

Каждый node, записывающий `{"messages": [new_msg]}`, **заменит** весь список на `[new_msg]`. Предыдущие сообщения потеряны.

```python
class State(TypedDict):
    messages: Annotated[list, add]
```

### 4. Отсутствие edge к END

```python
graph.add_node("format", format_node)
graph.add_edge("assess", "format")
```

Узел `format` не имеет выхода. При компиляции LangGraph может не выдать ошибку, но граф зависнет на этом узле.

```python
graph.add_edge("assess", "format")
graph.add_edge("format", END)
```

### 5. Tool без docstring

```python
@tool
def count_words(text: str) -> int:
    return len(text.split())
```

LLM использует docstring для понимания, что делает tool и когда его вызывать. Без описания модель не будет знать, зачем этот tool нужен, и может вызывать его некорректно или игнорировать.

```python
@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())
```

### 6. Sync-функция для LLM-вызова в node

```python
def assess_node(state):
    result = chain.invoke({"student_work": state["student_work"]})
    return {"assessment": result}
```

Sync `invoke` блокирует event loop FastAPI. Используй `async def` и `ainvoke`:

```python
async def assess_node(state):
    result = await chain.ainvoke({"student_work": state["student_work"]})
    return {"assessment": result}
```

---

## Что читать дальше

- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/) — архитектура и концепции
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — первый граф за 10 минут
- [Tool Calling in LangChain](https://python.langchain.com/docs/concepts/tool_calling/) — как LLM взаимодействует с tools
- [Human-in-the-loop Patterns](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — паттерны approve/edit/reject
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/) — persistence и state management
- **[Тема 11: Tool Use / Function Calling](topic_11_tool_use.md)** — глубокое погружение в инструменты: args_schema, StructuredTool, BaseTool, error handling, async tools, InjectedToolArg. Эта тема раскрывает tool calling подробно — то, что здесь дано в виде введения.
- **[Тема 13: Advanced Agentic Patterns](topic_13_agentic_patterns.md)** — ReAct, Reflection, Plan-and-Execute, Map-Reduce — продвинутые паттерны на основе LangGraph.
- **[Тема 14: Multi-Agent Systems](topic_14_multi_agent.md)** — координация нескольких агентов: supervisor, subgraphs, handoff.

**Следующая тема:** [Тема 7: Conversational AI](topic_07_conversational_ai.md) — диалоги с памятью и контекстом.

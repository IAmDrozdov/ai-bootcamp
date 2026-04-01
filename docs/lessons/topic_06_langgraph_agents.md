# Тема 6: LangGraph + Agents

> **Пререквизиты:** темы 1–4 (LCEL chains, structured output), рекомендуется тема 5 (RAG)
> **Зависимости:** `langgraph`, `langchain-anthropic`

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

LangGraph позволяет строить многошаговые пайплайны: анализ → выбор модели → оценка → ревью — где каждый шаг может принимать решения на основе результатов предыдущих.

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

## Практика

Четыре самостоятельных примера, каждый демонстрирует отдельный концепт LangGraph:

| Пример | Что делает | Концепт |
|--------|------------|---------|
| 1. Линейный граф | prepare → assess → format | StateGraph, nodes, edges |
| 2. Conditional routing | Выбор ветки по длине текста | Conditional edges |
| 3. Tool calling | LLM вызывает tools в цикле | ToolNode, ReAct-цикл, reducers |
| 4. Human-in-the-loop | Пауза для ревью перед публикацией | interrupt_before, checkpointer |

### Пример 1. Линейный граф: prepare → assess → format

Базовый StateGraph с тремя узлами и линейными рёбрами. Каждый узел читает state и возвращает обновления.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AssessmentState(TypedDict):
    student_work: str
    analysis: str
    score: int
    result: str


def prepare(state: AssessmentState) -> dict:
    work = state["student_work"]
    word_count = len(work.split())
    paragraphs = len([p for p in work.split("\n\n") if p.strip()])
    return {"analysis": f"Words: {word_count}, Paragraphs: {paragraphs}"}


def assess(state: AssessmentState) -> dict:
    word_count = int(state["analysis"].split("Words: ")[1].split(",")[0])
    if word_count > 100:
        return {"score": 80}
    return {"score": 50}


def format_result(state: AssessmentState) -> dict:
    return {"result": f"Score: {state['score']}/100 | {state['analysis']}"}


graph = StateGraph(AssessmentState)
graph.add_node("prepare", prepare)
graph.add_node("assess", assess)
graph.add_node("format_result", format_result)

graph.add_edge(START, "prepare")
graph.add_edge("prepare", "assess")
graph.add_edge("assess", "format_result")
graph.add_edge("format_result", END)

app = graph.compile()

result = app.invoke({
    "student_work": "The impact of artificial intelligence on modern education is profound. "
    "AI-powered tutoring systems can provide personalized learning experiences "
    "that adapt to individual student needs.\n\n"
    "This essay examines three key areas where AI transforms education: "
    "adaptive learning, automated assessment, and accessibility improvements.",
    "analysis": "",
    "score": 0,
    "result": "",
})

print(result["analysis"])
print(result["result"])
```

Каждый node возвращает только изменённые поля — LangGraph мержит их с текущим state.

### Пример 2. Conditional routing по длине текста

Router-функция анализирует state и выбирает следующий узел. Короткий текст идёт на быструю оценку, длинный — на полную.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class RoutedState(TypedDict):
    student_work: str
    word_count: int
    route: str
    assessment: str


def analyze(state: RoutedState) -> dict:
    return {"word_count": len(state["student_work"].split())}


def route_by_length(state: RoutedState) -> str:
    if state["word_count"] < 50:
        return "quick_assess"
    return "full_assess"


def quick_assess(state: RoutedState) -> dict:
    return {
        "assessment": f"Quick review ({state['word_count']} words): basic feedback",
        "route": "quick",
    }


def full_assess(state: RoutedState) -> dict:
    return {
        "assessment": f"Full review ({state['word_count']} words): detailed feedback",
        "route": "full",
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

app = graph.compile()

short_result = app.invoke({
    "student_work": "AI is changing education.",
    "word_count": 0,
    "route": "",
    "assessment": "",
})
print(f"Short → route: {short_result['route']}, {short_result['assessment']}")

long_text = " ".join(["Education and AI are transforming the modern world."] * 15)
long_result = app.invoke({
    "student_work": long_text,
    "word_count": 0,
    "route": "",
    "assessment": "",
})
print(f"Long  → route: {long_result['route']}, {long_result['assessment']}")
```

Router-функция — чистая функция, не меняющая state. Она только возвращает строку-ключ для выбора следующего узла.

### Пример 3. Tool calling — ReAct-цикл с ToolNode

LLM сама решает, какие tools вызвать. Цикл `agent → tools → agent` продолжается, пока LLM не перестанет запрашивать tools.

```python
import re
from typing import Annotated, TypedDict
from operator import add

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


@tool
def count_words(text: str) -> dict:
    """Count words, sentences, and paragraphs in the given text."""
    words = len(text.split())
    sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    return {"words": words, "sentences": max(sentences, 1), "paragraphs": max(paragraphs, 1)}


@tool
def check_citations(text: str) -> dict:
    """Find academic citations in APA format, e.g. (Smith, 2023)."""
    pattern = r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)"
    citations = re.findall(pattern, text)
    return {"count": len(citations), "citations": citations}


class ToolsState(TypedDict):
    student_work: str
    messages: Annotated[list, add]
    summary: str


tools = [count_words, check_citations]
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def agent(state: ToolsState) -> dict:
    if not state["messages"]:
        msg = HumanMessage(
            content=f"Analyze this text using available tools:\n\n{state['student_work']}"
        )
        response = llm_with_tools.invoke([msg])
        return {"messages": [msg, response]}
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: ToolsState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "summarize"


def summarize(state: ToolsState) -> dict:
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(f"{msg.name}: {msg.content}")
    return {"summary": "\n".join(tool_results)}


graph = StateGraph(ToolsState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "summarize": "summarize"},
)
graph.add_edge("tools", "agent")
graph.add_edge("summarize", END)

app = graph.compile()

result = app.invoke({
    "student_work": (
        "According to Smith (2023), artificial intelligence is transforming education. "
        "However, concerns remain valid (Brown, 2023).\n\n"
        "The first major impact is personalized learning.\n\n"
        "In conclusion, AI's potential to improve outcomes is significant."
    ),
    "messages": [],
    "summary": "",
})

print("Tool results:")
print(result["summary"])
print(f"\nTotal messages in conversation: {len(result['messages'])}")
```

`messages: Annotated[list, add]` — reducer, который конкатенирует списки вместо замены. Без него каждый node перезаписывал бы всю историю сообщений.

### Пример 4. Human-in-the-loop с interrupt_before

Граф останавливается перед узлом `publish`, давая возможность проверить результат. Для возобновления нужен checkpointer.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class ReviewState(TypedDict):
    text: str
    draft: str
    published: str


def create_draft(state: ReviewState) -> dict:
    return {"draft": f"Draft assessment of: {state['text'][:50]}..."}


def publish(state: ReviewState) -> dict:
    return {"published": f"PUBLISHED: {state['draft']}"}


graph = StateGraph(ReviewState)
graph.add_node("create_draft", create_draft)
graph.add_node("publish", publish)

graph.add_edge(START, "create_draft")
graph.add_edge("create_draft", "publish")
graph.add_edge("publish", END)

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["publish"],
)

config = {"configurable": {"thread_id": "review-1"}}

result = app.invoke(
    {"text": "Student essay about climate change...", "draft": "", "published": ""},
    config,
)

snapshot = app.get_state(config)
print(f"Draft: {snapshot.values['draft']}")
print(f"Next node: {snapshot.next}")
print(f"Published: {snapshot.values.get('published', '(not yet)')}")

final = app.invoke(None, config)
print(f"\nAfter resume — published: {final['published']}")
```

`interrupt_before=["publish"]` останавливает граф перед выполнением `publish`. Между паузой и возобновлением можно проверить или изменить state:

```python
app.update_state(config, {"draft": "Corrected draft by reviewer"})
final = app.invoke(None, config)
print(final["published"])
```

### Связь с теорией

| Пример | Теоретический концепт | Что демонстрирует |
|--------|----------------------|-------------------|
| 1. Линейный граф | StateGraph, nodes, edges (разделы 2, 3) | Базовый граф: define state → add nodes → add edges → compile → invoke |
| 2. Conditional routing | Conditional edges (раздел 5) | Router-функция анализирует state и выбирает ветку обработки |
| 3. Tool calling | Tool calling, ToolNode, reducers (разделы 4, 6) | ReAct-цикл: agent → tools → agent; `messages: Annotated[list, add]` накапливает историю |
| 4. Human-in-the-loop | interrupt_before, checkpointer (разделы 7, 8) | Пауза перед узлом, проверка state, возобновление выполнения |

Четыре паттерна LangGraph от простого к сложному: линейный → условный → циклический → с паузами. Каждый следующий пример добавляет один новый концепт.

---

## Чеклист самопроверки

- [ ] В чём разница между LCEL chain и LangGraph? Назови три ограничения LCEL, которые решает LangGraph.
- [ ] Что такое state reducer? Чем `Annotated[list, add]` (append) отличается от обычного `list` (replace)?
- [ ] Как работают conditional edges? Что принимает и возвращает router-функция?
- [ ] Объясни цикл tool calling: agent → (tool_calls?) → tools → agent → ... Когда цикл завершается?
- [ ] Зачем checkpointer? Почему interrupt не работает без него?
- [ ] Как `interrupt_before` реализует human-in-the-loop? Как возобновить граф после паузы?
- [ ] Почему node должен возвращать только изменённые поля, а не весь state?
- [ ] Запусти пример 2 с коротким и длинным текстом — какой route был выбран в каждом случае?
- [ ] Запусти пример 3 — какие tools вызвала LLM и почему?

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

### 6. Sync-функция для LLM-вызова в async-контексте

```python
def assess_node(state):
    result = chain.invoke({"student_work": state["student_work"]})
    return {"assessment": result}
```

Sync `invoke` блокирует event loop. Если граф запускается через `ainvoke` или `astream`, используй `async def` и `ainvoke` внутри node'ов:

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

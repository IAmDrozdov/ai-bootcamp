# Тема 6: LangGraph + Agents

> **Пререквизиты:** [Тема 1-4](topic_01_prompt_engineering.md), рекомендуется [Тема 5 (RAG)](topic_05_rag.md)
> **Где в проекте:** `app/graph/`
> **Зависимости:** `langgraph` (группа `agents`)

---

## Теория

### 1. От цепочек к графам

LCEL chains (Тема 2) — это **линейные** пайплайны: A → B → C. Они не поддерживают:
- **Циклы** — повтор шага до выполнения условия
- **Условные переходы** — разные ветки в зависимости от результата
- **Состояние** — общие данные, доступные всем шагам
- **Паузы** — остановка для ввода человека

LangGraph решает эти ограничения через **графы с типизированным состоянием**.

### 2. StateGraph — основная абстракция

StateGraph — это направленный граф, где:
- **State** — типизированный словарь, доступный всем узлам
- **Nodes** — функции, которые читают и обновляют state
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

### 3. Nodes — функции-обработчики

Node — это обычная функция, которая принимает state и возвращает **обновление** state:

```python
async def analyze_node(state: AssessmentState) -> dict:
    work = state["student_work"]
    word_count = len(work.split())
    return {"analysis": f"Word count: {word_count}, paragraphs: {work.count(chr(10))+1}"}
```

Важно: node возвращает **только изменённые поля**. LangGraph мержит их с текущим state. Не нужно возвращать весь state.

### 4. State reducers

По умолчанию: новое значение **заменяет** старое. Но можно настроить **append** для списков:

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]  # append, не replace
    final_answer: str               # replace (default)
```

С `Annotated[list, add]`: если node вернёт `{"messages": [new_msg]}`, новое сообщение **добавится** к списку, а не заменит его.

### 5. Conditional edges — ветвление

Условный переход выбирает следующий node на основе текущего state:

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

Router-функция возвращает строку — имя следующего node.

### 6. Tool calling — LLM решает, что вызвать

Инструменты (tools) — это функции с описанием, доступные LLM. Модель **сама решает**, когда и какой tool вызвать:

```python
from langchain_core.tools import tool

@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())

@tool
def check_citations(text: str) -> dict:
    """Check if the text contains academic citations."""
    import re
    citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text)
    return {"count": len(citations), "citations": citations}
```

LLM видит описание tool (name + docstring + параметры) и решает:
1. Нужно ли вызвать tool? (может ответить без него)
2. Какой tool вызвать?
3. С какими аргументами?

### 7. Human-in-the-loop

`interrupt_before` останавливает граф **перед** указанным node, ожидая подтверждения человека:

```python
graph = StateGraph(State)
# ... add nodes ...
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["publish_assessment"],
)
```

Поток: граф выполняется до "publish_assessment" → останавливается → человек проверяет → подтверждает → граф продолжает.

### 8. Checkpointing — сохранение состояния

Checkpointer сохраняет state после каждого node, позволяя:
- Восстановиться после ошибки
- Реализовать human-in-the-loop (пауза → возобновление)
- "Перемотать" граф к предыдущему шагу

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # in-memory, для разработки
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "assessment-123"}}
result = await app.ainvoke(state, config)
```

---

## Практические задания

### Задание 1: Простой граф

**Цель:** переписать assessment chain как StateGraph с 3 узлами.

**Файлы:** `app/graph/assessment_graph.py`, `experiments/t6_simple_graph.py`

**Критерии успеха:**
- Граф: prepare → assess → format_result
- State содержит: student_work, rubric, analysis, assessment_result
- Результат эквивалентен LCEL chain

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a LangGraph version of the assessment chain.

1. Create app/graph/assessment_graph.py:
   - Define AssessmentState(TypedDict) with fields: student_work, rubric, analysis, scores, result
   - Node "prepare": analyze the student work (word count, paragraph count, structure summary)
   - Node "assess": call LLM with assessment prompt, parse into scores
   - Node "format_result": compile final AssessmentResponse from scores + analysis
   - Edges: START → prepare → assess → format_result → END
   - Function build_assessment_graph(llm) -> CompiledGraph

2. Create experiments/t6_simple_graph.py:
   - Build the graph
   - Run it on the sample essay
   - Print state after each node (use astream with stream_mode="updates")
   - Compare output with the LCEL chain result

Install: uv pip install -e ".[agents]"
Run with: python -m experiments.t6_simple_graph
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/graph/assessment_graph.py:

1. STATE: Is AssessmentState properly typed with TypedDict?
2. NODES: Are there 3 distinct nodes with clear responsibilities?
3. EDGES: Is the graph structure correct (START → prepare → assess → format → END)?
4. LLM CALL: Is the LLM called in the assess node with proper prompt?
5. COMPARISON: Does the experiment compare output with LCEL chain?

Common mistakes:
- Returning full state from nodes instead of just changes
- Not using async (ainvoke) for LLM calls in nodes
- State fields not matching between nodes
- Missing START/END edges

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Conditional routing

**Цель:** добавить ветвление: короткие работы → быстрая оценка, длинные → полная.

**Файлы:** модификация `app/graph/assessment_graph.py`, `experiments/t6_routing.py`

**Критерии успеха:**
- Node `analyze_complexity` определяет сложность работы
- < 200 слов → быстрый путь (haiku), >= 200 → полный путь (sonnet)
- Оба пути производят корректный AssessmentResponse

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Add conditional routing to the assessment graph.

1. Modify app/graph/assessment_graph.py:
   - Add "analyze_complexity" node that counts words and determines complexity
   - Add "quick_assess" node using claude-haiku for fast assessment
   - Add "full_assess" node using claude-sonnet for thorough assessment
   - Add conditional edge from "analyze_complexity":
     if word_count < 200 → "quick_assess"
     else → "full_assess"
   - Both assess paths lead to "format_result" → END
   - Update AssessmentState with "complexity" and "model_used" fields

2. Create experiments/t6_routing.py:
   - Test with a short essay (<200 words) — should route to quick_assess
   - Test with a long essay (>500 words) — should route to full_assess
   - Print: which path was taken, which model was used, scores, timing
   - Compare quality difference between haiku and sonnet paths

Run with: python -m experiments.t6_routing
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the conditional routing implementation:

1. ROUTING LOGIC: Is complexity analysis correct (word count threshold)?
2. CONDITIONAL EDGE: Is add_conditional_edges used properly?
3. TWO MODELS: Are haiku and sonnet actually used for different paths?
4. CONVERGENCE: Do both paths lead to the same format_result node?
5. EXPERIMENT: Does it test both paths with appropriate inputs?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: Tool calling

**Цель:** дать LLM возможность вызывать инструменты для анализа работы.

**Файлы:** `app/graph/tools.py`, `experiments/t6_tools.py`

**Критерии успеха:**
- 2+ tools: count_words, check_structure
- LLM сам решает, нужно ли вызвать tool
- Tool results используются в оценке

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create tools for the assessment agent.

1. Create app/graph/tools.py:
   - @tool count_words(text: str) -> dict: count words, sentences, paragraphs
   - @tool check_structure(text: str) -> dict: check for introduction, conclusion, topic sentences, transitions
   - @tool check_citations(text: str) -> dict: find academic citations using regex patterns

2. Create a tool-calling agent node in app/graph/assessment_graph.py:
   - Bind tools to the LLM: llm.bind_tools([count_words, check_structure, check_citations])
   - Create a "tool_agent" node that:
     a. Calls LLM with tools bound
     b. If LLM returns tool_calls: execute the tools, add results to state
     c. If LLM returns text: move to assessment
   - Use a loop: tool_agent → (has tool calls?) → execute_tools → tool_agent
                                                → (no tool calls?) → assess

3. Create experiments/t6_tools.py:
   - Run the agent on different essays
   - Print: which tools were called, tool results, final assessment
   - Show that LLM chooses tools based on context

Run with: python -m experiments.t6_tools
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/graph/tools.py and the tool calling implementation:

1. TOOL DEFINITIONS: Are tools properly decorated with @tool and have clear docstrings?
2. TOOL BINDING: Is llm.bind_tools() used correctly?
3. AGENT LOOP: Is there a cycle (tool_agent → execute → tool_agent) until no more tool calls?
4. TOOL EXECUTION: Are tool calls actually executed and results fed back?
5. LLM AUTONOMY: Does the LLM decide which tools to call (not hardcoded)?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: Human-in-the-loop

**Цель:** добавить точку утверждения перед публикацией оценки.

**Файлы:** `experiments/t6_human_loop.py`

**Критерии успеха:**
- Граф останавливается перед финальным node
- Показывает черновик оценки
- Человек может подтвердить или запросить пересмотр
- После подтверждения — граф завершается

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Add human-in-the-loop to the assessment graph.

1. Create experiments/t6_human_loop.py:
   - Build assessment graph with interrupt_before=["publish"]
   - Add a "publish" node that marks assessment as final
   - Use MemorySaver as checkpointer

2. Flow:
   - Run graph until interrupt: result = await app.ainvoke(input, config)
   - Print the draft assessment from state
   - Ask user to confirm (input("Approve? [y/n]: "))
   - If approved: resume graph with await app.ainvoke(None, config)
   - If rejected: modify state and re-run from assessment node

3. Demonstrate:
   - First run: graph stops at "publish", shows draft
   - User approves: graph completes
   - Second run: user rejects, graph re-assesses

Config must include thread_id for checkpointing:
config = {"configurable": {"thread_id": "demo-1"}}

Run with: python -m experiments.t6_human_loop
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t6_human_loop.py:

1. INTERRUPT: Is interrupt_before used on the correct node?
2. CHECKPOINTER: Is MemorySaver configured?
3. RESUME: Can the graph be resumed after interrupt?
4. THREAD_ID: Is thread_id in config for state persistence?
5. USER INTERACTION: Is there a clear approve/reject flow?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 5: Полный Assessment Agent

**Цель:** собрать полный агентский граф, объединяющий все компоненты.

**Файлы:** `app/graph/assessment_agent.py`, `experiments/t6_full_agent.py`

**Критерии успеха:**
- Граф: analyze → retrieve_rubric → assess → review → (human confirms)
- Использует tools, conditional routing, human-in-the-loop
- Checkpointing для сохранения состояния

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Build the complete assessment agent graph.

Create app/graph/assessment_agent.py:

State: student_work, rubric_id, rubric, analysis, tool_results, assessment, review, approved

Nodes:
1. "analyze" — run tools (count_words, check_structure) on student work
2. "select_model" — conditional: short → haiku, long → sonnet
3. "retrieve_rubric" — load rubric from store by rubric_id
4. "assess" — run LLM assessment with rubric + analysis context
5. "review" — second LLM pass: review the assessment for consistency and fairness
6. "publish" — mark assessment as final (interrupt_before for human approval)

Edges:
START → analyze → select_model → retrieve_rubric → assess → review → publish → END
select_model: conditional based on complexity

Compile with MemorySaver and interrupt_before=["publish"].

Create experiments/t6_full_agent.py:
- Run the full agent pipeline
- Print state after each node
- Demonstrate human approval flow

Run with: python -m experiments.t6_full_agent
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/graph/assessment_agent.py:

1. COMPLETENESS: Are all 6 nodes implemented?
2. STATE: Does the TypedDict cover all necessary fields?
3. ROUTING: Is model selection based on analysis results?
4. REVIEW: Does the review node actually check the assessment?
5. HUMAN-IN-THE-LOOP: Is interrupt_before configured on "publish"?
6. CHECKPOINTING: Is MemorySaver used?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] В чём разница между LCEL chain и LangGraph? Когда переходить на граф?
- [ ] Что такое state reducer? Чем `replace` отличается от `add`?
- [ ] Как работают conditional edges? Что возвращает router-функция?
- [ ] Объясни цикл tool calling: LLM → tool call → execute → LLM → ...
- [ ] Зачем checkpointer? Что он сохраняет и когда?
- [ ] Как interrupt_before реализует human-in-the-loop?

---

## Частые ошибки

### 1. Возврат полного state из node

```python
# Плохо: перезаписывает ВСЕ поля
async def my_node(state):
    return {**state, "analysis": "done"}

# Хорошо: возвращай только изменения
async def my_node(state):
    return {"analysis": "done"}
```

### 2. Забыть checkpointer для interrupt

```python
# Не работает: interrupt без checkpointer
app = graph.compile(interrupt_before=["publish"])  # ValueError

# Правильно
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["publish"])
```

### 3. Неправильный тип state для reducers

```python
# Плохо: messages перезаписывается при каждом node
class State(TypedDict):
    messages: list  # replace behavior

# Хорошо: messages дополняется
class State(TypedDict):
    messages: Annotated[list, add]  # append behavior
```

---

## Что читать дальше

- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/) — архитектура
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — первый граф
- [Tool Calling](https://python.langchain.com/docs/concepts/tool_calling/) — как LLM вызывает tools
- [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — паттерны

**Следующая тема:** [Тема 7: Conversational AI](topic_07_conversational_ai.md) — как вести диалог с памятью.

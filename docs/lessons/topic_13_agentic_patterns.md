# Тема 13: Advanced Agentic Patterns — паттерны проектирования агентов

> **Пререквизиты:** [Тема 6 (LangGraph)](topic_06_langgraph_agents.md), [Тема 11 (Tool Use)](topic_11_tool_use.md)
> **Зависимости:** `langgraph`, `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. Зачем паттерны — от ad-hoc графов к архитектурным решениям

В теме 6 мы научились строить LangGraph-графы: объявлять state, добавлять nodes и edges, компилировать и вызывать. Но каждый граф строился с нуля — под конкретный use-case, с уникальной структурой. Когда в проекте 3–5 графов, это ещё управляемо. Когда их 15–20, начинаются проблемы:

- **Дублирование логики** — в каждом графе заново реализуется цикл «вызвать LLM → проверить tool calls → выполнить tools → вернуть результат»
- **Нет общего языка** — при обсуждении архитектуры приходится описывать каждый граф «от узла к узлу», вместо того чтобы сказать «здесь мы используем Plan-and-Execute»
- **Сложно масштабировать** — непонятно, как добавить параллелизм или самопроверку к существующему графу

Паттерны проектирования агентов решают эти проблемы так же, как GoF-паттерны решают их в OOP. Каждый паттерн — это **проверенная архитектурная схема** для типового класса задач. Вместо того чтобы каждый раз придумывать структуру графа заново, ты выбираешь паттерн, подходящий под задачу, и адаптируешь его.

Четыре основных паттерна, которые мы разберём:

| Паттерн | Суть | Типовая задача |
|---------|------|----------------|
| **ReAct** | Reasoning + Acting в цикле | Агент с tools, принимающий решения |
| **Reflection** | Generate → Critique → Revise | Улучшение качества через самопроверку |
| **Plan-and-Execute** | Составить план → выполнить по шагам | Сложные многошаговые задачи |
| **Map-Reduce** | Параллельная обработка → агрегация | Независимые подзадачи |

Каждый паттерн можно реализовать на LangGraph. Более того, LangGraph предоставляет примитивы, специально спроектированные для этих паттернов: `Send()` для fan-out, `Command` для динамической маршрутизации, `create_react_agent` для быстрого старта.

Важный принцип: **паттерн — это не цель, а инструмент**. Не нужно использовать Plan-and-Execute для задачи, которая решается простым LCEL chain. Правило: начинай с простейшего подхода и добавляй сложность, когда она оправдана результатом.

### 2. ReAct (Reasoning + Acting)

ReAct — самый распространённый паттерн для агентов с tools. Название — акроним от **Re**asoning + **Act**ing. Формальный цикл выглядит так:

```
Thought  → "Мне нужно узнать количество слов в эссе"
Action   → call count_words(text=...)
Observation → 342
Thought  → "Эссе короткое. Проверю глубину аргументации"
Action   → call analyze_arguments(text=...)
Observation → {"strong": 1, "weak": 2, "unsupported": 1}
Thought  → "Аргументация слабая. У меня достаточно данных для оценки"
Answer   → {score: 62, feedback: "..."}
```

**Отличие от простого tool calling.** В теме 11 мы реализовали tool calling — LLM получает список tools и вызывает нужные. Но простой tool calling — это однократный цикл: LLM решает, какие tools вызвать, вызывает их все за один проход, получает результаты и формирует ответ. ReAct добавляет **явное рассуждение** (Thought) перед каждым действием и **многошаговость** — после получения Observation агент может решить вызвать ещё один tool, а не сразу дать ответ.

**LangGraph реализация** ReAct — это граф из трёх элементов:

```
agent_node → should_continue (conditional edge) → tool_node → agent_node
                                                → END
```

- `agent_node` — вызывает LLM с текущими messages (включая results предыдущих tool calls)
- `should_continue` — проверяет, есть ли tool_calls в ответе LLM; если да — переход к tool_node, если нет — к END
- `tool_node` — выполняет tool calls и добавляет результаты в messages

LangGraph предоставляет готовую реализацию через `create_react_agent` из `langgraph.prebuilt`:

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())

@tool
def analyze_structure(text: str) -> dict:
    """Analyze the structural elements of the text."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return {
        "paragraph_count": len(paragraphs),
        "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1),
        "has_introduction": len(paragraphs) > 0 and len(paragraphs[0].split()) > 20,
        "has_conclusion": len(paragraphs) > 0 and len(paragraphs[-1].split()) > 20,
    }

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
tools = [count_words, analyze_structure]

agent = create_react_agent(llm, tools)
```

Под капотом `create_react_agent` строит именно тот граф, который описан выше. Но ты можешь кастомизировать его: добавить system prompt через `state_modifier`, ограничить количество итераций, подключить checkpointer.

**Применение к assessment-домену.** Представим, что агенту нужно оценить студенческую работу, но перед выставлением оценки он должен собрать аналитику: подсчитать слова, проанализировать структуру, проверить наличие цитат. ReAct позволяет агенту самому решить, какие tools вызвать и в каком порядке, основываясь на содержании работы.

**Ограничения ReAct:**

- **Зацикливание** — агент может бесконечно вызывать tools, не приближаясь к ответу. Особенно часто это происходит, если tools возвращают неоднозначные результаты.
- **Нет стратегического планирования** — агент действует реактивно (Thought → Action), не строя план наперёд. Для сложных многошаговых задач это неэффективно.
- **Стоимость** — каждая итерация цикла = вызов LLM. При 5-7 итерациях стоимость одного запроса вырастает в 5-7 раз.

**Max iterations — защита от зацикливания.** Всегда ограничивай количество циклов. В `create_react_agent` это делается через `recursion_limit` при вызове:

```python
result = await agent.ainvoke(
    {"messages": [("human", "Assess this essay: ...")]},
    config={"recursion_limit": 25},
)
```

`recursion_limit` — это максимальное количество **шагов графа** (не итераций цикла). Один цикл ReAct = 2 шага (agent_node + tool_node), поэтому `recursion_limit=25` ≈ 12 итераций. Значение по умолчанию — 25.

### 3. Reflection / Self-correction

Паттерн Reflection основан на идее: **один LLM генерирует результат, другой (или тот же) проверяет и даёт обратную связь, после чего первый улучшает результат**. Это цикл критического мышления, встроенный в граф.

Формально:

```
Generate → Reflect → (quality OK?) → YES → Done
                                    → NO  → Generate (с feedback)
```

**Два подхода к reflection:**

| Подход | Generator | Critic | Плюсы | Минусы |
|--------|-----------|--------|-------|--------|
| **Self-reflection** | LLM A | LLM A (другой prompt) | Дёшево, один вызов на итерацию | Модель может не замечать свои ошибки |
| **Cross-reflection** | LLM A | LLM B | Разные модели ловят разные ошибки | Дороже, нужно координировать |

Self-reflection — более распространённый подход. Одна и та же модель используется дважды: сначала с промптом генератора, потом с промптом критика. Ключевое условие — промпт критика должен содержать **чёткие критерии проверки**: рубрику, чек-лист, конкретные вопросы.

**LangGraph реализация:**

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class ReflectionState(TypedDict):
    student_work: str
    rubric: str
    draft: str
    feedback: str
    revision_count: int
    is_satisfactory: bool


async def generate_node(state: ReflectionState) -> dict:
    if state.get("feedback"):
        prompt = f"""Previous assessment:
{state['draft']}

Feedback on the assessment:
{state['feedback']}

Revise the assessment addressing the feedback.
Student work: {state['student_work']}
Rubric: {state['rubric']}"""
    else:
        prompt = f"""Assess the following student work.
Student work: {state['student_work']}
Rubric: {state['rubric']}"""

    response = await llm.ainvoke([("human", prompt)])
    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
    }


async def reflect_node(state: ReflectionState) -> dict:
    prompt = f"""Review this assessment for quality:

Assessment: {state['draft']}
Original work: {state['student_work']}
Rubric: {state['rubric']}

Check:
1. Does the assessment address ALL rubric criteria?
2. Are scores justified with specific evidence?
3. Is the feedback actionable?
4. Are scores consistent with the feedback text?

If satisfactory, respond: SATISFACTORY
Otherwise, provide specific feedback for improvement."""

    response = await llm.ainvoke([("human", prompt)])
    is_ok = "SATISFACTORY" in response.content.upper()
    return {
        "feedback": response.content,
        "is_satisfactory": is_ok,
    }


def should_continue(state: ReflectionState) -> str:
    if state["is_satisfactory"]:
        return "done"
    if state["revision_count"] >= 3:
        return "done"
    return "revise"


graph = StateGraph(ReflectionState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)

graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {
    "revise": "generate",
    "done": END,
})
```

**Критерии остановки** — важнейший аспект Reflection. Без них граф будет бесконечно улучшать результат (или думать, что улучшает):

1. **Max iterations** — жёсткий лимит на количество ревизий (обычно 2–4)
2. **Quality threshold** — критик подтверждает, что результат удовлетворителен
3. **No changes** — новая версия не отличается от предыдущей (модель не может улучшить)

На практике 2–3 итерации Reflection дают заметное улучшение качества. Дальнейшие итерации обычно дают marginal improvement при значительном увеличении стоимости и latency.

**Когда использовать Reflection:**

- Задачи, где качество критично (итоговые оценки, официальные отзывы)
- Есть чёткие, проверяемые критерии (рубрика, чек-лист)
- Допустимо увеличение latency в 2–4 раза
- Одной генерации недостаточно — модель часто пропускает критерии или даёт неконсистентные оценки

### 4. Plan-and-Execute

Plan-and-Execute решает проблему, с которой ReAct справляется плохо: **сложные задачи, где порядок действий неочевиден и нужно стратегическое планирование**.

Идея: вместо того чтобы действовать реактивно (Thought → Action → Observation), агент сначала **строит полный план**, а потом **выполняет его по шагам**. После каждого шага агент может **обновить план** с учётом новой информации.

Три роли в паттерне:

| Роль | Что делает | Prompt/Logic |
|------|-----------|--------------|
| **Planner** | Генерирует список шагов | LLM со structured output → `list[str]` |
| **Executor** | Выполняет один шаг | LLM (возможно с tools) |
| **Replanner** | Обновляет оставшийся план | LLM, видит результаты выполненных шагов |

**LangGraph реализация:**

```
plan → execute_step → replan → execute_step → ... → done
```

State содержит текущий план, индекс текущего шага и накопленные результаты:

```python
class PlanExecuteState(TypedDict):
    objective: str
    plan: list[str]
    current_step_index: int
    step_results: Annotated[list[dict], operator.add]
    final_result: str
```

Planner генерирует план как structured output — список строк, каждая описывает один шаг. Это важно: план должен быть **конкретным и выполнимым**, а не абстрактным. «Проанализировать эссе» — плохой шаг. «Подсчитать количество слов и абзацев» — хороший.

Replanner — ключевое отличие от статического плана. После выполнения каждого шага Replanner видит:
- Исходную задачу
- Текущий план
- Результаты выполненных шагов

И может: добавить шаги, удалить ненужные, изменить порядок, или решить, что план выполнен.

**Пример в assessment-домене:** «Оцени портфолио из 5 работ студента и дай итоговую рекомендацию».

Planner генерирует:
1. Загрузить и проанализировать работу 1 (эссе)
2. Загрузить и проанализировать работу 2 (код)
3. Загрузить и проанализировать работу 3 (презентация)
4. Загрузить и проанализировать работу 4 (отчёт)
5. Загрузить и проанализировать работу 5 (проект)
6. Сравнить результаты, найти паттерны развития
7. Сформировать итоговую рекомендацию

После шага 3 Replanner может решить: «Работы 1–3 показывают сильную аналитику, но слабое структурирование. Добавлю шаг: проверить структуру работ 4–5 отдельно».

**Когда использовать Plan-and-Execute:**

- Задача состоит из 5+ логических шагов
- Порядок действий зависит от промежуточных результатов
- Нужна адаптация к неожиданным данным
- Задача слишком сложна для однократной генерации

**Когда НЕ использовать:**

- Простые задачи (1–3 шага) — оверхед планирования не оправдан
- Все шаги известны заранее и не зависят от результатов — используй обычный граф
- Реальное время критично — планирование добавляет latency

### 5. Map-Reduce

Map-Reduce — паттерн для задач с **независимыми подзадачами**, которые можно обрабатывать параллельно. Название заимствовано из одноимённой парадигмы распределённых вычислений.

Две фазы:

| Фаза | Что происходит | LangGraph-механизм |
|------|---------------|-------------------|
| **Map** | Каждый элемент обрабатывается независимо | `Send()` API — fan-out |
| **Reduce** | Результаты объединяются в итоговый | Обычный node с reducer |

**Send() API** — ключевой примитив LangGraph для Map-Reduce. `Send()` позволяет динамически создавать параллельные ветки графа:

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END


def fan_out(state: MapReduceState) -> list[Send]:
    return [
        Send("process_criterion", {"criterion": c, "student_work": state["student_work"]})
        for c in state["criteria"]
    ]


graph = StateGraph(MapReduceState)
graph.add_node("process_criterion", process_criterion_node)
graph.add_node("aggregate", aggregate_node)

graph.add_conditional_edges(START, fan_out)
graph.add_edge("process_criterion", "aggregate")
graph.add_edge("aggregate", END)
```

Каждый `Send()` создаёт отдельную «копию» графа, которая обрабатывает свой элемент. Все копии работают параллельно. Когда все завершены, результаты собираются в reducer и передаются в следующий узел.

**Пример в assessment-домене:** оценка работы по 5 критериям параллельно.

Вместо того чтобы один LLM-вызов оценивал все 5 критериев (что приводит к потере фокуса на каждом), мы создаём 5 параллельных вызовов, каждый оценивает один критерий. Потом агрегируем результаты.

Преимущества:
- **Скорость** — 5 параллельных вызовов завершаются за время одного (если API не ограничивает)
- **Качество** — каждый вызов фокусируется на одном критерии, без потери контекста
- **Масштабируемость** — добавление новых критериев не увеличивает latency

Ограничения:
- **Стоимость** — 5 параллельных вызовов стоят в 5 раз больше одного
- **Нет перекрёстных ссылок** — оценка по критерию A не может учитывать оценку по критерию B (они независимы)
- **Rate limits** — параллельные вызовы могут упираться в лимиты API

**Когда использовать Map-Reduce:**

- Задача естественно разбивается на независимые подзадачи
- Подзадачи не зависят друг от друга
- Latency важнее стоимости
- Качество каждой подзадачи улучшается при изоляции (фокусированный prompt)

### 6. Комбинирование паттернов

Четыре базовых паттерна — это кирпичики. Реальные системы часто комбинируют несколько паттернов:

**ReAct + Reflection.** Агент с tools (ReAct) выполняет задачу, затем результат проверяется Reflection-циклом. Пример: ReAct-агент оценивает работу, используя tools для анализа → Reflection проверяет, что оценка консистентна и обоснована.

```
ReAct loop (gather data + assess) → Reflection loop (verify + improve)
```

**Plan-Execute + Map-Reduce.** Planner разбивает задачу на шаги. Executor обнаруживает, что некоторые шаги независимы, и запускает их параллельно через Map-Reduce.

```
Plan → [Step 1] → Map([Step 2a, 2b, 2c]) → Reduce → [Step 3] → Done
```

**Пример комбинации в assessment-проекте:** «Оцени 3 работы студента, каждую по 5 критериям, с проверкой качества».

1. **Plan** — составить план: загрузить работы, оценить каждую, сравнить, дать итог
2. **Map-Reduce (внешний)** — оценить 3 работы параллельно
3. **Map-Reduce (внутренний)** — для каждой работы оценить 5 критериев параллельно
4. **Reflection** — проверить итоговую оценку на консистентность
5. **Reduce** — агрегировать всё в финальный отчёт

Правило: **добавляй паттерн только когда простой подход не работает**. Комбинирование увеличивает сложность, стоимость и latency. Каждый дополнительный паттерн должен быть оправдан конкретным улучшением результата.

Практическая рекомендация — инкрементальный подход:

1. Начни с простого LCEL chain
2. Если нужны tools → добавь ReAct
3. Если качество нестабильно → оберни в Reflection
4. Если задача слишком сложна для одного шага → добавь Plan-Execute
5. Если есть независимые подзадачи → используй Map-Reduce

### 7. State design для каждого паттерна

Правильный дизайн state — основа надёжного графа. Каждый паттерн диктует свою структуру данных и свои reducers.

**ReAct state:**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
```

`messages` с `add_messages` reducer — канонический state для ReAct. Каждый node добавляет сообщения (HumanMessage, AIMessage, ToolMessage), а reducer аккумулирует их. `iteration_count` — опциональное поле для трекинга количества циклов.

**Reflection state:**

```python
class ReflectionState(TypedDict):
    student_work: str
    rubric: str
    draft: str
    feedback: str
    revision_count: int
    is_satisfactory: bool
```

Здесь `draft` и `feedback` **перезаписываются** на каждой итерации — нам нужна только последняя версия. `revision_count` инкрементируется. Reducers не нужны — стандартное поведение (перезапись) подходит.

**Plan-Execute state:**

```python
import operator

class PlanExecuteState(TypedDict):
    objective: str
    plan: list[str]
    current_step_index: int
    step_results: Annotated[list[dict], operator.add]
    final_result: str
```

`plan` перезаписывается при replanning. `step_results` накапливается через `operator.add`. `current_step_index` трекает прогресс.

**Map-Reduce state:**

```python
class MapReduceState(TypedDict):
    student_work: str
    criteria: list[dict]
    partial_results: Annotated[list[dict], operator.add]
    final_result: dict
```

`partial_results` с `operator.add` — ключевой элемент. Каждая параллельная ветка добавляет свой результат, reducer объединяет их в список. Node `aggregate` получает полный список и формирует `final_result`.

Важно: для Map-Reduce с `Send()` каждая ветка работает со своим «подсостоянием». `Send("node_name", {...})` передаёт данные конкретной ветке. Результат ветки мержится в основной state через reducer.

### 8. Дерево решений: выбор паттерна

При проектировании нового графа используй эту таблицу:

| Характеристика задачи | Рекомендуемый подход | Пример из assessment-домена |
|----------------------|---------------------|---------------------------|
| Простой запрос, без tools, один шаг | LCEL chain | «Оцени эссе по рубрике» |
| Нужны tools, агент решает что и когда | **ReAct** | «Проанализируй и оцени работу, используя доступные инструменты» |
| Качество критично, есть критерии проверки | **Reflection** | «Оцени работу и проверь, что оценка обоснована и консистентна» |
| Много шагов, порядок зависит от данных | **Plan-Execute** | «Оцени портфолио из 5 работ с итоговой рекомендацией» |
| Независимые подзадачи, можно параллелить | **Map-Reduce** | «Оцени работу по 5 критериям независимо» |
| Tools + проверка качества | **ReAct + Reflection** | «Собери данные tools, оцени, проверь и улучши» |
| Сложный план + параллельные шаги | **Plan-Execute + Map-Reduce** | «Составь план оценки, независимые шаги запусти параллельно» |

Блок-схема принятия решений:

```
Задача →
  ├── Один шаг, без tools? → LCEL chain
  ├── Нужны tools?
  │     ├── Да, reactive decision-making → ReAct
  │     └── Да, но шаги спланированы → Plan-Execute + tools
  ├── Качество нестабильно?
  │     └── Добавь Reflection к текущему подходу
  └── Есть независимые подзадачи?
        └── Map-Reduce (можно внутри Plan-Execute)
```

Критерий выбора — **минимальная достаточная сложность**. Каждый паттерн добавляет:

| Паттерн | Дополнительные LLM-вызовы | Latency множитель | Сложность кода |
|---------|--------------------------|-------------------|---------------|
| LCEL chain | 1 | 1x | Низкая |
| ReAct | 2–10 per iteration | 2–10x | Средняя |
| Reflection | 2 per iteration × N итераций | 2–8x | Средняя |
| Plan-Execute | 1 (plan) + N (steps) + N (replan) | 3–15x | Высокая |
| Map-Reduce | K параллельных + 1 (reduce) | ~1.5–2x (параллельно) | Средняя |

---

## Справочник API

### `create_react_agent`

```python
from langgraph.prebuilt import create_react_agent
```

Создаёт готовый ReAct-граф с tool-calling циклом.

| Параметр | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `model` | `BaseChatModel` | да | LLM с поддержкой tool calling |
| `tools` | `list[BaseTool \| Callable]` | да | Список tools, доступных агенту |
| `state_modifier` | `str \| SystemMessage \| Callable` | нет | System prompt или функция модификации state |
| `state_schema` | `type[TypedDict]` | нет | Кастомная схема state (по умолчанию `AgentState`) |
| `checkpointer` | `BaseCheckpointSaver` | нет | Persistence для state |
| `interrupt_before` | `list[str]` | нет | Узлы, перед которыми граф останавливается |
| `interrupt_after` | `list[str]` | нет | Узлы, после которых граф останавливается |

Возвращает: `CompiledGraph` — скомпилированный граф с Runnable-интерфейсом.

```python
agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[count_words, analyze_structure],
    state_modifier="You are an expert academic assessor. Use tools to gather data before scoring.",
)

result = await agent.ainvoke(
    {"messages": [("human", "Assess this essay: ...")]},
    config={"recursion_limit": 25},
)
```

### `Send`

```python
from langgraph.types import Send
```

Создаёт директиву для динамического fan-out — запуска нескольких параллельных веток графа.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `node` | `str` | Имя целевого узла |
| `arg` | `dict \| Any` | Данные для передачи в узел |

Используется в conditional edges для Map-Reduce:

```python
def fan_out(state: MyState) -> list[Send]:
    return [Send("process", {"item": item}) for item in state["items"]]

graph.add_conditional_edges(START, fan_out)
```

### `Command`

```python
from langgraph.types import Command
```

Позволяет node динамически управлять маршрутизацией — указывать следующий узел и обновлять state одновременно.

| Поле | Тип | Описание |
|------|-----|----------|
| `goto` | `str \| list[str]` | Следующий узел (или узлы) |
| `update` | `dict` | Обновления state |
| `resume` | `Any` | Значение для возобновления после interrupt |

```python
from langgraph.types import Command

async def router_node(state: MyState) -> Command:
    if state["score"] < 50:
        return Command(goto="detailed_review", update={"needs_review": True})
    return Command(goto="finalize", update={"needs_review": False})
```

### `StateGraph` — композиция подграфов

```python
from langgraph.graph import StateGraph
```

Подграфы позволяют вложить один граф в другой как обычный node. Это основа для композиции паттернов.

```python
inner_graph = StateGraph(InnerState)
inner_graph.add_node(...)
inner_compiled = inner_graph.compile()

outer_graph = StateGraph(OuterState)
outer_graph.add_node("inner_step", inner_compiled)
```

Ключевое: state подграфа и родительского графа могут отличаться. LangGraph автоматически маппит общие поля.

### `RetryPolicy`

```python
from langgraph.pregel import RetryPolicy
```

Политика повторных попыток для node'ов — полезна при нестабильных API-вызовах.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `initial_interval` | `float` | 0.5 | Начальная задержка (секунды) |
| `backoff_factor` | `float` | 2.0 | Множитель для exponential backoff |
| `max_interval` | `float` | 128.0 | Максимальная задержка |
| `max_attempts` | `int` | 3 | Максимум попыток |
| `jitter` | `bool` | True | Добавлять случайный jitter |
| `retry_on` | `type[Exception] \| tuple \| Callable` | `Exception` | Какие исключения ретраить |

```python
graph.add_node(
    "assess",
    assess_node,
    retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
)
```

### `add_messages`

```python
from langgraph.graph.message import add_messages
```

Reducer для поля `messages` в state. Вместо перезаписи — добавляет новые сообщения к существующим. Также обрабатывает дедупликацию по `id` — если сообщение с таким `id` уже есть, оно обновляется.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### `ToolNode`

```python
from langgraph.prebuilt import ToolNode
```

Готовый node, который извлекает tool_calls из последнего AIMessage, выполняет соответствующие tools и возвращает ToolMessages.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `tools` | `list[BaseTool \| Callable]` | Список доступных tools |
| `handle_tool_errors` | `bool \| str \| Callable` | Как обрабатывать ошибки tools |

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([count_words, analyze_structure])
graph.add_node("tools", tool_node)
```

При `handle_tool_errors=True` (по умолчанию) ошибки выполнения tool возвращаются как текст ToolMessage, а не как исключение. Это позволяет LLM увидеть ошибку и отреагировать (например, вызвать tool с другими параметрами).

---

## Практика

### Пример 1. ReAct — reasoning + acting в цикле

Агент получает студенческую работу и решает, какие tools вызвать для сбора аналитики. После каждого tool-вызова он анализирует результат и решает — нужно ещё данных или пора выставлять оценку.

```python
import re
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


@tool
def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())


@tool
def check_citations(text: str) -> dict:
    """Check for citation patterns in the text."""
    patterns = [r'\([A-Z][a-z]+,?\s*\d{4}\)', r'\[[0-9]+\]']
    citations = []
    for p in patterns:
        citations.extend(re.findall(p, text))
    return {"citation_count": len(citations), "examples": citations[:5]}


@tool
def analyze_vocabulary(text: str) -> dict:
    """Analyze vocabulary complexity."""
    words = text.lower().split()
    unique = set(words)
    return {
        "total_words": len(words),
        "unique_words": len(unique),
        "vocabulary_richness": round(len(unique) / max(len(words), 1), 3),
    }


TOOLS = [count_words, check_citations, analyze_vocabulary]


class ReactState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int


llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(TOOLS)


async def agent_node(state: ReactState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def should_continue(state: ReactState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


graph = StateGraph(ReactState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

react_app = graph.compile()

student_work = (
    "The impact of climate change on biodiversity is a critical topic in modern ecology. "
    "Rising temperatures have led to shifts in species distribution patterns globally. "
    "However, some argue that natural adaptation can mitigate these effects. "
    "Studies by Smith (2020) and Johnson (2021) demonstrate significant coral reef degradation."
)

result = await react_app.ainvoke(
    {
        "messages": [
            ("system",
             "You are an academic assessor. Use tools to gather data about the work, "
             "then provide a score (0-100) and detailed feedback."),
            ("human", f"Assess this student work:\n\n{student_work}"),
        ],
        "iteration_count": 0,
    },
    config={"recursion_limit": 25},
)

for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Tool call: {tc['name']}")
    elif isinstance(msg, ToolMessage):
        print(f"  → {msg.content[:120]}")

print(f"\nIterations: {result['iteration_count']}")
print(f"\nAssessment:\n{result['messages'][-1].content[:500]}")
```

Граф содержит два узла: `agent` (вызов LLM) и `tools` (выполнение tool calls). Conditional edge `should_continue` проверяет наличие `tool_calls` в ответе LLM — если есть, цикл продолжается. `recursion_limit=25` ограничивает максимальное количество шагов графа (~12 итераций ReAct-цикла).

### Пример 2. Reflection — генерация и самопроверка

Generator создаёт оценку работы, Critic проверяет её по чек-листу (completeness, evidence, consistency, actionability). Если качество недостаточно — Generator получает feedback и улучшает результат. Цикл ограничен `MAX_REVISIONS`.

```python
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END


class ReflectionState(TypedDict):
    student_work: str
    criteria: list[str]
    draft: str
    feedback: str
    revision_count: int
    is_satisfactory: bool
    feedback_history: list[str]


llm = ChatAnthropic(model="claude-sonnet-4-20250514")

MAX_REVISIONS = 3


async def generate_node(state: ReflectionState) -> dict:
    criteria_text = ", ".join(state["criteria"])
    if state.get("feedback") and state.get("draft"):
        prompt = (
            f"Your previous assessment received feedback:\n{state['feedback']}\n\n"
            f"Revise your assessment. Criteria: {criteria_text}\n"
            f"Student work: {state['student_work']}\n"
            f"Previous assessment: {state['draft']}"
        )
    else:
        prompt = (
            f"Assess the student work on criteria: {criteria_text}. "
            f"For each criterion give a score (0-25) and evidence-based feedback.\n\n"
            f"Student work: {state['student_work']}"
        )
    response = await llm.ainvoke([("human", prompt)])
    return {"draft": response.content, "revision_count": state.get("revision_count", 0) + 1}


async def reflect_node(state: ReflectionState) -> dict:
    criteria_text = ", ".join(state["criteria"])
    prompt = (
        f"Review this assessment for quality.\n\n"
        f"Student work: {state['student_work']}\n"
        f"Criteria: {criteria_text}\n"
        f"Assessment: {state['draft']}\n\n"
        "Check:\n"
        "1. COMPLETENESS — all criteria addressed?\n"
        "2. EVIDENCE — scores justified with specific references?\n"
        "3. CONSISTENCY — scores match feedback?\n"
        "4. ACTIONABILITY — feedback specific enough to improve?\n\n"
        "If satisfactory, respond: SATISFACTORY\n"
        "Otherwise, provide specific feedback for improvement."
    )
    response = await llm.ainvoke([("human", prompt)])
    is_ok = "SATISFACTORY" in response.content.upper()
    history = [*state.get("feedback_history", []), response.content]
    return {"feedback": response.content, "is_satisfactory": is_ok, "feedback_history": history}


def should_continue(state: ReflectionState) -> str:
    if state.get("is_satisfactory"):
        return "done"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "done"
    return "revise"


graph = StateGraph(ReflectionState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)
graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {"revise": "generate", "done": END})

reflection_app = graph.compile()

result = await reflection_app.ainvoke({
    "student_work": (
        "Climate change affects biodiversity. Many species are dying. "
        "We need to do something about it."
    ),
    "criteria": ["content", "structure", "argumentation", "evidence"],
    "draft": "",
    "feedback": "",
    "revision_count": 0,
    "is_satisfactory": False,
    "feedback_history": [],
})

print(f"Revisions: {result['revision_count']}")
print(f"Satisfactory: {result['is_satisfactory']}")
for i, fb in enumerate(result["feedback_history"], 1):
    print(f"\n--- Feedback {i} ---")
    print(fb[:300])
print(f"\nFinal assessment:\n{result['draft'][:500]}")
```

Generator и Critic используют одну модель, но разные промпты, оптимизированные под свою роль. Промпт Critic'а содержит чёткий чек-лист — без него обратная связь будет слишком расплывчатой. Двойной критерий остановки: `is_satisfactory=True` или `revision_count >= MAX_REVISIONS` — без жёсткого лимита граф может зациклиться.

### Пример 3. Plan-and-Execute — планирование и пошаговое выполнение

Planner генерирует план через structured output (`Plan`). Executor выполняет каждый шаг с контекстом предыдущих результатов. Replanner корректирует оставшийся план после каждого шага — может добавлять, удалять или менять порядок шагов.

```python
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


class Step(BaseModel):
    description: str = Field(description="What this step should accomplish")


class Plan(BaseModel):
    steps: list[Step] = Field(description="Ordered list of steps")


class PlanExecuteState(TypedDict):
    objective: str
    student_works: list[str]
    plan: list[str]
    current_step_index: int
    step_results: Annotated[list[dict], operator.add]
    final_result: str


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(Plan)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a planning agent. Create a concrete step-by-step plan. "
     "Number of student works: {num_works}"),
    ("human",
     "Objective: {objective}\n\nWorks preview:\n{works_preview}\n\nCreate a plan."),
])

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an executor agent. Complete the given step thoroughly."),
    ("human",
     "Objective: {objective}\nCurrent step: {current_step}\n"
     "Previous results:\n{previous_results}\n\nStudent work:\n{relevant_work}\n\nExecute this step."),
])


async def plan_node(state: PlanExecuteState) -> dict:
    works_preview = "\n".join(
        f"Work {i+1}: {w[:200]}..." for i, w in enumerate(state["student_works"])
    )
    response = await structured_llm.ainvoke(
        PLANNER_PROMPT.format_messages(
            objective=state["objective"],
            num_works=len(state["student_works"]),
            works_preview=works_preview,
        )
    )
    return {"plan": [s.description for s in response.steps], "current_step_index": 0}


async def execute_node(state: PlanExecuteState) -> dict:
    idx = state["current_step_index"]
    previous_results = "\n".join(
        f"Step {r['step_index']+1}: {r['result'][:300]}"
        for r in state.get("step_results", [])
    ) or "No previous results."
    work_idx = min(idx, len(state["student_works"]) - 1)
    response = await llm.ainvoke(
        EXECUTOR_PROMPT.format_messages(
            objective=state["objective"],
            current_step=state["plan"][idx],
            previous_results=previous_results,
            relevant_work=state["student_works"][work_idx],
        )
    )
    return {
        "step_results": [{"step_index": idx, "step": state["plan"][idx], "result": response.content}],
        "current_step_index": idx + 1,
    }


async def replan_node(state: PlanExecuteState) -> dict:
    idx = state["current_step_index"]
    if idx >= len(state["plan"]):
        return {}
    completed = "\n".join(
        f"Step {r['step_index']+1}: {r['result'][:200]}"
        for r in state.get("step_results", [])
    )
    remaining = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(state["plan"][idx:], start=idx)
    )
    response = await structured_llm.ainvoke([
        ("system",
         "You are a replanning agent. Adjust the remaining plan if needed. "
         "Return empty steps list if the objective is already achieved."),
        ("human",
         f"Objective: {state['objective']}\n\nCompleted:\n{completed}\n\n"
         f"Remaining:\n{remaining}\n\nProvide updated remaining steps."),
    ])
    if response.steps:
        return {"plan": state["plan"][:idx] + [s.description for s in response.steps]}
    return {}


def should_continue(state: PlanExecuteState) -> str:
    if state["current_step_index"] >= len(state["plan"]):
        return "finalize"
    return "execute"


async def finalize_node(state: PlanExecuteState) -> dict:
    all_results = "\n\n".join(
        f"### Step {r['step_index']+1}: {r['step']}\n{r['result']}"
        for r in state.get("step_results", [])
    )
    response = await llm.ainvoke([("human",
        f"Objective: {state['objective']}\n\nAll step results:\n{all_results}\n\n"
        "Synthesize into a final recommendation."
    )])
    return {"final_result": response.content}


graph = StateGraph(PlanExecuteState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("replan", replan_node)
graph.add_node("finalize", finalize_node)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "replan")
graph.add_conditional_edges("replan", should_continue, {
    "execute": "execute",
    "finalize": "finalize",
})
graph.add_edge("finalize", END)

plan_app = graph.compile()

result = await plan_app.ainvoke(
    {
        "objective": "Assess the portfolio and provide a recommendation on student progress",
        "student_works": [
            "Essay about climate change impacts on marine ecosystems...",
            "Lab report on water quality analysis methodology...",
            "Research proposal for renewable energy study...",
        ],
        "plan": [],
        "current_step_index": 0,
        "step_results": [],
        "final_result": "",
    },
    config={"recursion_limit": 50},
)

print("Plan:")
for i, step in enumerate(result["plan"], 1):
    print(f"  {i}. {step}")
print(f"\nSteps executed: {len(result['step_results'])}")
print(f"\nFinal result:\n{result['final_result'][:500]}")
```

Structured output для плана (`with_structured_output(Plan)`) гарантирует стандартный формат — всегда список конкретных шагов, а не свободный текст. Replanner видит результаты выполненных шагов и может добавить или удалить оставшиеся — это ключевое отличие от статического плана. `recursion_limit=50` — Plan-Execute может потребовать много шагов графа.

### Пример 4. Map-Reduce — параллельная обработка с агрегацией

Каждый критерий оценки обрабатывается отдельным LLM-вызовом параллельно через `Send()`. Агрегирующий узел собирает все частичные результаты и формирует итоговое резюме.

```python
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    criterion_name: str
    score: int
    max_score: int
    feedback: str


class MapReduceState(TypedDict):
    student_work: str
    criteria: list[dict]
    partial_results: Annotated[list[dict], operator.add]
    overall_score: int
    max_overall_score: int
    summary: str


class CriterionState(TypedDict):
    student_work: str
    criterion: dict
    partial_results: Annotated[list[dict], operator.add]


llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(CriterionScore)


def fan_out_criteria(state: MapReduceState) -> list[Send]:
    return [
        Send("assess_criterion", {
            "student_work": state["student_work"],
            "criterion": c,
            "partial_results": [],
        })
        for c in state["criteria"]
    ]


async def assess_criterion_node(state: CriterionState) -> dict:
    c = state["criterion"]
    prompt = (
        f"You are an assessor focused on one criterion.\n\n"
        f"Criterion: {c['name']}\nDescription: {c['description']}\n"
        f"Max score: {c['max_score']}\n\n"
        f"Student work:\n{state['student_work']}\n\n"
        f"Evaluate ONLY on this criterion. Score 0 to {c['max_score']}."
    )
    result = await structured_llm.ainvoke([("human", prompt)])
    return {"partial_results": [{
        "criterion_name": result.criterion_name,
        "score": result.score,
        "max_score": result.max_score,
        "feedback": result.feedback,
    }]}


async def aggregate_node(state: MapReduceState) -> dict:
    results = state["partial_results"]
    overall = sum(r["score"] for r in results)
    max_overall = sum(r["max_score"] for r in results)
    results_text = "\n".join(
        f"- {r['criterion_name']}: {r['score']}/{r['max_score']} — {r['feedback']}"
        for r in results
    )
    response = await llm.ainvoke([("human",
        f"Synthesize these evaluations into a 2-3 sentence summary:\n\n{results_text}"
    )])
    return {
        "overall_score": overall,
        "max_overall_score": max_overall,
        "summary": response.content,
    }


graph = StateGraph(MapReduceState)
graph.add_node("assess_criterion", assess_criterion_node)
graph.add_node("aggregate", aggregate_node)
graph.add_conditional_edges(START, fan_out_criteria)
graph.add_edge("assess_criterion", "aggregate")
graph.add_edge("aggregate", END)

mr_app = graph.compile()

result = await mr_app.ainvoke({
    "student_work": (
        "The impact of climate change on biodiversity is a critical topic. "
        "Rising temperatures lead to shifts in species distribution. "
        "Studies show coral reef degradation is accelerating."
    ),
    "criteria": [
        {"name": "content", "description": "Depth and accuracy of content", "max_score": 25},
        {"name": "structure", "description": "Organization and logical flow", "max_score": 25},
        {"name": "argumentation", "description": "Quality of arguments and evidence", "max_score": 25},
        {"name": "language", "description": "Grammar, style, clarity", "max_score": 25},
    ],
    "partial_results": [],
    "overall_score": 0,
    "max_overall_score": 0,
    "summary": "",
})

for r in result["partial_results"]:
    print(f"{r['criterion_name']}: {r['score']}/{r['max_score']}")
    print(f"  {r['feedback'][:150]}\n")
print(f"Overall: {result['overall_score']}/{result['max_overall_score']}")
print(f"\nSummary: {result['summary'][:300]}")
```

`Send()` создаёт параллельные ветки — по одной на каждый критерий. `CriterionState` изолирует данные каждой ветки. `operator.add` на поле `partial_results` объединяет результаты всех веток в один список. Node `aggregate` суммирует баллы детерминировано и генерирует текстовое резюме через LLM.

### Связь с теорией

Каждый пример реализует один из четырёх паттернов из теоретической части:

| Пример | Паттерн | Ключевой механизм |
|--------|---------|-------------------|
| Пример 1 | ReAct (раздел 2) | `bind_tools` + `should_continue` conditional edge |
| Пример 2 | Reflection (раздел 3) | Generator + Critic + conditional loop |
| Пример 3 | Plan-Execute (раздел 4) | Structured planner + executor + replanner cycle |
| Пример 4 | Map-Reduce (раздел 5) | `Send()` fan-out + `operator.add` reducer |

**State design** (раздел 7) реализован в каждом графе:
- ReAct: `messages` с `add_messages` reducer
- Reflection: `draft`/`feedback` с перезаписью, `revision_count` для трекинга
- Plan-Execute: `plan` с перезаписью при replanning, `step_results` с `operator.add`
- Map-Reduce: `partial_results` с `operator.add`, `CriterionState` для параллельных веток

Паттерны можно **комбинировать** (раздел 6): например, обернуть ReAct-агента в Reflection-цикл для проверки качества оценки, или использовать Map-Reduce внутри Plan-Execute для параллельного выполнения независимых шагов.

---

## Чеклист самопроверки

- [ ] Можешь объяснить разницу между ReAct и простым tool calling? Когда одного вызова tools достаточно, а когда нужен ReAct-цикл?
- [ ] Нарисуй (на бумаге или в коде) структуру графа для Reflection: какие узлы, какие edges, какой conditional edge определяет остановку?
- [ ] Почему в Reflection используются два разных промпта (Generator и Critic)? Что будет, если использовать один?
- [ ] Объясни роль Replanner в Plan-Execute. Когда он добавляет шаги? Когда удаляет? Что будет без него?
- [ ] Как `Send()` реализует параллелизм в Map-Reduce? Почему нельзя просто запустить несколько node'ов?
- [ ] Почему `partial_results` использует `Annotated[list[dict], operator.add]`? Что произойдёт без reducer'а?
- [ ] Запусти пример ReAct и посмотри вывод tool calls — какие tools вызвала модель и почему?
- [ ] Запусти пример Reflection с `MAX_REVISIONS=1` и `MAX_REVISIONS=3` — сравни качество финальной оценки. Улучшился ли результат?
- [ ] Запусти пример Map-Reduce с 2 критериями и с 5 — как изменилось время выполнения? Почему?
- [ ] Выбери паттерн для задачи: «Проверь 10 домашних работ студентов по одной рубрике и составь рейтинг». Обоснуй выбор, используя дерево решений из раздела 8.

---

## Частые ошибки

### 1. Нет ограничения итераций в ReAct

```python
result = await agent.ainvoke({"messages": messages})
```

Без `recursion_limit` агент может зациклиться, вызывая tools бесконечно. Это приведёт к огромному расходу токенов и timeout'ам.

```python
result = await agent.ainvoke(
    {"messages": messages},
    config={"recursion_limit": 25},
)
```

Всегда устанавливай `recursion_limit`. Для production — от 15 до 30 в зависимости от задачи.

### 2. Reflection без чёткого чек-листа в промпте Critic'а

```python
critic_prompt = "Review this assessment and provide feedback if needed."
```

Расплывчатый промпт приводит к расплывчатому feedback'у. Critic будет говорить «looks good» или давать общие замечания, которые Generator не сможет использовать.

```python
critic_prompt = (
    "Review this assessment on these dimensions:\n"
    "1. COMPLETENESS: Does it address ALL criteria?\n"
    "2. EVIDENCE: Are scores supported by specific quotes?\n"
    "3. CONSISTENCY: Do scores match the feedback?\n"
    "If satisfactory, respond: SATISFACTORY\n"
    "Otherwise, list specific issues."
)
```

### 3. Map-Reduce без reducer на `partial_results`

```python
class MapReduceState(TypedDict):
    partial_results: list[dict]
```

Без `Annotated[..., operator.add]` каждая параллельная ветка **перезапишет** `partial_results` своим результатом. В итоге `aggregate` получит результат только последней завершившейся ветки.

```python
class MapReduceState(TypedDict):
    partial_results: Annotated[list[dict], operator.add]
```

### 4. Plan-Execute с Planner без structured output

```python
plan_response = await llm.ainvoke("Create a plan...")
plan = plan_response.content.split("\n")
```

Парсинг текста ненадёжен: модель может добавить вступительный текст, нумерацию, пустые строки. Structured output гарантирует формат.

```python
class Plan(BaseModel):
    steps: list[Step]

structured_llm = llm.with_structured_output(Plan)
plan_response = await structured_llm.ainvoke(messages)
plan = [step.description for step in plan_response.steps]
```

### 5. Слишком много паттернов для простой задачи

```python
graph = build_plan_execute_with_react_and_reflection_and_map_reduce(...)
```

Комбинирование 3–4 паттернов для задачи, которая решается простым LCEL chain, — это antipattern. Каждый паттерн добавляет latency (×2–10), стоимость и точки отказа.

Правило: начинай с LCEL chain. Добавляй паттерн, только если простой подход даёт неудовлетворительные результаты.

### 6. Send() с мутабельными данными

```python
shared_context = {"key": "value"}

def fan_out(state):
    return [
        Send("process", {"item": item, "context": shared_context})
        for item in state["items"]
    ]
```

Если параллельные ветки мутируют `shared_context`, возникнут race conditions. Каждый `Send()` должен получать независимую копию данных.

```python
def fan_out(state):
    return [
        Send("process", {"item": item, "context": {**shared_context}})
        for item in state["items"]
    ]
```

---

## Что читать дальше

- [LangGraph Concepts: Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) — расширение паттернов на мультиагентные системы
- [LangGraph How-To: Create ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) — официальный гайд по ReAct
- [LangGraph How-To: Map-Reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) — fan-out / fan-in паттерны
- [LangGraph How-To: Subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/) — композиция графов
- [Plan-and-Execute Agent (LangGraph)](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/) — tutorial по Plan-Execute
- [Reflexion Paper (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366) — академическая основа паттерна Reflection
- [ReAct Paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) — оригинальная работа по ReAct
- [LangGraph Retry Policies](https://langchain-ai.github.io/langgraph/how-tos/node-retries/) — обработка ошибок в node'ах

**Следующая тема:** [Тема 14: Multi-Agent Systems](topic_14_multi_agent.md) — координация нескольких агентов в единой системе.

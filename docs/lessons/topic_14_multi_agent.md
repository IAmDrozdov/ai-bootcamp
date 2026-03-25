# Тема 14: Multi-Agent Systems — команды агентов

> **Пререквизиты:** [Тема 6 (LangGraph)](topic_06_langgraph_agents.md), [Тема 11 (Tool Use)](topic_11_tool_use.md), [Тема 13 (Agentic Patterns)](topic_13_agentic_patterns.md)
> **Что добавляем в проект:** `app/api/v1/multi_agent.py`, `app/graph/agents/analyzer.py`, `app/graph/agents/scorer.py`, `app/graph/agents/reviewer.py`, `app/graph/multi_agent.py`, `app/schemas/multi_agent.py`
> **Зависимости:** `langgraph`, `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. Зачем мульти-агенты

В теме 6 мы построили первый граф — единый `StateGraph`, в котором все узлы, tools и промпты живут в одном пространстве. Для простых пайплайнов (анализ → оценка → формат) этого достаточно. Но по мере усложнения системы возникают проблемы.

**Проблема масштабирования единого агента.** Представь агента, у которого 20+ tools: подсчёт слов, анализ структуры, проверка цитирования, оценка по рубрике, проверка плагиата, форматирование отчёта, отправка нотификаций. Модель получает описание всех tools в system prompt — это сотни токенов контекста. На практике, когда tool'ов больше 10-15, модель начинает:

- **Путаться в выборе tool** — вызывает `format_report` вместо `analyze_structure`, потому что описания похожи
- **Игнорировать tools** — «забывает» о существовании tool'ов, описанных в конце длинного списка
- **Генерировать невалидные вызовы** — путает аргументы между tool'ами с похожей сигнатурой

Промпт тоже растёт: инструкции для анализа, инструкции для оценки, инструкции для ревью — всё в одном system message. Модели сложнее следовать длинным многозадачным инструкциям, чем коротким фокусированным.

**Решение — декомпозиция.** Вместо одного агента с 20 tools создаём несколько специализированных агентов, каждый с 3-5 tools и фокусированным промптом:

| Агент | Задача | Tools |
|-------|--------|-------|
| Analyzer | Анализ текста студента | `count_words`, `analyze_structure`, `check_citations` |
| Scorer | Выставление оценок по критериям | `score_criterion`, `calculate_total` |
| Reviewer | Проверка качества оценки | `validate_scores`, `check_consistency` |

**Аналогия.** Один человек-оркестр может сыграть простую мелодию. Но симфонию играет оркестр — группа музыкантов, каждый мастер своего инструмента, под управлением дирижёра. В мульти-агентной системе дирижёр — это supervisor, а музыканты — специализированные агенты-workers.

В нашем проекте assessment system — идеальный кандидат для мульти-агентности. Процесс оценки естественно разбивается на фазы: сначала нужно **проанализировать** работу (структура, длина, цитаты), затем **оценить** по рубрике (баллы за каждый критерий), и наконец **проверить** качество оценки (консистентность, обоснованность). Каждая фаза — отдельная экспертиза, отдельный промпт, отдельный набор инструментов.

### 2. Архитектуры мульти-агентных систем

Существует четыре основных архитектуры координации агентов. Выбор зависит от характера задачи, степени связанности между агентами и требований к гибкости.

**Supervisor — центральный координатор.**

Один агент (supervisor) принимает задачу и решает, какому worker-агенту её делегировать. Worker выполняет работу и возвращает результат supervisor'у. Supervisor анализирует результат и решает: отправить другому worker'у, вернуть к тому же, или завершить.

```
User → Supervisor → Analyzer → Supervisor → Scorer → Supervisor → END
```

Supervisor видит полную картину и может динамически менять порядок вызовов. Это самая распространённая архитектура — она проста в реализации и отладке.

**Hierarchical — дерево координаторов.**

Расширение supervisor-паттерна для больших систем. Top-level supervisor делегирует mid-level supervisor'ам, те — worker'ам. Пример: `Head Assessor → Analysis Team Lead → (Text Analyzer, Citation Checker) + Scoring Team Lead → (Rubric Scorer, Consistency Checker)`.

```
Head Supervisor
├── Analysis Supervisor
│   ├── Text Analyzer
│   └── Citation Checker
└── Scoring Supervisor
    ├── Rubric Scorer
    └── Consistency Checker
```

Используется, когда worker'ов десятки и один supervisor не справляется с маршрутизацией.

**Network / Swarm — равноправные агенты.**

Нет центрального координатора. Каждый агент решает, кому передать управление дальше. Агент A может передать агенту B, тот — агенту C или обратно A. Паттерн гибкий, но сложный в отладке — нет единой точки контроля.

```
Analyzer ↔ Scorer ↔ Reviewer
    ↑___________↓
```

Применяется в creative/research задачах, где порядок действий непредсказуем.

**Sequential Pipeline — линейная цепочка.**

Фиксированный порядок: каждый агент выполняет свой этап и передаёт результат следующему. Нет обратной связи, нет динамической маршрутизации.

```
Analyzer → Scorer → Reviewer → END
```

Простейшая архитектура. Подходит, когда порядок этапов всегда одинаков.

**Сравнительная таблица:**

| Архитектура | Гибкость | Сложность | Отладка | Когда использовать |
|-------------|----------|-----------|---------|-------------------|
| Supervisor | Высокая | Средняя | Легко — один control flow | Большинство задач, динамический workflow |
| Hierarchical | Очень высокая | Высокая | Средне — дерево вызовов | 10+ агентов, подкоманды |
| Network / Swarm | Максимальная | Очень высокая | Сложно — нет центра | Creative, research, brainstorming |
| Sequential Pipeline | Низкая | Низкая | Тривиально — линейный поток | Фиксированный порядок этапов |

**В нашем проекте** мы реализуем два варианта: **supervisor** (для гибкого assessment, где supervisor решает какие агенты нужны) и **sequential pipeline** (для стандартного потока analyzer → scorer → reviewer). Дополнительно покажем **параллельное выполнение** — два scorer'а оценивают независимо, arbiter разрешает расхождения.

### 3. Supervisor pattern в LangGraph

Supervisor — это обычный `StateGraph`, в котором один узел (supervisor) принимает решение о маршрутизации. В LangGraph supervisor реализуется как LLM-вызов с function calling: модель возвращает имя следующего агента, и `Command(goto=...)` направляет граф к нужному узлу.

**Supervisor prompt** — ключевой элемент. Он описывает каждого worker-агента: что тот умеет, когда его вызывать, какие данные ожидает. Чем точнее описание workers, тем лучше supervisor принимает решения.

```python
SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor coordinating an assessment team.

Your team members:
- analyzer: Analyzes student work structure, word count, citations, vocabulary.
  Call when you need text analysis before scoring.
- scorer: Scores student work against a rubric criteria.
  Call after analysis is complete and rubric is provided.
- reviewer: Reviews scores for consistency and fairness.
  Call after scoring to validate results.

Given the conversation so far, decide which team member should act next,
or respond with FINISH if the task is complete.
"""
```

**Router function.** Supervisor возвращает структурированный ответ — имя следующего агента. Это реализуется через `with_structured_output` или через tool calling.

```python
from pydantic import BaseModel
from typing import Literal


class SupervisorDecision(BaseModel):
    next_agent: Literal["analyzer", "scorer", "reviewer", "FINISH"]
    reasoning: str
```

**Command для маршрутизации.** LangGraph предоставляет `Command` из `langgraph.types` — объект, который одновременно обновляет state и указывает следующий узел:

```python
from langgraph.types import Command


async def supervisor_node(state: MultiAgentState) -> Command[Literal["analyzer", "scorer", "reviewer", "__end__"]]:
    response = await supervisor_chain.ainvoke({
        "messages": state["messages"],
        "team_members": "analyzer, scorer, reviewer",
    })

    if response.next_agent == "FINISH":
        return Command(goto="__end__", update={"final_decision": response.reasoning})

    return Command(
        goto=response.next_agent,
        update={"messages": [HumanMessage(content=f"Supervisor delegates to {response.next_agent}: {response.reasoning}")]},
    )
```

**Цикл supervisor'а.** После каждого worker'а управление возвращается supervisor'у. Он анализирует результат и решает: вызвать другого worker'а, повторить текущего, или завершить. Это позволяет supervisor'у адаптировать workflow на лету — например, если analyzer обнаружил проблемы с цитированием, supervisor может отправить работу на повторный анализ перед оценкой.

```python
graph = StateGraph(MultiAgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("analyzer", analyzer_node)
graph.add_node("scorer", scorer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "supervisor")
graph.add_edge("analyzer", "supervisor")
graph.add_edge("scorer", "supervisor")
graph.add_edge("reviewer", "supervisor")

app = graph.compile()
```

Обрати внимание: от каждого worker'а идёт ребро обратно к supervisor'у. Маршрутизация от supervisor'а к workers происходит через `Command(goto=...)`, поэтому явные рёбра от supervisor'а к workers не нужны.

### 4. Subgraphs — агенты как вложенные графы

Каждый агент в мульти-агентной системе может быть не просто функцией, а полноценным `StateGraph` — со своими узлами, рёбрами, условной логикой и даже tool calling. Такой вложенный граф называется **subgraph**.

**Зачем subgraphs?** Если worker-агент — это обычная функция, он ограничен одним LLM-вызовом. Но реальный агент может требовать цикла: вызвать tool → проанализировать результат → вызвать другой tool → сформировать ответ. Subgraph позволяет реализовать эту логику внутри worker'а.

**Создание subgraph.**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AnalyzerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    analysis_result: str


def analyze_text(state: AnalyzerState) -> dict:
    word_count = len(state["student_work"].split())
    paragraphs = state["student_work"].count("\n\n") + 1
    return {
        "analysis_result": f"Words: {word_count}, Paragraphs: {paragraphs}",
    }


analyzer_graph = StateGraph(AnalyzerState)
analyzer_graph.add_node("analyze", analyze_text)
analyzer_graph.add_edge(START, "analyze")
analyzer_graph.add_edge("analyze", END)

analyzer_compiled = analyzer_graph.compile()
```

**Вложение в родительский граф.** Скомпилированный subgraph добавляется как обычный узел:

```python
parent_graph = StateGraph(MultiAgentState)
parent_graph.add_node("analyzer", analyzer_compiled)
```

LangGraph автоматически вызовет subgraph, передав ему ту часть parent state, которая совпадает по ключам с state subgraph'а.

**State mapping.** Ключевой вопрос: как данные из parent state попадают в subgraph state и обратно? LangGraph использует совпадение имён полей. Если parent state содержит поле `student_work` и subgraph state тоже — значение будет передано автоматически. Если имена не совпадают, нужен явный маппинг.

**Input/Output schemas.** Для более чёткого контроля можно определить отдельные input и output schemas:

```python
class AnalyzerInput(TypedDict):
    student_work: str


class AnalyzerOutput(TypedDict):
    analysis_result: str


analyzer_graph = StateGraph(AnalyzerState, input=AnalyzerInput, output=AnalyzerOutput)
```

Теперь при вызове subgraph'а из parent'а: на входе subgraph получит только `student_work`, на выходе вернёт только `analysis_result`. Остальные поля internal state (например, `messages`) остаются приватными для subgraph'а.

**Преимущества subgraphs:**

| Преимущество | Описание |
|-------------|----------|
| Изоляция | Internal state не утекает в parent graph |
| Переиспользование | Один subgraph можно использовать в разных parent graphs |
| Независимое тестирование | Subgraph тестируется как самостоятельная единица |
| Инкапсуляция | Сложность внутренней логики скрыта за простым интерфейсом |
| Разделение ответственности | Каждая команда разрабатывает свой subgraph |

### 5. Agent handoff — передача управления

Handoff — момент, когда один агент завершает работу и передаёт управление другому. Это ключевой механизм координации в мульти-агентной системе. От качества handoff'а зависит, получит ли следующий агент достаточно контекста для работы.

**Два протокола handoff'а в LangGraph:**

**1. Через shared state.** Агент пишет результат в state, следующий агент читает оттуда. Простой и прозрачный подход.

```python
async def analyzer_node(state: MultiAgentState) -> dict:
    analysis = await run_analysis(state["student_work"])
    return {"analyzer_result": analysis}


async def scorer_node(state: MultiAgentState) -> dict:
    analysis = state["analyzer_result"]
    scores = await run_scoring(analysis, state["rubric"])
    return {"scorer_result": scores}
```

Scorer читает `analyzer_result` из state — это и есть handoff через shared state.

**2. Через `Command(goto=...)`** с update. Агент явно указывает, кому передать управление и какие данные передать:

```python
from langgraph.types import Command


async def analyzer_node(state: MultiAgentState) -> Command[Literal["scorer"]]:
    analysis = await run_analysis(state["student_work"])
    return Command(
        goto="scorer",
        update={
            "analyzer_result": analysis,
            "messages": [AIMessage(content=f"Analysis complete: {analysis}")],
        },
    )
```

Этот подход даёт агенту контроль над маршрутизацией — он сам решает, кому передать результат.

**State transfer — какие данные передавать.** При handoff'е важно передать достаточно контекста, но не перегрузить следующего агента. Паттерн: каждый агент читает то, что ему нужно, и пишет свой результат:

```
Analyzer → пишет analyzer_result (структура, word count, цитаты)
    ↓
Scorer → читает analyzer_result + rubric → пишет scorer_result (баллы)
    ↓
Reviewer → читает analyzer_result + scorer_result → пишет reviewer_result (валидация)
```

**Human-in-the-loop между агентами.** LangGraph позволяет поставить паузу (`interrupt`) между агентами — например, после Scorer и перед Reviewer, чтобы преподаватель мог посмотреть оценки перед финальным ревью:

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["reviewer"],
)
```

### 6. Shared vs Isolated state

Управление state'ом — одно из ключевых архитектурных решений в мульти-агентной системе. Два крайних подхода: полностью общий state и полностью изолированный.

**Shared state** — все агенты читают и пишут в один словарь. Любой агент видит результаты любого другого агента.

```python
class SharedState(TypedDict):
    student_work: str
    rubric: str
    messages: Annotated[list[AnyMessage], add_messages]
    analyzer_result: str
    scorer_result: dict
    reviewer_result: dict
    final_result: dict
```

Плюсы: простота, полная прозрачность. Минусы: любой агент может случайно перезаписать чужие данные; state разрастается; все агенты тесно связаны через структуру state.

**Isolated state** — каждый агент имеет свой internal state. С родительским графом он общается через узкий интерфейс (input/output schemas).

```python
class AnalyzerInternalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    word_count: int
    paragraph_count: int
    citations: list[str]
    analysis_result: str
```

Родительский граф видит только `student_work` (вход) и `analysis_result` (выход). Внутренние поля (`word_count`, `paragraph_count`, `citations`) скрыты.

**Проблемы shared state на практике:**

1. **Конфликты** — два агента пишут в одно поле одновременно (при параллельном выполнении)
2. **Нечитаемый state** — 30+ полей, непонятно кто что пишет и читает
3. **Тесная связанность** — изменение state одного агента ломает другого

**Рекомендуемый паттерн: message passing.** Агенты общаются через `messages: Annotated[list, add_messages]` — каждый агент добавляет сообщение с результатом. Это работает как лог коммуникации:

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    final_result: dict


async def analyzer_node(state: MultiAgentState) -> dict:
    analysis = await run_analysis(state["student_work"])
    return {
        "messages": [AIMessage(content=f"[Analyzer] {analysis}", name="analyzer")],
    }
```

Каждый агент добавляет сообщение с именем (`name="analyzer"`), и все последующие агенты видят историю коммуникации.

**Гибридный паттерн** — каждый агент пишет в своё выделенное поле плюс в общий `messages`:

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    analyzer_result: str
    scorer_result: dict
    reviewer_result: dict
    final_result: dict
```

Это сочетает преимущества обоих подходов: структурированный доступ к данным через именованные поля + полная история коммуникации через messages.

### 7. Координация и коммуникация

Мульти-агентная система — это не просто набор агентов, а система с координацией. Как агенты узнают, когда действовать и какие данные использовать?

**Message-based коммуникация.** Агенты общаются через `messages` в shared state. Каждый агент видит сообщения предыдущих агентов и может на них реагировать. Это основной паттерн в LangGraph.

```python
async def scorer_node(state: MultiAgentState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

Scorer получает все предыдущие messages (включая результат Analyzer'а) и строит свой ответ на их основе.

**Event-based координация.** Conditional edges позволяют одному агенту триггерить другого на основе условий:

```python
def route_after_analysis(state: MultiAgentState) -> str:
    if "insufficient data" in state.get("analyzer_result", ""):
        return "request_more_info"
    return "scorer"


graph.add_conditional_edges("analyzer", route_after_analysis)
```

Если Analyzer обнаружил недостаток данных, управление идёт к узлу запроса дополнительной информации, а не к Scorer'у.

**Aggregation — сбор результатов от нескольких агентов.** Когда несколько агентов работают параллельно, нужен механизм сбора результатов. В LangGraph для этого используется `Send()`:

```python
from langgraph.types import Send


def fan_out_to_scorers(state: MultiAgentState) -> list[Send]:
    return [
        Send("scorer_a", {"student_work": state["student_work"], "rubric": state["rubric"]}),
        Send("scorer_b", {"student_work": state["student_work"], "rubric": state["rubric"]}),
    ]


graph.add_conditional_edges("analyzer", fan_out_to_scorers)
```

Оба scorer'а получат одинаковые данные и будут выполняться параллельно. Результаты обоих агрегируются в state.

**Conflict resolution.** Когда два scorer'а дают разные оценки, нужен arbiter — агент, который разрешает расхождения:

```python
async def arbiter_node(state: MultiAgentState) -> dict:
    scorer_a_result = state["scorer_a_result"]
    scorer_b_result = state["scorer_b_result"]

    if abs(scorer_a_result["total"] - scorer_b_result["total"]) <= 5:
        final = {
            "total": (scorer_a_result["total"] + scorer_b_result["total"]) // 2,
            "method": "average",
        }
    else:
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        resolution = await llm.ainvoke([
            SystemMessage(content="You are an arbiter. Two scorers disagree. Analyze both and produce a fair final score."),
            HumanMessage(content=f"Scorer A: {scorer_a_result}\nScorer B: {scorer_b_result}"),
        ])
        final = {"resolution": resolution.content, "method": "arbiter_llm"}

    return {"final_result": final}
```

Если расхождение незначительное (≤5 баллов), берётся среднее. Если значительное — LLM-arbiter анализирует оба результата и выносит решение.

### 8. Масштабирование и production

Мульти-агентные системы в production требуют внимания к стоимости, наблюдаемости и устойчивости к ошибкам.

**Выбор моделей для разных агентов.** Не все агенты одинаково сложны. Supervisor принимает решения о маршрутизации — ему нужна мощная модель с хорошим reasoning. Workers выполняют узкие задачи — можно использовать дешёвую модель:

| Роль | Модель | Стоимость | Обоснование |
|------|--------|-----------|-------------|
| Supervisor | claude-sonnet-4-20250514 | $$$ | Сложные решения о маршрутизации |
| Analyzer | claude-haiku-3-20250414 | $ | Простой текстовый анализ |
| Scorer | claude-sonnet-4-20250514 | $$$ | Точность оценки критична |
| Reviewer | claude-haiku-3-20250414 | $ | Проверка по формальным правилам |

```python
supervisor_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
analyzer_llm = ChatAnthropic(model="claude-haiku-3-20250414", temperature=0)
scorer_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
reviewer_llm = ChatAnthropic(model="claude-haiku-3-20250414", temperature=0)
```

**Cost optimization.** Не каждый запрос требует всех агентов. Supervisor может решить, что для короткого эссе достаточно scorer'а без предварительного анализа. Или что ревью не нужно для работы с идеальными оценками:

```python
async def supervisor_node(state):
    if len(state["student_work"].split()) < 50:
        return Command(goto="scorer", update={"skip_analysis": True})
    return Command(goto="analyzer")
```

**Observability.** В мульти-агентной системе критически важно видеть, какой агент что сделал, сколько это стоило и заняло времени. Каждый агент должен добавлять metadata:

```python
async def analyzer_node(state):
    config = {"metadata": {"agent": "analyzer", "step": "text_analysis"}}
    result = await analyzer_chain.ainvoke(state, config=config)
    return {"analyzer_result": result}
```

При использовании LangSmith или аналогичных трейсинг-систем metadata позволяет фильтровать трейсы по агенту, видеть стоимость каждого агента отдельно, и находить узкие места.

**Error handling.** Если один агент упал, вся система не должна падать. Стратегии:

1. **Retry** — повторить вызов агента (с тем же или изменённым промптом)
2. **Fallback** — переключиться на альтернативную реализацию (например, rule-based scorer вместо LLM-scorer)
3. **Skip** — пропустить агента и продолжить (reviewer не критичен — можно обойтись без ревью)
4. **Graceful degradation** — вернуть частичный результат с пометкой о неполноте

```python
async def safe_agent_call(agent_fn, state, fallback_fn=None):
    try:
        return await agent_fn(state)
    except Exception as e:
        if fallback_fn:
            return await fallback_fn(state)
        return {"error": str(e), "status": "agent_failed"}
```

**Тестирование.** Модульное тестирование каждого агента отдельно — главное преимущество мульти-агентной архитектуры. Каждый subgraph тестируется изолированно:

```python
async def test_analyzer_agent():
    result = await analyzer_compiled.ainvoke({
        "student_work": "This is a test essay with five paragraphs...",
        "messages": [],
    })
    assert "analysis_result" in result
    assert result["analysis_result"] != ""


async def test_scorer_agent():
    result = await scorer_compiled.ainvoke({
        "analyzer_result": "Words: 500, Paragraphs: 5, Citations: 3",
        "rubric": "Score content 1-10, structure 1-10",
        "messages": [],
    })
    assert "scorer_result" in result
```

Интеграционные тесты проверяют взаимодействие агентов. Unit-тесты проверяют каждого агента в изоляции.

---

## Справочник API

### `Command`

**Импорт:** `from langgraph.types import Command`

Объект, позволяющий узлу одновременно обновить state и указать следующий узел для выполнения. Используется вместо пары `return update` + conditional edge.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `goto` | `str \| list[str]` | Имя узла (или список узлов) для перехода |
| `update` | `dict \| None` | Обновления state, аналогично обычному return из node |
| `resume` | `Any \| None` | Значение для возобновления прерванного графа |
| `graph` | `Command.PARENT \| None` | Перенаправление в родительский граф |

**Type hint.** Для корректной визуализации графа используй `Command` с `Literal`:

```python
from langgraph.types import Command
from typing import Literal


async def my_node(state) -> Command[Literal["agent_a", "agent_b", "__end__"]]:
    return Command(goto="agent_a", update={"key": "value"})
```

`Literal` перечисляет все возможные пункты назначения — LangGraph использует это для построения рёбер при визуализации.

**Пример с PARENT:**

```python
async def subgraph_node(state) -> Command:
    return Command(goto="parent_node", update={"result": "done"}, graph=Command.PARENT)
```

### `create_react_agent`

**Импорт:** `from langgraph.prebuilt import create_react_agent`

Создаёт готовый ReAct-агент — граф с циклом LLM → tool → LLM → ... . Удобен для worker-агентов, которым нужен tool calling.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | `BaseChatModel` | — | LLM для агента |
| `tools` | `list[BaseTool \| Callable]` | — | Список tools |
| `prompt` | `str \| ChatPromptTemplate \| None` | `None` | System prompt для агента |
| `checkpointer` | `BaseCheckpointSaver \| None` | `None` | Checkpointer |
| `state_schema` | `type \| None` | `None` | Кастомная schema state |
| `state_modifier` | `Callable \| None` | `None` | Функция модификации state перед LLM |

**Возвращает:** `CompiledGraph` — готовый к использованию граф.

**Пример:**

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


@tool
def analyze_structure(text: str) -> dict:
    """Analyze text structure: paragraphs, sentences."""
    paragraphs = text.split("\n\n")
    return {"paragraphs": len(paragraphs), "avg_length": sum(len(p.split()) for p in paragraphs) // max(len(paragraphs), 1)}


analyzer_agent = create_react_agent(
    model=ChatAnthropic(model="claude-haiku-3-20250414"),
    tools=[count_words, analyze_structure],
    prompt="You are a text analyzer. Analyze the given student work thoroughly.",
)
```

### `StateGraph` — subgraph composition

**Импорт:** `from langgraph.graph import StateGraph, START, END`

При добавлении скомпилированного графа как узла родительского графа, LangGraph автоматически обрабатывает вызов subgraph'а.

**Паттерн вложения:**

```python
child_graph = StateGraph(ChildState)
child_graph.add_node("step", step_fn)
child_graph.add_edge(START, "step")
child_graph.add_edge("step", END)
child_compiled = child_graph.compile()

parent_graph = StateGraph(ParentState)
parent_graph.add_node("child", child_compiled)
```

**Input/Output schemas для изоляции:**

```python
class ChildInput(TypedDict):
    query: str

class ChildOutput(TypedDict):
    result: str

child_graph = StateGraph(ChildState, input=ChildInput, output=ChildOutput)
```

| Параметр | Описание |
|----------|----------|
| `input` | TypedDict, определяющий какие поля parent state передаются в subgraph |
| `output` | TypedDict, определяющий какие поля subgraph state возвращаются в parent |

### `Send`

**Импорт:** `from langgraph.types import Send`

Объект для fan-out — запуска нескольких экземпляров одного узла с разными входными данными. Используется для параллельного выполнения агентов.

**Конструктор:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `node` | `str` | Имя целевого узла |
| `arg` | `dict` | State для этого экземпляра |

**Пример — два scorer'а параллельно:**

```python
from langgraph.types import Send


def spawn_parallel_scorers(state: MultiAgentState) -> list[Send]:
    base_input = {
        "student_work": state["student_work"],
        "rubric": state["rubric"],
        "analyzer_result": state["analyzer_result"],
    }
    return [
        Send("scorer_strict", {**base_input, "scoring_style": "strict"}),
        Send("scorer_lenient", {**base_input, "scoring_style": "lenient"}),
    ]
```

Каждый `Send` создаёт независимую ветку выполнения. Результаты всех веток собираются через reducer в state.

### `add_messages`

**Импорт:** `from langgraph.graph.message import add_messages`

Reducer-функция для поля `messages` в state. Обрабатывает добавление, обновление и удаление сообщений по `id`.

**Поведение:**

| Сценарий | Что происходит |
|----------|---------------|
| Новое сообщение (новый `id`) | Добавляется в конец списка |
| Сообщение с существующим `id` | Заменяет существующее (update) |
| `RemoveMessage(id=...)` | Удаляет сообщение с данным id |

**Использование:**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Это стандартный паттерн для мульти-агентных систем: все агенты пишут в `messages`, reducer гарантирует корректное накопление.

### `MemorySaver` для multi-agent checkpointing

**Импорт:** `from langgraph.checkpoint.memory import MemorySaver`

В мульти-агентном контексте checkpointer сохраняет state на каждом шаге, включая переходы между агентами. Это позволяет:

- Возобновить выполнение с любого шага
- Поставить паузу между агентами (human-in-the-loop)
- Просмотреть историю state'ов для отладки

**Пример:**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["reviewer"],
)

config = {"configurable": {"thread_id": "assessment-42"}}
result = await app.ainvoke(initial_state, config)

state = await app.aget_state(config)
print(state.next)
```

### State annotation patterns для multi-agent

**Паттерн 1: Dedicated fields.**

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    analyzer_result: str
    scorer_result: dict
    reviewer_result: dict
```

Каждый агент пишет в своё поле. Плюс: типизированный доступ. Минус: state растёт с числом агентов.

**Паттерн 2: Messages only.**

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Все данные передаются через messages. Плюс: гибкость. Минус: парсинг messages для извлечения данных.

**Паттерн 3: Hybrid с operator.add.**

```python
from operator import add


class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    agent_results: Annotated[list[dict], add]
    final_result: dict
```

`agent_results` накапливает результаты всех агентов через `operator.add`. Каждый агент добавляет свой словарь в список.

---

## Практика: роутер `/api/v1/multi-agent`

### Шаг 1. Схемы — `app/schemas/multi_agent.py`

Определяем Pydantic-модели для входных и выходных данных мульти-агентного assessment'а.

```python
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    agent_name: str = Field(description="Name of the agent that produced this result")
    status: str = Field(description="completed, failed, or skipped")
    result: dict = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0)


class MultiAgentAssessmentRequest(BaseModel):
    student_work: str = Field(min_length=1)
    rubric: str = Field(default="Score the work on content (1-10), structure (1-10), and language (1-10).")
    mode: str = Field(default="supervisor", pattern="^(supervisor|pipeline|parallel)$")


class AnalysisResult(BaseModel):
    word_count: int = Field(default=0)
    paragraph_count: int = Field(default=0)
    citation_count: int = Field(default=0)
    structure_notes: str = Field(default="")
    vocabulary_level: str = Field(default="unknown")


class CriterionScore(BaseModel):
    name: str
    score: int = Field(ge=0, le=10)
    max_score: int = Field(default=10)
    justification: str = Field(default="")


class ScoringResult(BaseModel):
    criteria_scores: list[CriterionScore] = Field(default_factory=list)
    total_score: int = Field(default=0)
    max_total_score: int = Field(default=0)
    scoring_notes: str = Field(default="")


class ReviewResult(BaseModel):
    is_consistent: bool = Field(default=True)
    issues_found: list[str] = Field(default_factory=list)
    suggested_adjustments: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    review_notes: str = Field(default="")


class MultiAgentAssessmentResponse(BaseModel):
    analysis: AnalysisResult | None = None
    scoring: ScoringResult | None = None
    review: ReviewResult | None = None
    agent_trace: list[AgentResult] = Field(default_factory=list)
    final_summary: str = Field(default="")
    total_duration_ms: float = Field(default=0.0)


class SupervisorRequest(BaseModel):
    student_work: str = Field(min_length=1)
    rubric: str = Field(default="Score the work on content (1-10), structure (1-10), and language (1-10).")
    max_iterations: int = Field(default=5, ge=1, le=10)


class SupervisorResponse(BaseModel):
    decisions: list[dict] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)
    final_result: dict = Field(default_factory=dict)


class PipelineRequest(BaseModel):
    student_work: str = Field(min_length=1)
    rubric: str = Field(default="Score the work on content (1-10), structure (1-10), and language (1-10).")


class PipelineResponse(BaseModel):
    analysis: AnalysisResult
    scoring: ScoringResult
    review: ReviewResult
    agent_trace: list[AgentResult] = Field(default_factory=list)


class ParallelScoringRequest(BaseModel):
    student_work: str = Field(min_length=1)
    rubric: str = Field(default="Score the work on content (1-10), structure (1-10), and language (1-10).")
    analysis: str = Field(default="")


class ParallelScoringResponse(BaseModel):
    scorer_a_result: ScoringResult
    scorer_b_result: ScoringResult
    final_result: ScoringResult
    resolution_method: str = Field(description="average or arbiter_llm")
    agent_trace: list[AgentResult] = Field(default_factory=list)
```

Обрати внимание на `AgentResult` — модель для трейсинга каждого агента. Она содержит имя агента, статус, результат и время выполнения. Это даёт клиенту полную прозрачность: какие агенты были вызваны, в каком порядке, сколько времени заняли.

### Шаг 2. Agent: Analyzer — `app/graph/agents/analyzer.py`

Analyzer специализируется на текстовом анализе: подсчёт слов, определение структуры, анализ словарного запаса, подсчёт цитирований. Он не выставляет оценки — только собирает фактическую информацию о работе.

```python
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated

from app.config import get_settings


@tool
def count_words(text: str) -> int:
    """Count the total number of words in the text."""
    return len(text.split())


@tool
def count_paragraphs(text: str) -> int:
    """Count paragraphs separated by blank lines."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs)


@tool
def count_citations(text: str) -> int:
    """Count citation references like [1], (Author, 2024), etc."""
    import re
    bracket_refs = re.findall(r"\[\d+\]", text)
    paren_refs = re.findall(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.)?)?,\s*\d{4}\)", text)
    return len(bracket_refs) + len(paren_refs)


@tool
def analyze_vocabulary(text: str) -> dict:
    """Analyze vocabulary: unique words, average word length, long words ratio."""
    words = text.lower().split()
    if not words:
        return {"unique_words": 0, "avg_word_length": 0, "long_words_ratio": 0}
    unique = set(words)
    avg_len = sum(len(w) for w in words) / len(words)
    long_words = [w for w in words if len(w) > 8]
    return {
        "unique_words": len(unique),
        "avg_word_length": round(avg_len, 1),
        "long_words_ratio": round(len(long_words) / len(words), 3),
    }


ANALYZER_PROMPT = """You are a text analyzer for student work assessment.
Your job is to analyze the given text thoroughly using the available tools.

Always perform these steps:
1. Count words using count_words tool
2. Count paragraphs using count_paragraphs tool
3. Count citations using count_citations tool
4. Analyze vocabulary using analyze_vocabulary tool

After gathering all data, provide a comprehensive analysis summary.
Do NOT score or grade the work — only analyze factual properties."""


class AnalyzerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    analysis_result: str


def create_analyzer_agent() -> create_react_agent:
    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-haiku-3-20250414",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
    return create_react_agent(
        model=llm,
        tools=[count_words, count_paragraphs, count_citations, analyze_vocabulary],
        prompt=ANALYZER_PROMPT,
    )


async def run_analyzer(student_work: str) -> dict:
    agent = create_analyzer_agent()
    start = time.monotonic()
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"Analyze this student work:\n\n{student_work}")],
    })
    duration_ms = (time.monotonic() - start) * 1000
    last_message = result["messages"][-1]
    return {
        "analysis": last_message.content,
        "duration_ms": round(duration_ms, 1),
    }
```

Analyzer использует `create_react_agent` — готовый ReAct-граф, который в цикле вызывает tools до тех пор, пока не получит достаточно информации. Промпт инструктирует агента вызвать все четыре tools последовательно, а затем сформировать итоговый анализ.

Модель — `claude-haiku-3-20250414`: Analyzer выполняет механическую работу (подсчёт слов, парсинг), для которой мощная модель не нужна.

### Шаг 3. Agent: Scorer — `app/graph/agents/scorer.py`

Scorer принимает анализ + рубрику и выставляет баллы по каждому критерию. Это самый ответственный агент — от точности оценки зависит качество всей системы.

```python
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

from app.config import get_settings


class ScoringOutput(BaseModel):
    criteria_scores: list[dict] = Field(description="List of {name, score, max_score, justification}")
    total_score: int
    max_total_score: int
    scoring_notes: str


SCORER_PROMPT = """You are an expert scorer for student work assessment.
You receive text analysis results and a scoring rubric.

For each criterion in the rubric:
1. Evaluate the student work against the criterion
2. Assign a score within the allowed range
3. Provide a brief justification

Be fair but rigorous. Base scores on evidence from the analysis, not assumptions."""


class ScorerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    analysis: str
    scoring_result: str


async def score_node(state: ScorerState) -> dict:
    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(ScoringOutput)

    response = await structured_llm.ainvoke([
        SystemMessage(content=SCORER_PROMPT),
        HumanMessage(content=(
            f"Student work:\n{state['student_work']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Rubric:\n{state['rubric']}\n\n"
            "Score the work according to the rubric."
        )),
    ])

    return {"scoring_result": response.model_dump_json(), "messages": []}


scorer_graph = StateGraph(ScorerState)
scorer_graph.add_node("score", score_node)
scorer_graph.add_edge(START, "score")
scorer_graph.add_edge("score", END)

scorer_compiled = scorer_graph.compile()


async def run_scorer(student_work: str, rubric: str, analysis: str, style: str = "balanced") -> dict:
    start = time.monotonic()

    prompt_suffix = ""
    if style == "strict":
        prompt_suffix = "\nBe especially strict. Deduct points for any minor issues."
    elif style == "lenient":
        prompt_suffix = "\nBe generous. Give benefit of the doubt to the student."

    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(ScoringOutput)

    response = await structured_llm.ainvoke([
        SystemMessage(content=SCORER_PROMPT + prompt_suffix),
        HumanMessage(content=(
            f"Student work:\n{student_work}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Rubric:\n{rubric}\n\n"
            "Score the work according to the rubric."
        )),
    ])

    duration_ms = (time.monotonic() - start) * 1000
    return {
        "scoring": response.model_dump(),
        "duration_ms": round(duration_ms, 1),
    }
```

Scorer использует `with_structured_output` для гарантированного получения типизированного результата. Модель — `claude-sonnet-4-20250514`, потому что точность оценки критически важна.

Обрати внимание на параметр `style` в `run_scorer` — он позволяет запустить scorer с разными инструкциями (strict / lenient / balanced). Это используется в parallel-эндпоинте, где два scorer'а оценивают одну работу с разной строгостью.

### Шаг 4. Agent: Reviewer — `app/graph/agents/reviewer.py`

Reviewer — агент контроля качества. Он получает результаты анализа и оценки, проверяет консистентность и обоснованность. Это финальный gate перед выдачей результата.

```python
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

from app.config import get_settings


class ReviewOutput(BaseModel):
    is_consistent: bool = Field(description="Whether scores are consistent with analysis")
    issues_found: list[str] = Field(default_factory=list)
    suggested_adjustments: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    review_notes: str


REVIEWER_PROMPT = """You are a quality reviewer for student work assessments.
You receive the original student work, text analysis, and scoring results.

Your job:
1. Check if scores are consistent with the analysis findings
2. Identify any scoring issues (too harsh, too lenient, unjustified scores)
3. Suggest adjustments if needed
4. Rate your confidence in the overall assessment quality (0.0 to 1.0)

Be objective. A good assessment has scores that logically follow from the analysis."""


class ReviewerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    analysis: str
    scoring: str
    review_result: str


async def review_node(state: ReviewerState) -> dict:
    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-haiku-3-20250414",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(ReviewOutput)

    response = await structured_llm.ainvoke([
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(content=(
            f"Student work:\n{state['student_work']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Scoring:\n{state['scoring']}\n\n"
            "Review the assessment quality."
        )),
    ])

    return {"review_result": response.model_dump_json(), "messages": []}


reviewer_graph = StateGraph(ReviewerState)
reviewer_graph.add_node("review", review_node)
reviewer_graph.add_edge(START, "review")
reviewer_graph.add_edge("review", END)

reviewer_compiled = reviewer_graph.compile()


async def run_reviewer(student_work: str, analysis: str, scoring: str) -> dict:
    start = time.monotonic()

    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-haiku-3-20250414",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(ReviewOutput)

    response = await structured_llm.ainvoke([
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(content=(
            f"Student work:\n{student_work}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Scoring:\n{scoring}\n\n"
            "Review the assessment quality."
        )),
    ])

    duration_ms = (time.monotonic() - start) * 1000
    return {
        "review": response.model_dump(),
        "duration_ms": round(duration_ms, 1),
    }
```

Reviewer использует `claude-haiku-3-20250414` — его задача формальная (проверка консистентности), не требует мощного reasoning. Выходная модель `ReviewOutput` включает флаг `is_consistent`, список найденных проблем и уровень уверенности.

### Шаг 5. Multi-Agent Orchestrator — `app/graph/multi_agent.py`

Центральный модуль — здесь собираются все агенты в единый граф. Реализуем три архитектуры: supervisor, pipeline и parallel.

```python
import json
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.agents.analyzer import run_analyzer
from app.graph.agents.scorer import run_scorer
from app.graph.agents.reviewer import run_reviewer


class SupervisorDecision(BaseModel):
    next_agent: Literal["analyzer", "scorer", "reviewer", "FINISH"] = Field(
        description="Which agent to call next"
    )
    reasoning: str = Field(description="Why this agent is needed next")


class OrchestratorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    student_work: str
    rubric: str
    analyzer_result: str
    scorer_result: str
    reviewer_result: str
    agent_trace: list[dict]
    iteration: int


SUPERVISOR_SYSTEM = """You are a supervisor coordinating a student work assessment team.

Your team:
- analyzer: Analyzes text properties (word count, structure, citations, vocabulary).
  Call first to gather factual data about the student work.
- scorer: Scores the work against a rubric. Requires analysis results.
  Call after analyzer has completed.
- reviewer: Reviews scoring quality and consistency.
  Call after scorer has completed.

Rules:
1. Always call analyzer first unless analysis is already in state
2. Always call scorer after analyzer
3. Call reviewer after scorer to validate
4. Respond FINISH when all three have completed
5. If reviewer finds issues, you may re-call scorer

Current state will show which agents have already produced results."""


async def supervisor_node(
    state: OrchestratorState,
) -> Command[Literal["analyzer_worker", "scorer_worker", "reviewer_worker", "__end__"]]:
    settings = get_settings()
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=1024,
    )
    structured_llm = llm.with_structured_output(SupervisorDecision)

    status_parts = []
    if state.get("analyzer_result"):
        status_parts.append(f"Analyzer completed: {state['analyzer_result'][:200]}")
    if state.get("scorer_result"):
        status_parts.append(f"Scorer completed: {state['scorer_result'][:200]}")
    if state.get("reviewer_result"):
        status_parts.append(f"Reviewer completed: {state['reviewer_result'][:200]}")
    status = "\n".join(status_parts) if status_parts else "No agents have run yet."

    decision = await structured_llm.ainvoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=(
            f"Student work:\n{state['student_work'][:500]}\n\n"
            f"Rubric:\n{state['rubric']}\n\n"
            f"Current status:\n{status}\n\n"
            "Which agent should act next?"
        )),
    ])

    agent_map = {
        "analyzer": "analyzer_worker",
        "scorer": "scorer_worker",
        "reviewer": "reviewer_worker",
        "FINISH": "__end__",
    }

    target = agent_map[decision.next_agent]
    trace_entry = {
        "agent": "supervisor",
        "decision": decision.next_agent,
        "reasoning": decision.reasoning,
        "iteration": state.get("iteration", 0),
    }
    new_trace = state.get("agent_trace", []) + [trace_entry]

    return Command(
        goto=target,
        update={
            "agent_trace": new_trace,
            "iteration": state.get("iteration", 0) + 1,
            "messages": [AIMessage(content=f"Supervisor → {decision.next_agent}: {decision.reasoning}")],
        },
    )


async def analyzer_worker(state: OrchestratorState) -> dict:
    result = await run_analyzer(state["student_work"])
    trace_entry = {
        "agent": "analyzer",
        "status": "completed",
        "duration_ms": result["duration_ms"],
    }
    return {
        "analyzer_result": result["analysis"],
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[Analyzer] {result['analysis']}", name="analyzer")],
    }


async def scorer_worker(state: OrchestratorState) -> dict:
    analysis = state.get("analyzer_result", "No analysis available")
    result = await run_scorer(state["student_work"], state["rubric"], analysis)
    scoring_json = json.dumps(result["scoring"], indent=2)
    trace_entry = {
        "agent": "scorer",
        "status": "completed",
        "duration_ms": result["duration_ms"],
    }
    return {
        "scorer_result": scoring_json,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[Scorer] {scoring_json}", name="scorer")],
    }


async def reviewer_worker(state: OrchestratorState) -> dict:
    analysis = state.get("analyzer_result", "")
    scoring = state.get("scorer_result", "")
    result = await run_reviewer(state["student_work"], analysis, scoring)
    review_json = json.dumps(result["review"], indent=2)
    trace_entry = {
        "agent": "reviewer",
        "status": "completed",
        "duration_ms": result["duration_ms"],
    }
    return {
        "reviewer_result": review_json,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[Reviewer] {review_json}", name="reviewer")],
    }


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyzer_worker", analyzer_worker)
    graph.add_node("scorer_worker", scorer_worker)
    graph.add_node("reviewer_worker", reviewer_worker)

    graph.add_edge(START, "supervisor")
    graph.add_edge("analyzer_worker", "supervisor")
    graph.add_edge("scorer_worker", "supervisor")
    graph.add_edge("reviewer_worker", "supervisor")

    return graph


class PipelineState(TypedDict):
    student_work: str
    rubric: str
    analyzer_result: str
    scorer_result: str
    reviewer_result: str
    agent_trace: list[dict]


async def pipeline_analyze(state: PipelineState) -> dict:
    result = await run_analyzer(state["student_work"])
    return {
        "analyzer_result": result["analysis"],
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "analyzer", "status": "completed", "duration_ms": result["duration_ms"],
        }],
    }


async def pipeline_score(state: PipelineState) -> dict:
    result = await run_scorer(state["student_work"], state["rubric"], state["analyzer_result"])
    return {
        "scorer_result": json.dumps(result["scoring"]),
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "scorer", "status": "completed", "duration_ms": result["duration_ms"],
        }],
    }


async def pipeline_review(state: PipelineState) -> dict:
    result = await run_reviewer(state["student_work"], state["analyzer_result"], state["scorer_result"])
    return {
        "reviewer_result": json.dumps(result["review"]),
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "reviewer", "status": "completed", "duration_ms": result["duration_ms"],
        }],
    }


def build_pipeline_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("analyze", pipeline_analyze)
    graph.add_node("score", pipeline_score)
    graph.add_node("review", pipeline_review)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "score")
    graph.add_edge("score", "review")
    graph.add_edge("review", END)

    return graph


class ParallelState(TypedDict):
    student_work: str
    rubric: str
    analysis: str
    scorer_a_result: str
    scorer_b_result: str
    final_result: str
    agent_trace: list[dict]


async def parallel_scorer_a(state: ParallelState) -> dict:
    result = await run_scorer(state["student_work"], state["rubric"], state["analysis"], style="strict")
    return {
        "scorer_a_result": json.dumps(result["scoring"]),
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "scorer_a_strict", "status": "completed", "duration_ms": result["duration_ms"],
        }],
    }


async def parallel_scorer_b(state: ParallelState) -> dict:
    result = await run_scorer(state["student_work"], state["rubric"], state["analysis"], style="lenient")
    return {
        "scorer_b_result": json.dumps(result["scoring"]),
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "scorer_b_lenient", "status": "completed", "duration_ms": result["duration_ms"],
        }],
    }


async def arbiter_node(state: ParallelState) -> dict:
    score_a = json.loads(state["scorer_a_result"])
    score_b = json.loads(state["scorer_b_result"])

    total_a = score_a.get("total_score", 0)
    total_b = score_b.get("total_score", 0)
    max_score = max(score_a.get("max_total_score", 30), score_b.get("max_total_score", 30))

    if abs(total_a - total_b) <= 3:
        final = {
            "total_score": (total_a + total_b) // 2,
            "max_total_score": max_score,
            "resolution_method": "average",
            "scoring_notes": f"Scores close enough (A={total_a}, B={total_b}). Averaged.",
        }
    else:
        settings = get_settings()
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,
        )
        response = await llm.ainvoke([
            SystemMessage(content="You are an arbiter. Two scorers gave different scores. Analyze both and produce a fair final score as JSON with keys: total_score, max_total_score, scoring_notes."),
            HumanMessage(content=f"Scorer A (strict): {state['scorer_a_result']}\n\nScorer B (lenient): {state['scorer_b_result']}"),
        ])
        try:
            final = json.loads(response.content)
            final["resolution_method"] = "arbiter_llm"
        except json.JSONDecodeError:
            final = {
                "total_score": (total_a + total_b) // 2,
                "max_total_score": max_score,
                "resolution_method": "average_fallback",
                "scoring_notes": response.content,
            }

    return {
        "final_result": json.dumps(final),
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "arbiter", "status": "completed",
        }],
    }


def build_parallel_graph() -> StateGraph:
    graph = StateGraph(ParallelState)

    graph.add_node("scorer_a", parallel_scorer_a)
    graph.add_node("scorer_b", parallel_scorer_b)
    graph.add_node("arbiter", arbiter_node)

    graph.add_edge(START, "scorer_a")
    graph.add_edge(START, "scorer_b")
    graph.add_edge("scorer_a", "arbiter")
    graph.add_edge("scorer_b", "arbiter")
    graph.add_edge("arbiter", END)

    return graph
```

Модуль содержит три builder-функции:

- `build_supervisor_graph()` — граф с supervisor-узлом, который через `Command(goto=...)` маршрутизирует к workers. Рёбра от workers ведут обратно к supervisor, создавая цикл.
- `build_pipeline_graph()` — линейный граф `analyze → score → review`. Нет supervisor'а, фиксированный порядок.
- `build_parallel_graph()` — два scorer'а запускаются от START параллельно (обрати внимание: два ребра из START к разным узлам). Результаты собирает arbiter.

### Шаг 6. Router — `app/api/v1/multi_agent.py`

Роутер предоставляет четыре эндпоинта — по одному для каждого паттерна мульти-агентного взаимодействия.

```python
import json
import time

from fastapi import APIRouter, HTTPException

from app.graph.multi_agent import (
    build_supervisor_graph,
    build_pipeline_graph,
    build_parallel_graph,
)
from app.graph.agents.analyzer import run_analyzer
from app.schemas.multi_agent import (
    MultiAgentAssessmentRequest,
    MultiAgentAssessmentResponse,
    SupervisorRequest,
    SupervisorResponse,
    PipelineRequest,
    PipelineResponse,
    ParallelScoringRequest,
    ParallelScoringResponse,
    AnalysisResult,
    ScoringResult,
    ReviewResult,
    AgentResult,
)

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


@router.post("/assess")
async def multi_agent_assess(request: MultiAgentAssessmentRequest) -> MultiAgentAssessmentResponse:
    start = time.monotonic()

    if request.mode == "pipeline":
        graph = build_pipeline_graph().compile()
        result = await graph.ainvoke({
            "student_work": request.student_work,
            "rubric": request.rubric,
            "analyzer_result": "",
            "scorer_result": "",
            "reviewer_result": "",
            "agent_trace": [],
        })
    elif request.mode == "supervisor":
        graph = build_supervisor_graph().compile()
        result = await graph.ainvoke({
            "student_work": request.student_work,
            "rubric": request.rubric,
            "analyzer_result": "",
            "scorer_result": "",
            "reviewer_result": "",
            "agent_trace": [],
            "iteration": 0,
            "messages": [],
        })
    elif request.mode == "parallel":
        analysis_data = await run_analyzer(request.student_work)
        graph = build_parallel_graph().compile()
        result = await graph.ainvoke({
            "student_work": request.student_work,
            "rubric": request.rubric,
            "analysis": analysis_data["analysis"],
            "scorer_a_result": "",
            "scorer_b_result": "",
            "final_result": "",
            "agent_trace": [],
        })
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

    duration_ms = (time.monotonic() - start) * 1000
    agent_trace = [AgentResult(agent_name=t.get("agent", ""), status=t.get("status", "completed"), duration_ms=t.get("duration_ms", 0)) for t in result.get("agent_trace", [])]

    analysis = None
    if result.get("analyzer_result"):
        analysis = AnalysisResult(structure_notes=result["analyzer_result"])

    scoring = None
    scorer_data = result.get("scorer_result", "") or result.get("final_result", "")
    if scorer_data:
        try:
            parsed = json.loads(scorer_data) if isinstance(scorer_data, str) else scorer_data
            scoring = ScoringResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            scoring = ScoringResult(scoring_notes=str(scorer_data))

    review = None
    if result.get("reviewer_result"):
        try:
            parsed = json.loads(result["reviewer_result"])
            review = ReviewResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            review = ReviewResult(review_notes=str(result["reviewer_result"]))

    return MultiAgentAssessmentResponse(
        analysis=analysis,
        scoring=scoring,
        review=review,
        agent_trace=agent_trace,
        final_summary=f"Assessment completed in {request.mode} mode",
        total_duration_ms=round(duration_ms, 1),
    )


@router.post("/supervisor")
async def supervisor_assess(request: SupervisorRequest) -> SupervisorResponse:
    graph = build_supervisor_graph().compile()
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": request.rubric,
        "analyzer_result": "",
        "scorer_result": "",
        "reviewer_result": "",
        "agent_trace": [],
        "iteration": 0,
        "messages": [],
    })

    trace = result.get("agent_trace", [])
    decisions = [t for t in trace if t.get("agent") == "supervisor"]
    agent_results = [
        AgentResult(
            agent_name=t.get("agent", ""),
            status=t.get("status", "completed"),
            duration_ms=t.get("duration_ms", 0),
        )
        for t in trace if t.get("agent") != "supervisor"
    ]

    final = {}
    if result.get("scorer_result"):
        try:
            final = json.loads(result["scorer_result"])
        except (json.JSONDecodeError, TypeError):
            final = {"raw": result["scorer_result"]}

    return SupervisorResponse(decisions=decisions, agent_results=agent_results, final_result=final)


@router.post("/pipeline")
async def pipeline_assess(request: PipelineRequest) -> PipelineResponse:
    graph = build_pipeline_graph().compile()
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": request.rubric,
        "analyzer_result": "",
        "scorer_result": "",
        "reviewer_result": "",
        "agent_trace": [],
    })

    analysis = AnalysisResult(structure_notes=result.get("analyzer_result", ""))

    scoring = ScoringResult()
    if result.get("scorer_result"):
        try:
            parsed = json.loads(result["scorer_result"])
            scoring = ScoringResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            scoring = ScoringResult(scoring_notes=str(result["scorer_result"]))

    review = ReviewResult()
    if result.get("reviewer_result"):
        try:
            parsed = json.loads(result["reviewer_result"])
            review = ReviewResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            review = ReviewResult(review_notes=str(result["reviewer_result"]))

    agent_trace = [
        AgentResult(
            agent_name=t.get("agent", ""),
            status=t.get("status", "completed"),
            duration_ms=t.get("duration_ms", 0),
        )
        for t in result.get("agent_trace", [])
    ]

    return PipelineResponse(analysis=analysis, scoring=scoring, review=review, agent_trace=agent_trace)


@router.post("/parallel")
async def parallel_assess(request: ParallelScoringRequest) -> ParallelScoringResponse:
    analysis_text = request.analysis
    if not analysis_text:
        analysis_data = await run_analyzer(request.student_work)
        analysis_text = analysis_data["analysis"]

    graph = build_parallel_graph().compile()
    result = await graph.ainvoke({
        "student_work": request.student_work,
        "rubric": request.rubric,
        "analysis": analysis_text,
        "scorer_a_result": "",
        "scorer_b_result": "",
        "final_result": "",
        "agent_trace": [],
    })

    scorer_a = ScoringResult()
    if result.get("scorer_a_result"):
        try:
            parsed = json.loads(result["scorer_a_result"])
            scorer_a = ScoringResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            scorer_a = ScoringResult(scoring_notes=str(result["scorer_a_result"]))

    scorer_b = ScoringResult()
    if result.get("scorer_b_result"):
        try:
            parsed = json.loads(result["scorer_b_result"])
            scorer_b = ScoringResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            scorer_b = ScoringResult(scoring_notes=str(result["scorer_b_result"]))

    final = ScoringResult()
    resolution_method = "average"
    if result.get("final_result"):
        try:
            parsed = json.loads(result["final_result"])
            resolution_method = parsed.pop("resolution_method", "average")
            final = ScoringResult(**parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            final = ScoringResult(scoring_notes=str(result["final_result"]))

    agent_trace = [
        AgentResult(
            agent_name=t.get("agent", ""),
            status=t.get("status", "completed"),
            duration_ms=t.get("duration_ms", 0),
        )
        for t in result.get("agent_trace", [])
    ]

    return ParallelScoringResponse(
        scorer_a_result=scorer_a,
        scorer_b_result=scorer_b,
        final_result=final,
        resolution_method=resolution_method,
        agent_trace=agent_trace,
    )
```

**Четыре эндпоинта, четыре паттерна:**

| Эндпоинт | Паттерн | Описание |
|-----------|---------|----------|
| `POST /multi-agent/assess` | Universal | Принимает `mode` (supervisor/pipeline/parallel), запускает соответствующий граф |
| `POST /multi-agent/supervisor` | Supervisor | Supervisor-агент координирует workers, возвращает решения и трейс |
| `POST /multi-agent/pipeline` | Sequential | Фиксированный пайплайн: analyzer → scorer → reviewer |
| `POST /multi-agent/parallel` | Parallel + Arbiter | Два scorer'а параллельно, arbiter разрешает расхождения |

### Шаг 7. Registration + Testing

**Регистрация роутера.** Добавляем роутер в `app/api/router.py`:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, prompts, multi_agent

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(multi_agent.router)
```

**Проверка эндпоинтов.** Запускаем сервер и тестируем:

```bash
uvicorn app.main:app --reload
```

**Тест pipeline-эндпоинта:**

```bash
curl -X POST http://localhost:8000/api/v1/multi-agent/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The impact of climate change on biodiversity is a pressing concern. Rising temperatures affect ecosystems globally. Species migration patterns are shifting as habitats change. Marine ecosystems face acidification. Conservation efforts must adapt to these new realities. Research shows that 30% of species face extinction risk by 2050.",
    "rubric": "Score on: content depth (1-10), structure (1-10), evidence use (1-10)"
  }'
```

**Тест supervisor-эндпоинта:**

```bash
curl -X POST http://localhost:8000/api/v1/multi-agent/supervisor \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Machine learning has transformed many industries. Neural networks enable image recognition. Natural language processing powers chatbots and translation tools.",
    "rubric": "Score on: content depth (1-10), structure (1-10), evidence use (1-10)",
    "max_iterations": 5
  }'
```

**Тест parallel-эндпоинта:**

```bash
curl -X POST http://localhost:8000/api/v1/multi-agent/parallel \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Quantum computing represents a paradigm shift in computation. Unlike classical bits, qubits can exist in superposition states. This enables parallel processing of multiple states simultaneously. Major tech companies are investing billions in quantum research.",
    "rubric": "Score on: content depth (1-10), structure (1-10), evidence use (1-10)"
  }'
```

Обрати внимание на `agent_trace` в ответах — он показывает, какие агенты были вызваны, в каком порядке и сколько времени заняли. В supervisor-ответе дополнительно есть `decisions` — решения supervisor'а о маршрутизации.

### Связь с теорией

| Концепция из теории | Где реализовано |
|---------------------|----------------|
| Supervisor pattern (§3) | `build_supervisor_graph()` — supervisor_node маршрутизирует через `Command(goto=...)` |
| Sequential pipeline (§2) | `build_pipeline_graph()` — фиксированный `analyze → score → review` |
| Subgraphs (§4) | Каждый агент реализован как отдельный модуль со своим StateGraph |
| Agent handoff (§5) | Workers пишут результат в state, следующий агент читает оттуда |
| Shared state (§6) | `OrchestratorState` с полями `analyzer_result`, `scorer_result`, `reviewer_result` |
| Message passing (§6) | Workers добавляют `AIMessage` с `name` в `messages` |
| Parallel + arbiter (§7) | `build_parallel_graph()` — два scorer'а от START + arbiter_node |
| Model selection (§8) | Analyzer/Reviewer → haiku (дешёвая), Supervisor/Scorer → sonnet (мощная) |
| Observability (§8) | `agent_trace` — список AgentResult с duration_ms |
| Error handling (§8) | try/except с fallback при парсинге JSON в router'е |

---

## Чеклист самопроверки

- [ ] Объясни, почему один агент с 20+ tools хуже, чем несколько агентов с 3-5 tools каждый
- [ ] Нарисуй схему supervisor-паттерна: куда идут рёбра, кто принимает решения, где циклы
- [ ] В чём разница между `Command(goto="scorer")` и обычным conditional edge?
- [ ] Зачем subgraph'у отдельные `input` и `output` schemas? Что будет без них?
- [ ] Как работает handoff от Analyzer к Scorer через shared state? Какое поле связывает их?
- [ ] В чём проблема shared state при параллельном выполнении агентов? Как `Annotated[list, add]` решает её?
- [ ] Запусти `/multi-agent/pipeline` и `/multi-agent/supervisor` с одинаковой работой — сравни `agent_trace`
- [ ] Запусти `/multi-agent/parallel` — в каком случае вызывается LLM-arbiter, а в каком берётся среднее?
- [ ] Почему для Scorer используется sonnet, а для Analyzer — haiku? Когда можно сэкономить?
- [ ] Как добавить нового агента (например, `plagiarism_checker`) в supervisor-граф? Какие файлы нужно изменить?

---

## Частые ошибки

### 1. Supervisor зацикливается

```python
async def supervisor_node(state) -> Command[Literal["analyzer", "__end__"]]:
    return Command(goto="analyzer", update={})
```

Если supervisor всегда возвращает одного и того же агента, граф зацикливается бесконечно. Supervisor должен проверять state и учитывать, что агент уже отработал.

```python
async def supervisor_node(state) -> Command[Literal["analyzer", "scorer", "__end__"]]:
    if state.get("analyzer_result"):
        return Command(goto="scorer", update={})
    return Command(goto="analyzer", update={})
```

Всегда добавляй guard — лимит итераций или проверку завершённости.

### 2. State не передаётся в subgraph

```python
class ParentState(TypedDict):
    work: str
    result: str

class ChildState(TypedDict):
    student_work: str
    analysis: str
```

Имена полей не совпадают: `work` vs `student_work`. LangGraph не делает автоматический маппинг по смыслу — только по имени поля. Subgraph получит пустой `student_work`.

```python
class ParentState(TypedDict):
    student_work: str
    result: str

class ChildState(TypedDict):
    student_work: str
    analysis: str
```

Согласуй имена полей между parent и child state, или используй input/output schemas.

### 3. Параллельные агенты перезаписывают state

```python
class State(TypedDict):
    result: str

async def scorer_a(state) -> dict:
    return {"result": "Score A: 8/10"}

async def scorer_b(state) -> dict:
    return {"result": "Score B: 7/10"}
```

Оба агента пишут в `result`. При параллельном выполнении один перезапишет другого. Решение: отдельные поля или list-reducer.

```python
class State(TypedDict):
    scorer_a_result: str
    scorer_b_result: str
```

### 4. Забыли ребро от worker обратно к supervisor

```python
graph.add_edge(START, "supervisor")
graph.add_node("analyzer", analyzer_fn)
```

Без ребра `analyzer → supervisor` после выполнения analyzer граф не знает, куда идти дальше. Граф зависнет или выбросит ошибку.

```python
graph.add_edge(START, "supervisor")
graph.add_node("analyzer", analyzer_fn)
graph.add_edge("analyzer", "supervisor")
```

### 5. Command без type hint — граф теряет рёбра при визуализации

```python
async def supervisor_node(state):
    return Command(goto="analyzer")
```

Граф скомпилируется и будет работать, но визуализация (`graph.get_graph().draw_mermaid()`) не покажет рёбра от supervisor к workers — LangGraph не знает возможных пунктов назначения.

```python
async def supervisor_node(state) -> Command[Literal["analyzer", "scorer", "__end__"]]:
    return Command(goto="analyzer")
```

`Literal` в type hint перечисляет все возможные goto-значения.

### 6. Sync вызовы в async node

```python
async def analyzer_worker(state):
    result = run_analyzer_sync(state["student_work"])
    return {"analyzer_result": result}
```

Sync-вызов блокирует event loop. Если `run_analyzer_sync` содержит LLM-вызов — это может занять секунды, блокируя все остальные запросы FastAPI.

```python
async def analyzer_worker(state):
    result = await run_analyzer(state["student_work"])
    return {"analyzer_result": result}
```

Все LLM-вызовы и I/O-операции внутри node'ов должны быть async.

---

## Что читать дальше

- [LangGraph Multi-Agent Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) — архитектуры supervisor, swarm, hierarchical
- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/) — пошаговые примеры
- [Command API Reference](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command) — goto, update, graph
- [Subgraph How-To](https://langchain-ai.github.io/langgraph/how-tos/subgraph/) — вложенные графы, state mapping
- [Send for Fan-Out](https://langchain-ai.github.io/langgraph/how-tos/send/) — параллельный запуск узлов
- [LangGraph Supervisor Library](https://github.com/langchain-ai/langgraph-supervisor) — готовый supervisor-паттерн
- [LangGraph Swarm Library](https://github.com/langchain-ai/langgraph-swarm) — swarm-паттерн для peer-to-peer агентов

**Следующая тема:** [Тема 15: MCP (Model Context Protocol)](topic_15_mcp.md)

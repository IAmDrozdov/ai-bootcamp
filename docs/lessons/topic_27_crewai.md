# Тема 27: CrewAI — мульти-агентные системы с ролями

> **Пререквизиты:** [Тема 6](topic_06_langgraph_agents.md), [Тема 14](topic_14_multi_agent.md)
> **Зависимости:** `crewai`, `crewai-tools`

---

## Теория

### 1. Зачем CrewAI

CrewAI — фреймворк для построения мульти-агентных систем, где каждый агент имеет роль, цель и backstory. Если LangGraph — это "конструктор графов" (низкоуровневый, гибкий), то CrewAI — это "менеджер команды" (высокоуровневый, ролевой).

Сравнение с LangGraph:

| Аспект | LangGraph | CrewAI |
|---|---|---|
| Абстракция | Граф с нодами и рёбрами | Команда с ролями и задачами |
| Уровень | Низкий — ты контролируешь всё | Высокий — фреймворк управляет потоком |
| Гибкость | Максимальная | Ограничена паттернами (sequential, hierarchical) |
| Multi-agent | Строишь сам через subgraphs | Из коробки |
| Кривая обучения | Высокая | Низкая |
| Когда использовать | Сложная логика, нестандартные потоки | Стандартные team-паттерны, быстрый старт |

Ключевая метафора CrewAI: **ты собираешь команду (crew) из специалистов (agents), даёшь им задания (tasks), и они работают вместе**.

### 2. Три основных понятия: Agent, Task, Crew

**Agent** — специалист с ролью, целью и контекстом:

```python
from crewai import Agent

analyzer = Agent(
    role="Литературный аналитик",
    goal="Провести глубокий анализ структуры и аргументации эссе",
    backstory=(
        "Ты — профессор литературоведения с 20-летним опытом. "
        "Специализируешься на академическом письме и критическом анализе."
    ),
    llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)
```

`role` определяет специализацию. `goal` — что агент должен достичь. `backstory` — контекст, который помогает модели "войти в роль". Это не просто system prompt — CrewAI использует `role`, `goal` и `backstory` для формирования промпта, делегирования задач и разрешения конфликтов.

**Task** — конкретное задание с описанием, ожидаемым результатом и назначенным агентом:

```python
from crewai import Task

analysis_task = Task(
    description=(
        "Проанализируй эссе студента:\n\n{essay}\n\n"
        "Определи: тему, тезис, качество аргументации, "
        "использование доказательств, структуру."
    ),
    expected_output="Детальный аналитический отчёт с оценкой каждого аспекта",
    agent=analyzer,
)
```

**Crew** — команда агентов с задачами и процессом выполнения:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[analyzer, scorer, reviewer],
    tasks=[analysis_task, scoring_task, review_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"essay": "AI is transforming society..."})
```

### 3. Processes — как агенты работают вместе

CrewAI поддерживает два процесса:

**Sequential** — задачи выполняются последовательно. Результат каждой задачи передаётся следующей.

```
Analyzer → analysis_report → Scorer → scores → Reviewer → final_assessment
```

```python
crew = Crew(
    agents=[analyzer, scorer, reviewer],
    tasks=[analysis_task, scoring_task, review_task],
    process=Process.sequential,
)
```

**Hierarchical** — менеджер-агент распределяет задачи, контролирует качество и координирует команду:

```python
crew = Crew(
    agents=[analyzer, scorer, reviewer],
    tasks=[analysis_task, scoring_task, review_task],
    process=Process.hierarchical,
    manager_llm="anthropic/claude-sonnet-4-20250514",
)
```

В hierarchical режиме CrewAI создаёт менеджер-агента, который решает: кому делегировать задачу, нужна ли переработка, когда результат достаточно хорош.

| Process | Контроль | Когда использовать |
|---|---|---|
| Sequential | Фиксированный порядок | Линейные пайплайны, предсказуемый workflow |
| Hierarchical | Менеджер решает | Сложные задачи, нужна адаптация, quality gate |

### 4. Tools — инструменты агентов

CrewAI поддерживает инструменты из пакета `crewai-tools` и custom tools:

```python
from crewai import Agent
from crewai_tools import SerperDevTool, FileReadTool, WebsiteSearchTool

researcher = Agent(
    role="Исследователь",
    goal="Найти релевантные источники и данные",
    backstory="Ты — исследователь с навыками поиска академических источников.",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm="anthropic/claude-sonnet-4-20250514",
)
```

Custom tool через декоратор:

```python
from crewai.tools import tool

@tool("Word Counter")
def count_words(text: str) -> str:
    """Подсчитывает количество слов в тексте. Используй для оценки объёма работы."""
    count = len(text.split())
    return f"Текст содержит {count} слов"

@tool("Citation Checker")
def check_citations(text: str) -> str:
    """Проверяет наличие академических цитирований в формате (Author, Year)."""
    import re
    citations = re.findall(r'\(([^)]+,\s*\d{4})\)', text)
    if not citations:
        return "Цитирования не найдены."
    return f"Найдено {len(citations)}: {', '.join(citations)}"

analyzer = Agent(
    role="Аналитик",
    goal="Проанализировать структуру эссе",
    backstory="Эксперт по академическому письму.",
    tools=[count_words, check_citations],
    llm="anthropic/claude-sonnet-4-20250514",
)
```

### 5. Task Configuration — контроль качества

Task имеет множество параметров для контроля:

```python
scoring_task = Task(
    description="Выстави оценку от 1 до 10 на основе анализа.",
    expected_output="JSON с полями: score (int), feedback (str), criteria_scores (dict)",
    agent=scorer,
    context=[analysis_task],
    output_json=ScoringResult,
    human_input=False,
)
```

| Параметр | Описание |
|---|---|
| `description` | Что нужно сделать (поддерживает `{переменные}`) |
| `expected_output` | Описание ожидаемого результата (для LLM) |
| `agent` | Кто выполняет |
| `context` | Список задач, чьи результаты нужны как контекст |
| `output_json` | Pydantic-модель для structured output |
| `output_file` | Сохранить результат в файл |
| `human_input` | Запросить подтверждение у человека |
| `async_execution` | Выполнить асинхронно (параллельно с другими) |

### 6. Delegation — агенты делегируют друг другу

По умолчанию агенты могут делегировать задачи друг другу (`allow_delegation=True`). Это значит, что если Scorer не уверен в анализе, он может попросить Analyzer уточнить.

```python
scorer = Agent(
    role="Оценщик",
    goal="Выставить точную и обоснованную оценку",
    backstory="Строгий, но справедливый эксперт.",
    allow_delegation=True,
    llm="anthropic/claude-sonnet-4-20250514",
)
```

Delegation работает через tool calling: CrewAI автоматически добавляет tool "Delegate work to co-worker" каждому агенту с `allow_delegation=True`.

### 7. Memory — память команды

CrewAI поддерживает несколько типов памяти:

| Тип | Описание | Хранилище |
|---|---|---|
| Short-term | Контекст текущей сессии | In-memory |
| Long-term | Опыт прошлых запусков | SQLite |
| Entity | Информация об entities (люди, проекты) | In-memory |

```python
crew = Crew(
    agents=[analyzer, scorer],
    tasks=[analysis_task, scoring_task],
    process=Process.sequential,
    memory=True,
    verbose=True,
)
```

С `memory=True` агенты "помнят" предыдущие оценки и результаты. При повторном запуске с похожим эссе агент может использовать прошлый опыт для калибровки.

### 8. Structured Output — Pydantic models

CrewAI поддерживает typed output через Pydantic:

```python
from pydantic import BaseModel, Field
from crewai import Task

class AssessmentOutput(BaseModel):
    score: int = Field(ge=1, le=10, description="Оценка от 1 до 10")
    feedback: str = Field(description="Развёрнутый фидбек")
    strengths: list[str] = Field(description="Сильные стороны")
    weaknesses: list[str] = Field(description="Слабые стороны")
    recommendations: list[str] = Field(description="Рекомендации")

review_task = Task(
    description="Составь финальный отчёт по оценке эссе.",
    expected_output="Структурированная оценка с баллом, фидбеком и рекомендациями",
    agent=reviewer,
    output_pydantic=AssessmentOutput,
)
```

`result.pydantic` вернёт `AssessmentOutput` с валидированными данными.

---

## Справочник API

### Agent

```python
from crewai import Agent

agent = Agent(
    role: str,
    goal: str,
    backstory: str,
    llm: str | Any = None,
    tools: list[BaseTool] = [],
    verbose: bool = False,
    allow_delegation: bool = True,
    max_iter: int = 25,
    max_rpm: int | None = None,
    memory: bool = True,
    step_callback: Callable | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `role` | `str` | Роль агента (используется в промпте и делегировании) |
| `goal` | `str` | Цель агента |
| `backstory` | `str` | Контекст и экспертиза |
| `llm` | `str` | Модель: `"openai/gpt-4o"`, `"anthropic/claude-sonnet-4-20250514"` |
| `tools` | `list` | Доступные инструменты |
| `allow_delegation` | `bool` | Может ли делегировать задачи другим агентам |
| `max_iter` | `int` | Максимум итераций (ReAct loop) |

### Task

```python
from crewai import Task

task = Task(
    description: str,
    expected_output: str,
    agent: Agent | None = None,
    context: list[Task] = [],
    tools: list[BaseTool] = [],
    output_json: type[BaseModel] | None = None,
    output_pydantic: type[BaseModel] | None = None,
    output_file: str | None = None,
    human_input: bool = False,
    async_execution: bool = False,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `description` | `str` | Описание задачи (поддерживает `{переменные}`) |
| `expected_output` | `str` | Что ожидается на выходе |
| `context` | `list[Task]` | Задачи, чьи результаты нужны как вход |
| `output_pydantic` | `type` | Pydantic-модель для structured output |
| `human_input` | `bool` | Запросить подтверждение человека |
| `async_execution` | `bool` | Параллельное выполнение |

### Crew

```python
from crewai import Crew, Process

crew = Crew(
    agents: list[Agent],
    tasks: list[Task],
    process: Process = Process.sequential,
    verbose: bool = False,
    memory: bool = False,
    manager_llm: str | None = None,
    max_rpm: int | None = None,
    language: str = "en",
    output_log_file: str | None = None,
)

result = crew.kickoff(inputs: dict = {})
```

| Параметр | Тип | Описание |
|---|---|---|
| `process` | `Process` | `Process.sequential` или `Process.hierarchical` |
| `memory` | `bool` | Включить память команды |
| `manager_llm` | `str` | Модель для менеджера (только hierarchical) |
| `language` | `str` | Язык общения агентов |

### CrewOutput

```python
result = crew.kickoff(inputs={...})

result.raw           # str — сырой текст
result.pydantic      # BaseModel | None — если output_pydantic
result.json_dict     # dict | None — если output_json
result.tasks_output  # list[TaskOutput] — результаты каждой задачи
result.token_usage   # dict — использование токенов
```

### @tool декоратор

```python
from crewai.tools import tool

@tool("Tool Name")
def my_tool(param1: str, param2: int = 10) -> str:
    """Описание инструмента для LLM."""
    return f"Result: {param1}, {param2}"
```

---

## Практика

### Пример 1: Базовая команда — анализ + оценка

Два агента: аналитик разбирает эссе, оценщик выставляет балл.

```python
from crewai import Agent, Task, Crew, Process

analyzer = Agent(
    role="Литературный аналитик",
    goal="Провести глубокий анализ структуры и содержания эссе",
    backstory=(
        "Профессор литературоведения с 20-летним опытом. "
        "Специализация — академическое письмо и критический анализ."
    ),
    llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)

scorer = Agent(
    role="Оценщик",
    goal="Выставить объективную оценку от 1 до 10 на основе анализа",
    backstory=(
        "Опытный преподаватель, известный справедливостью и точностью оценок. "
        "Всегда обосновывает баллы конкретными примерами из работы."
    ),
    llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)

analysis_task = Task(
    description=(
        "Проанализируй эссе:\n\n{essay}\n\n"
        "Определи: тему, тезис, качество аргументации, "
        "использование доказательств, структуру."
    ),
    expected_output=(
        "Аналитический отчёт: тема, тезис, качество аргументации (сильная/средняя/слабая), "
        "список доказательств, структурные замечания."
    ),
    agent=analyzer,
)

scoring_task = Task(
    description="На основе анализа выстави оценку от 1 до 10. Обоснуй каждый балл.",
    expected_output="Оценка (число), обоснование по каждому критерию, итоговый фидбек.",
    agent=scorer,
    context=[analysis_task],
)

crew = Crew(
    agents=[analyzer, scorer],
    tasks=[analysis_task, scoring_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={
    "essay": (
        "The rapid advancement of AI poses fundamental questions about the nature "
        "of work. Brynjolfsson and McAfee (2014) argue for a 'race against the "
        "machine', while Autor (2015) suggests complementarity."
    )
})

print("=== Final Result ===")
print(result.raw)
print(f"\nTokens: {result.token_usage}")
```

### Пример 2: Три агента с tools

Аналитик с инструментами + оценщик + рецензент.

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import re

@tool("Word Counter")
def count_words(text: str) -> str:
    """Подсчитывает слова в тексте."""
    return f"Количество слов: {len(text.split())}"

@tool("Citation Checker")
def check_citations(text: str) -> str:
    """Находит академические цитирования (Author, Year)."""
    citations = re.findall(r'\(([^)]+,\s*\d{4})\)', text)
    if not citations:
        return "Цитирования не найдены."
    return f"Найдено {len(citations)}: {', '.join(citations)}"

analyzer = Agent(
    role="Структурный аналитик",
    goal="Объективно проанализировать структуру и содержание эссе",
    backstory="Эксперт по академическому письму. Использует инструменты для объективного анализа.",
    tools=[count_words, check_citations],
    llm="anthropic/claude-sonnet-4-20250514",
)

scorer = Agent(
    role="Оценщик",
    goal="Выставить точную оценку на основе анализа",
    backstory="Строгий, но справедливый эксперт.",
    llm="anthropic/claude-sonnet-4-20250514",
)

reviewer = Agent(
    role="Рецензент",
    goal="Проверить согласованность анализа и оценки, дать финальные рекомендации",
    backstory="Главный редактор академического журнала. Ищет несоответствия.",
    llm="anthropic/claude-sonnet-4-20250514",
)

task_analyze = Task(
    description="Проанализируй эссе:\n\n{essay}\n\nИспользуй инструменты.",
    expected_output="Структурный анализ: слова, цитирования, качество аргументов.",
    agent=analyzer,
)

task_score = Task(
    description="Выстави оценку 1-10 на основе анализа.",
    expected_output="Оценка и обоснование.",
    agent=scorer,
    context=[task_analyze],
)

task_review = Task(
    description="Проверь анализ и оценку. Согласованы ли они? Есть ли пропуски?",
    expected_output="Финальная рецензия: подтверждение оценки или рекомендация пересмотра.",
    agent=reviewer,
    context=[task_analyze, task_score],
)

crew = Crew(
    agents=[analyzer, scorer, reviewer],
    tasks=[task_analyze, task_score, task_review],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={
    "essay": "The intersection of AI and labor economics presents a nuanced challenge."
})
print(result.raw)
```

### Пример 3: Hierarchical процесс

Менеджер-агент координирует команду и контролирует качество.

```python
from crewai import Agent, Task, Crew, Process

analyzer = Agent(
    role="Аналитик",
    goal="Анализ эссе",
    backstory="Эксперт по структуре текста.",
    llm="anthropic/claude-haiku-3.5",
)

scorer = Agent(
    role="Оценщик",
    goal="Выставление баллов",
    backstory="Справедливый эксперт.",
    llm="anthropic/claude-haiku-3.5",
)

reviewer = Agent(
    role="Рецензент",
    goal="Проверка качества",
    backstory="Контроль согласованности.",
    llm="anthropic/claude-haiku-3.5",
)

tasks = [
    Task(
        description="Проанализируй: {essay}",
        expected_output="Анализ структуры и аргументации",
        agent=analyzer,
    ),
    Task(
        description="Выстави оценку на основе анализа",
        expected_output="Оценка 1-10 с обоснованием",
        agent=scorer,
    ),
    Task(
        description="Проверь и утверди финальную оценку",
        expected_output="Утверждённый результат",
        agent=reviewer,
    ),
]

crew = Crew(
    agents=[analyzer, scorer, reviewer],
    tasks=tasks,
    process=Process.hierarchical,
    manager_llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)

result = crew.kickoff(inputs={"essay": "AI is transforming..."})
print(result.raw)
```

### Пример 4: Structured Output

Получение типизированного результата через Pydantic.

```python
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

class AssessmentOutput(BaseModel):
    score: int = Field(ge=1, le=10, description="Итоговая оценка")
    grade: str = Field(description="Буквенная оценка: A/B/C/D/F")
    feedback: str = Field(description="Развёрнутый фидбек")
    strengths: list[str] = Field(description="Сильные стороны")
    weaknesses: list[str] = Field(description="Слабые стороны")
    recommendations: list[str] = Field(description="Рекомендации по улучшению")

assessor = Agent(
    role="Академический оценщик",
    goal="Предоставить всестороннюю и структурированную оценку",
    backstory="Профессор с опытом оценки тысяч студенческих работ.",
    llm="anthropic/claude-sonnet-4-20250514",
)

task = Task(
    description=(
        "Оцени эссе студента:\n\n{essay}\n\n"
        "Рубрика: Тезис (0-3), Доказательства (0-3), Структура (0-2), Язык (0-2)."
    ),
    expected_output="Полная структурированная оценка.",
    agent=assessor,
    output_pydantic=AssessmentOutput,
)

crew = Crew(agents=[assessor], tasks=[task], process=Process.sequential)

result = crew.kickoff(inputs={
    "essay": "The rapid advancement of AI poses fundamental questions..."
})

assessment = result.pydantic
print(f"Score: {assessment.score}/10 ({assessment.grade})")
print(f"Strengths: {assessment.strengths}")
print(f"Weaknesses: {assessment.weaknesses}")
print(f"Recommendations: {assessment.recommendations}")
```

### Пример 5: Параллельные задачи

Async execution для задач, которые можно выполнить параллельно.

```python
from crewai import Agent, Task, Crew, Process

structure_analyst = Agent(
    role="Структурный аналитик",
    goal="Оценить структуру эссе",
    backstory="Специалист по академической структуре.",
    llm="anthropic/claude-haiku-3.5",
)

content_analyst = Agent(
    role="Контент-аналитик",
    goal="Оценить содержание и аргументацию",
    backstory="Специалист по критическому мышлению.",
    llm="anthropic/claude-haiku-3.5",
)

synthesizer = Agent(
    role="Синтезатор",
    goal="Объединить два анализа в итоговую оценку",
    backstory="Главный оценщик, принимающий финальное решение.",
    llm="anthropic/claude-sonnet-4-20250514",
)

task_structure = Task(
    description="Оцени структуру:\n{essay}",
    expected_output="Оценка структуры: введение, основная часть, заключение, связность.",
    agent=structure_analyst,
    async_execution=True,
)

task_content = Task(
    description="Оцени содержание:\n{essay}",
    expected_output="Оценка содержания: тезис, аргументы, доказательства, глубина.",
    agent=content_analyst,
    async_execution=True,
)

task_synthesize = Task(
    description="Объедини анализ структуры и содержания в итоговую оценку 1-10.",
    expected_output="Итоговая оценка с обоснованием.",
    agent=synthesizer,
    context=[task_structure, task_content],
)

crew = Crew(
    agents=[structure_analyst, content_analyst, synthesizer],
    tasks=[task_structure, task_content, task_synthesize],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={
    "essay": "The rapid advancement of AI poses fundamental questions..."
})
print(result.raw)
```

### Пример 6: Batch-оценка нескольких эссе

Оценка набора работ одной командой с memory.

```python
from crewai import Agent, Task, Crew, Process

assessor = Agent(
    role="Оценщик эссе",
    goal="Выставить справедливую и калиброванную оценку",
    backstory=(
        "Опытный преподаватель. При оценке учитывает контекст: "
        "если оценил предыдущую работу на 8, следующая на похожую тему "
        "должна оцениваться последовательно."
    ),
    llm="anthropic/claude-sonnet-4-20250514",
)

essays = [
    "AI is transforming society in profound ways. Studies show 47% automation risk.",
    "Machine learning uses data.",
    "The ethical implications of AI extend beyond accuracy metrics. Patient autonomy, "
    "informed consent, and equitable access are critical dimensions.",
]

results = []
for i, essay in enumerate(essays):
    task = Task(
        description=f"Оцени эссе #{i+1}:\n\n{essay}",
        expected_output="Оценка 1-10 и краткий фидбек (2-3 предложения).",
        agent=assessor,
    )

    crew = Crew(
        agents=[assessor],
        tasks=[task],
        process=Process.sequential,
        memory=True,
    )

    result = crew.kickoff()
    results.append(result.raw)
    print(f"Essay #{i+1}: {result.raw[:100]}...")

print("\n=== All Results ===")
for i, r in enumerate(results):
    print(f"\n--- Essay #{i+1} ---\n{r}")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | Sequential crew: analyzer → scorer |
| 2 | Tools + три агента + context |
| 3 | Hierarchical процесс с менеджером |
| 4 | Structured output (Pydantic) |
| 5 | Параллельные задачи (async_execution) |
| 6 | Memory для калибровки batch-оценки |

---

## Чеклист самопроверки

- [ ] В чём разница между CrewAI и LangGraph? Когда выбрать CrewAI?
- [ ] Что такое `role`, `goal`, `backstory` у Agent? Как они влияют на поведение?
- [ ] Чем `Process.sequential` отличается от `Process.hierarchical`?
- [ ] Что такое delegation? Когда агент делегирует задачу другому?
- [ ] Как работает `context` у Task? Как передаётся результат между задачами?
- [ ] Как создать custom tool в CrewAI? Что нужно для хорошего описания?
- [ ] Как получить structured output (Pydantic) из CrewAI?
- [ ] Какие типы memory поддерживает CrewAI? Зачем нужна long-term memory?
- [ ] Как организовать параллельное выполнение задач?
- [ ] Чем `output_pydantic` отличается от `output_json` в Task?

---

## Частые ошибки

### 1. Одинаковые role и backstory

```python
agent1 = Agent(role="AI Assistant", goal="Help user", backstory="You are helpful.")
agent2 = Agent(role="AI Assistant", goal="Help user", backstory="You are helpful.")
```

```python
analyzer = Agent(
    role="Структурный аналитик",
    goal="Проанализировать структуру и аргументацию эссе",
    backstory="Профессор с 20-летним опытом в академическом письме.",
)
scorer = Agent(
    role="Количественный оценщик",
    goal="Выставить числовые баллы по каждому критерию",
    backstory="Математик, ценящий точность и обоснованность каждого балла.",
)
```

Одинаковые агенты — бессмысленная команда. CrewAI использует `role` и `backstory` для формирования промпта и делегирования. Чем уникальнее роли, тем лучше качество.

### 2. Забыли context между задачами

```python
task_analyze = Task(description="Проанализируй {essay}", agent=analyzer, ...)
task_score = Task(description="Выстави оценку", agent=scorer, ...)
```

```python
task_analyze = Task(description="Проанализируй {essay}", agent=analyzer, ...)
task_score = Task(description="Выстави оценку", agent=scorer, context=[task_analyze], ...)
```

Без `context` оценщик не видит результат анализа. В sequential процессе предыдущий результат передаётся автоматически, но `context` делает зависимость явной и работает в любом процессе.

### 3. Hierarchical без manager_llm

```python
crew = Crew(
    agents=[a1, a2],
    tasks=[t1, t2],
    process=Process.hierarchical,
)
```

```python
crew = Crew(
    agents=[a1, a2],
    tasks=[t1, t2],
    process=Process.hierarchical,
    manager_llm="anthropic/claude-sonnet-4-20250514",
)
```

В hierarchical процессе нужен менеджер. Без `manager_llm` CrewAI упадёт с ошибкой. Для менеджера рекомендуется сильная модель — он принимает решения о делегировании и качестве.

### 4. Слишком много итераций

```python
agent = Agent(
    role="Оценщик",
    goal="...",
    backstory="...",
    max_iter=100,
    llm="anthropic/claude-sonnet-4-20250514",
)
```

```python
agent = Agent(
    role="Оценщик",
    goal="...",
    backstory="...",
    max_iter=15,
    llm="anthropic/claude-sonnet-4-20250514",
)
```

`max_iter=100` может привести к зацикливанию и огромным расходам. Для большинства задач 10-20 итераций достаточно. Если агент не справляется за 15 итераций — проблема в промпте или tools, не в количестве попыток.

### 5. Verbose в production

```python
crew = Crew(agents=agents, tasks=tasks, verbose=True)
```

```python
crew = Crew(agents=agents, tasks=tasks, verbose=False)
```

`verbose=True` выводит все промежуточные сообщения, tool calls и результаты в stdout. Полезно для debug, но в production засоряет логи и замедляет работу. Используй structured logging вместо verbose.

---

## Что читать дальше

- [CrewAI Documentation](https://docs.crewai.com/) — полная документация
- [CrewAI Agents](https://docs.crewai.com/core-concepts/agents/) — конфигурация агентов
- [CrewAI Tasks](https://docs.crewai.com/core-concepts/tasks/) — задачи и workflow
- [CrewAI Crews](https://docs.crewai.com/core-concepts/crews/) — команды и процессы
- [CrewAI Tools](https://docs.crewai.com/core-concepts/tools/) — встроенные и custom tools
- [CrewAI Memory](https://docs.crewai.com/core-concepts/memory/) — типы памяти
- [CrewAI Examples](https://docs.crewai.com/examples/) — готовые примеры
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI) — исходный код

**Предыдущая тема:** [Тема 26: LiteLLM](topic_26_litellm.md) — единый API для 100+ LLM-провайдеров.

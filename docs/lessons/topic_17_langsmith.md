# Тема 17: LangSmith — платформа LangChain для LLM ops

> **Пререквизиты:** [Тема 8: Observability](topic_08_observability.md), рекомендуется [Тема 16: Langfuse](topic_16_langfuse.md)
> **Что добавляем в проект:** `app/api/v1/langsmith.py`, `app/services/langsmith_service.py`, `app/schemas/langsmith.py`
> **Зависимости:** `langsmith`, `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. LangSmith — что это и зачем

LangSmith — это managed-платформа от LangChain Inc. для полного цикла разработки LLM-приложений: от прототипа до production. В отличие от инструментов мониторинга общего назначения (Datadog, Prometheus), LangSmith спроектирован вокруг специфики LLM: промпты, токены, цепочки вызовов, недетерминированные ответы и оценка качества.

LangSmith решает четыре фундаментальные задачи:

**Tracing** — запись каждого вызова LLM с полным контекстом: входной промпт, выходной ответ, количество токенов, латентность, модель, параметры сэмплирования. Трейсы вложенные: один assessment может включать preprocessing → LLM call → parsing → postprocessing, и каждый шаг записывается как child run.

**Hub** — централизованный реестр промптов с версионированием. Вместо хардкода шаблонов в коде, промпты хранятся в Hub и загружаются по имени. Это позволяет менять промпты без деплоя кода и вести полную историю изменений.

**Evaluation** — систематическая оценка качества LLM-ответов. Datasets с эталонными данными, автоматические evaluators (built-in и custom), LLM-as-judge, comparison experiments для A/B-тестирования промптов.

**Monitoring** — production-мониторинг: dashboards с метриками латентности, стоимости, качества; автоматические alerts при деградации; annotation queues для ручной проверки.

#### Pricing и доступность

| Tier | Traces/месяц | Datasets | Hub | Цена |
|------|-------------|----------|-----|------|
| Developer (free) | 5,000 | 3 | Public only | $0 |
| Plus | 50,000 | Unlimited | Private + Public | $39/user/month |
| Enterprise | Unlimited | Unlimited | Private + Org | Custom |

Критическое отличие от Langfuse: LangSmith — **managed-only**. Нет self-hosted варианта. Все данные хранятся на серверах LangChain Inc. Для организаций с strict data residency requirements это может быть blockerом.

#### Когда LangSmith, когда Langfuse

| Критерий | LangSmith | Langfuse |
|----------|-----------|----------|
| Интеграция с LangChain | Нативная, zero-config | Через callback handler |
| Self-hosted | Нет | Да (Docker, Kubernetes) |
| Prompt management | Hub — полноценный registry | Prompt management в UI |
| Evaluation API | `evaluate()` — мощный, с comparison | Через scores + datasets |
| Pricing | $39+/user/month | Open-source (бесплатно при self-host) |
| Provider lock-in | Привязка к экосистеме LangChain | Provider-agnostic |
| Data residency | Только US/EU cloud | Полный контроль при self-host |
| OpenTelemetry | Нет | Да |

Рекомендация для проекта: если вы полностью на LangChain и не требуется self-hosting — LangSmith даёт наилучший developer experience. Если нужен open-source, self-hosted или multi-framework — Langfuse. Оба инструмента можно использовать одновременно: LangSmith для evaluation и prompt management, Langfuse для production tracing.

### 2. Трейсинг в LangSmith

Трейсинг — основа всей платформы. Каждый вызов LLM автоматически записывается с полным контекстом, создавая auditable log всех операций.

#### Автоматический трейсинг

Для LangChain-приложений трейсинг включается двумя переменными окружения:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."
os.environ["LANGCHAIN_PROJECT"] = "ai-assessment"
```

После этого **каждый** вызов `chain.ainvoke()`, `llm.ainvoke()`, `prompt.invoke()` автоматически записывается в LangSmith. Не нужно менять код, добавлять callback handlers или декораторы — всё работает через глобальный трейсер.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assessor."),
    ("human", "Assess this work: {student_work}"),
])

chain = prompt | llm
result = await chain.ainvoke({"student_work": "Essay text..."})
```

Этот код при `LANGCHAIN_TRACING_V2=true` автоматически создаст trace в LangSmith с двумя child runs: один для prompt template, один для LLM call. В UI будут видны rendered промпт, ответ модели, количество токенов, латентность.

#### Модель данных

LangSmith использует единую абстракцию **Run** для всех операций:

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | UUID | Уникальный идентификатор run |
| `name` | str | Имя операции (`ChatAnthropic`, `ChatPromptTemplate`, custom name) |
| `run_type` | str | `"chain"`, `"llm"`, `"tool"`, `"prompt"`, `"retriever"` |
| `parent_run_id` | UUID | Ссылка на родительский run (для вложенности) |
| `inputs` | dict | Входные данные |
| `outputs` | dict | Выходные данные |
| `error` | str | Текст ошибки (если run провалился) |
| `start_time` | datetime | Время начала |
| `end_time` | datetime | Время завершения |
| `extra` | dict | Дополнительные данные: модель, токены, стоимость |
| `tags` | list[str] | Теги для фильтрации |
| `metadata` | dict | Произвольные метаданные |

Runs организованы в деревья: корневой run (обычно chain) содержит child runs (prompt, LLM, parser). Это позволяет видеть breakdown каждого шага:

```
Run: "assessment_chain" (chain, 4.2s)
├── Run: "ChatPromptTemplate" (prompt, 0.001s)
│   └── Input: {student_work: "..."} → Output: [SystemMessage, HumanMessage]
├── Run: "ChatAnthropic" (llm, 4.1s)
│   └── Input: messages → Output: AIMessage, tokens: 1250+800
└── Run: "PydanticOutputParser" (parser, 0.002s)
    └── Input: AIMessage → Output: AssessmentResponse
```

#### @traceable — трейсинг обычных функций

Для функций, которые не являются LangChain Runnables, используется декоратор `@traceable`:

```python
from langsmith import traceable

@traceable(name="preprocess_student_work", run_type="chain")
def preprocess_student_work(text: str) -> dict:
    word_count = len(text.split())
    paragraph_count = text.count("\n\n") + 1
    return {
        "text": text,
        "word_count": word_count,
        "paragraph_count": paragraph_count,
    }

@traceable(name="format_assessment_result", run_type="chain")
def format_assessment_result(result: dict) -> str:
    score = result["overall_score"]
    summary = result["summary"]
    return f"Score: {score}/100\n\n{summary}"
```

`@traceable` работает с любыми Python-функциями — sync и async. Вложенные вызовы `@traceable` функций внутри другой `@traceable` функции автоматически создают parent-child relationship:

```python
@traceable(name="full_assessment_pipeline")
async def full_pipeline(student_work: str, rubric: dict) -> dict:
    preprocessed = preprocess_student_work(student_work)
    result = await run_llm_assessment(preprocessed, rubric)
    formatted = format_assessment_result(result)
    return {"result": result, "formatted": formatted}
```

В LangSmith UI это отобразится как дерево: `full_assessment_pipeline` → `preprocess_student_work` + `run_llm_assessment` + `format_assessment_result`.

#### Теги и метаданные

Теги и метаданные — механизм организации и фильтрации traces. Теги — плоский список строк для быстрого фильтра. Метаданные — произвольный dict для хранения контекста.

```python
result = await chain.ainvoke(
    {"student_work": "..."},
    config={
        "tags": ["assessment", "v2", "essay"],
        "metadata": {
            "user_id": "student_42",
            "rubric_version": "2.1",
            "deployment": "canary",
        },
        "run_name": "essay_assessment_v2",
    },
)
```

В LangSmith UI можно фильтровать runs по тегам (`tag:assessment AND tag:v2`), метаданным (`metadata.rubric_version = "2.1"`), времени, модели, статусу (success/error), латентности (> 5s).

#### wrap_openai — трейсинг для raw OpenAI SDK

Если часть кода использует OpenAI SDK напрямую (без LangChain), `wrap_openai` добавляет трейсинг:

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Assess this essay..."}],
)
```

Каждый вызов `client.chat.completions.create()` будет записан в LangSmith с входными messages, ответом, токенами и латентностью. Это полезно для migration: можно добавить трейсинг к существующему коду без перехода на LangChain.

### 3. LangChain Hub — реестр промптов

Hub — это marketplace промптов с версионированием, который решает две проблемы: (1) промпты hardcoded в коде — изменение требует деплоя, (2) нет истории изменений промптов — непонятно, какая версия дала регрессию.

#### Загрузка промпта из Hub

```python
from langchain import hub

prompt = hub.pull("langchain-ai/assessment-rubric-prompt")

chain = prompt | llm
result = await chain.ainvoke({"student_work": "...", "rubric": "..."})
```

`hub.pull()` загружает `ChatPromptTemplate` (или другой тип промпта) из реестра. По умолчанию загружается последняя версия. Для reproducibility можно указать конкретный commit hash:

```python
prompt = hub.pull("my-org/assessment-prompt:a1b2c3d4")
```

Это гарантирует, что production-код всегда использует проверенную версию промпта, даже если кто-то опубликовал новую.

#### Публикация промпта

```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert essay assessor. Evaluate student work against the provided rubric. Be specific in your feedback."),
    ("human", "Rubric:\n{rubric}\n\nStudent work:\n{student_work}"),
])

hub.push("my-org/essay-assessment-v2", prompt)
```

Каждый `hub.push()` создаёт новую версию (commit). Старые версии остаются доступны по hash. Это аналог git для промптов: можно откатиться к любой предыдущей версии.

#### Public vs Private

| Видимость | Кто видит | Кто может pull | Use case |
|-----------|-----------|----------------|----------|
| Public | Все | Все | Шаблоны, примеры, community промпты |
| Private | Только owner/org | Только owner/org | Бизнес-промпты с proprietary logic |

По умолчанию промпты private. Для public нужно явно указать при создании в UI или передать флаг в API.

#### Workflow с Hub

Типичный production workflow:

1. Разработчик создаёт/редактирует промпт в LangSmith UI (или через `hub.push()`)
2. Запускает evaluation на golden dataset с новой версией промпта
3. Если метрики улучшились — обновляет commit hash в конфигурации приложения
4. Production-код подтягивает новую версию при следующем деплое

Этот workflow отделяет код (логика, цепочки, парсинг) от контента (промпты, инструкции, рубрики). Промптеры могут итерировать над текстом без участия разработчиков.

### 4. Datasets — тестовые наборы данных

Datasets в LangSmith — это структурированные коллекции примеров с входными данными и ожидаемыми выходами. Они являются фундаментом evaluation-процесса.

#### Структура dataset

Каждый dataset состоит из **examples**. Каждый example содержит:

| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| `inputs` | dict | Да | Входные данные для target function |
| `outputs` | dict | Нет | Ожидаемые выходные данные (reference) |
| `metadata` | dict | Нет | Дополнительный контекст |

Пример для assessment-системы:

```python
example = {
    "inputs": {
        "student_work": "Climate change is a major threat...",
        "rubric_id": "essay_default",
    },
    "outputs": {
        "overall_score": 65,
        "summary": "The essay presents a clear thesis but lacks supporting evidence...",
    },
    "metadata": {
        "quality_level": "medium",
        "graded_by": "expert_reviewer_1",
    },
}
```

#### Создание dataset через API

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="assessment-golden-v1",
    description="Golden dataset for essay assessment evaluation",
)

client.create_examples(
    inputs=[
        {"student_work": "Essay 1 text...", "rubric_id": "essay_default"},
        {"student_work": "Essay 2 text...", "rubric_id": "essay_default"},
        {"student_work": "Essay 3 text...", "rubric_id": "essay_default"},
    ],
    outputs=[
        {"overall_score": 85, "summary": "Strong essay with clear thesis..."},
        {"overall_score": 42, "summary": "Weak essay, lacks structure..."},
        {"overall_score": 71, "summary": "Decent work, needs more evidence..."},
    ],
    dataset_name="assessment-golden-v1",
)
```

#### Splits — разделение dataset

LangSmith поддерживает splits для организации данных внутри dataset:

```python
client.create_examples(
    inputs=[...],
    outputs=[...],
    dataset_name="assessment-golden-v1",
    splits=["test"],
)

client.create_examples(
    inputs=[...],
    outputs=[...],
    dataset_name="assessment-golden-v1",
    splits=["train"],
)
```

Splits полезны для:
- **test** — основной набор для evaluation (не менять после создания)
- **train** — примеры для few-shot или fine-tuning
- **validation** — для hyperparameter tuning (temperature, max_tokens)

#### Создание из traces

Самый практичный способ построения dataset — конвертация production traces. В LangSmith UI можно:

1. Отфильтровать traces по тегу или метаданным
2. Выбрать "хорошие" traces (высокий score, positive feedback)
3. Нажать "Add to Dataset" → выбрать существующий или создать новый
4. LangSmith автоматически извлечёт inputs и outputs из trace

Через API:

```python
runs = client.list_runs(
    project_name="ai-assessment",
    filter='has(tags, "verified") and status = "success"',
    limit=50,
)

inputs = []
outputs = []
for run in runs:
    inputs.append(run.inputs)
    outputs.append(run.outputs)

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_name="assessment-from-traces",
)
```

Этот подход создаёт flywheel: production → traces → verified examples → dataset → evaluation → improved prompts → better production.

#### Загрузка из CSV/JSONL

```python
import json

with open("golden_data.jsonl") as f:
    examples = [json.loads(line) for line in f]

client.create_examples(
    inputs=[e["input"] for e in examples],
    outputs=[e["expected_output"] for e in examples],
    dataset_name="assessment-golden-v1",
)
```

### 5. Evaluation — оценка качества

Evaluation в LangSmith — это систематический прогон target function на dataset с автоматической оценкой результатов. Функция `evaluate()` — центральный API для этого процесса.

#### Базовая структура evaluation

```python
from langsmith import evaluate

def assess_student_work(inputs: dict) -> dict:
    chain = build_assessment_chain()
    result = chain.invoke(inputs)
    return {"overall_score": result.overall_score, "summary": result.summary}

results = evaluate(
    assess_student_work,
    data="assessment-golden-v1",
    evaluators=[exact_match_evaluator],
    experiment_prefix="assessment-v2",
)
```

`evaluate()` делает следующее:
1. Загружает все examples из dataset
2. Для каждого example вызывает target function с `example.inputs`
3. Для каждого результата вызывает все evaluators с парой (predicted, expected)
4. Записывает results как experiment в LangSmith

#### Target function

Target function — это функция, которую вы хотите оценить. Она принимает `inputs: dict` и возвращает `dict`:

```python
def target(inputs: dict) -> dict:
    student_work = inputs["student_work"]
    rubric_id = inputs["rubric_id"]
    result = run_assessment(student_work, rubric_id)
    return {
        "overall_score": result.overall_score,
        "summary": result.summary,
        "criterion_scores": [
            {"name": cs.criterion_name, "score": cs.score}
            for cs in result.criterion_scores
        ],
    }
```

Target function может быть sync или async. Для async используется та же сигнатура:

```python
async def target(inputs: dict) -> dict:
    result = await chain.ainvoke(inputs)
    return {"overall_score": result.overall_score}
```

#### Built-in evaluators

LangSmith предоставляет набор готовых evaluators:

| Evaluator | Что измеряет | Как работает |
|-----------|-------------|-------------|
| `exact_match` | Точное совпадение predicted и expected | `predicted == expected` → 1.0 или 0.0 |
| `embedding_distance` | Семантическая близость текстов | cosine similarity между embeddings |
| `json_edit_distance` | Структурная близость JSON-ответов | Normalized edit distance между JSON strings |
| `json_schema_match` | Соответствие JSON schema | Валидация predicted против expected schema |

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

results = evaluate(
    target,
    data="assessment-golden-v1",
    evaluators=[
        LangChainStringEvaluator("embedding_distance"),
    ],
    experiment_prefix="assessment-v2-embedding",
)
```

#### Custom evaluators

Custom evaluator — функция, которая принимает `run` и `example` и возвращает `EvaluationResult`:

```python
from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult

def score_accuracy_evaluator(run: Run, example: Example) -> EvaluationResult:
    predicted_score = run.outputs["overall_score"]
    expected_score = example.outputs["overall_score"]
    diff = abs(predicted_score - expected_score)

    if diff <= 5:
        score = 1.0
    elif diff <= 10:
        score = 0.7
    elif diff <= 20:
        score = 0.3
    else:
        score = 0.0

    return EvaluationResult(
        key="score_accuracy",
        score=score,
        comment=f"Predicted: {predicted_score}, Expected: {expected_score}, Diff: {diff}",
    )
```

```python
results = evaluate(
    target,
    data="assessment-golden-v1",
    evaluators=[score_accuracy_evaluator],
    experiment_prefix="assessment-v2-accuracy",
)
```

#### LLM-as-Judge evaluator

Для оценки качества текстовых ответов (summary, feedback) используется LLM-as-judge — другая LLM оценивает ответ:

```python
from langsmith.evaluation import LangChainStringEvaluator

correctness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "correctness": "Is the assessment accurate and well-reasoned? "
            "Does it correctly identify strengths and weaknesses?",
        },
    },
)

helpfulness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": "Is the feedback actionable and specific? "
            "Does it help the student improve?",
        },
    },
)

results = evaluate(
    target,
    data="assessment-golden-v1",
    evaluators=[correctness_evaluator, helpfulness_evaluator],
    experiment_prefix="assessment-v2-llm-judge",
)
```

LLM-as-judge добавляет стоимость: каждый evaluation example требует дополнительного LLM-вызова. Для dataset из 50 примеров с двумя evaluators — это 100 дополнительных вызовов. Планируйте бюджет.

#### Comparison experiments

Самый мощный use case evaluation — A/B-сравнение двух версий на одном dataset:

```python
def target_v1(inputs: dict) -> dict:
    chain = build_chain_v1()
    result = chain.invoke(inputs)
    return {"overall_score": result.overall_score, "summary": result.summary}

def target_v2(inputs: dict) -> dict:
    chain = build_chain_v2()
    result = chain.invoke(inputs)
    return {"overall_score": result.overall_score, "summary": result.summary}

results_v1 = evaluate(
    target_v1,
    data="assessment-golden-v1",
    evaluators=[score_accuracy_evaluator],
    experiment_prefix="assessment-v1",
)

results_v2 = evaluate(
    target_v2,
    data="assessment-golden-v1",
    evaluators=[score_accuracy_evaluator],
    experiment_prefix="assessment-v2",
)
```

В LangSmith UI оба эксперимента отображаются рядом на одном dataset, с side-by-side сравнением метрик. Можно видеть: на каких примерах v2 лучше, на каких хуже, общие aggregate scores.

#### Summary evaluators

Summary evaluators работают на уровне всего dataset, а не отдельных примеров. Они принимают все runs и examples и возвращают агрегированную метрику:

```python
from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult

def mean_score_diff(runs: list[Run], examples: list[Example]) -> EvaluationResult:
    diffs = []
    for run, example in zip(runs, examples):
        predicted = run.outputs["overall_score"]
        expected = example.outputs["overall_score"]
        diffs.append(abs(predicted - expected))

    mean_diff = sum(diffs) / len(diffs)
    return EvaluationResult(
        key="mean_absolute_error",
        score=mean_diff,
        comment=f"MAE across {len(diffs)} examples",
    )
```

```python
results = evaluate(
    target,
    data="assessment-golden-v1",
    evaluators=[score_accuracy_evaluator],
    summary_evaluators=[mean_score_diff],
    experiment_prefix="assessment-v2-with-mae",
)
```

#### Параметры evaluate()

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `target` | Callable | — | Функция для оценки |
| `data` | str | — | Имя dataset или dataset ID |
| `evaluators` | list | `[]` | Список per-example evaluators |
| `summary_evaluators` | list | `[]` | Список aggregate evaluators |
| `experiment_prefix` | str | `""` | Префикс имени эксперимента |
| `max_concurrency` | int | `5` | Макс. параллельных вызовов target |
| `num_repetitions` | int | `1` | Сколько раз повторить каждый пример |
| `metadata` | dict | `None` | Метаданные эксперимента |

`max_concurrency` критически важен: при 50 examples и `max_concurrency=50` все 50 LLM-вызовов пойдут одновременно, что вызовет rate limiting. Рекомендуется 3-10.

`num_repetitions` > 1 полезен для оценки consistency: при temperature > 0 один пример даёт разные результаты, и repetitions показывают дисперсию.

### 6. Online Evaluation и Monitoring

Offline evaluation (предыдущий раздел) запускается перед деплоем. Online evaluation работает непрерывно в production, автоматически оценивая новые traces.

#### Rules — автоматические evaluators

Rules в LangSmith — это триггеры, которые запускают evaluator на каждом новом trace, соответствующем фильтру:

```
Rule: "assess_quality"
  Filter: tag = "assessment" AND run_type = "chain"
  Evaluator: custom LLM-as-judge
  Action: Add feedback score to run
```

Конфигурация через UI:

1. Перейти в Project → Rules → Create Rule
2. Задать фильтр: `run_type = "chain" AND has(tags, "production")`
3. Выбрать evaluator (built-in или custom)
4. Настроить sampling rate (100% для critical paths, 10% для high-volume)

Rules выполняются асинхронно — они не замедляют production-ответы. Evaluator вызывается после завершения trace.

#### Automation actions

Помимо добавления scores, rules могут запускать actions:

| Action | Описание |
|--------|----------|
| Add to annotation queue | Отправить trace на ручную проверку |
| Add to dataset | Добавить trace как example в dataset |
| Send webhook | HTTP POST на ваш endpoint |
| Add feedback score | Записать score из evaluator |

Пример automation pipeline:

```
Trigger: new trace с tag "assessment"
→ Evaluator: проверить overall_score в допустимом диапазоне (0-100)
→ If score out of range:
    → Add to annotation queue "quality_review"
    → Send webhook to Slack (#alerts channel)
→ Else:
    → Add feedback score "valid_range" = 1.0
```

#### Dashboards

LangSmith предоставляет built-in dashboards с метриками:

| Метрика | Описание | Alert threshold (пример) |
|---------|----------|------------------------|
| P50/P95/P99 latency | Латентность по перцентилям | P95 > 10s |
| Error rate | % traces с ошибками | > 5% |
| Token usage | Средний input/output tokens | Рост > 50% за неделю |
| Cost per trace | Стоимость одного вызова | > $0.05 per trace |
| Feedback scores | Средний score из evaluators | < 0.7 |
| Throughput | Traces per minute | < 10 (если ожидается больше) |

Dashboards обновляются в реальном времени. Для SLA-мониторинга можно настроить alert: "если % traces с quality_score > 0.8 упал ниже 90% — уведомить".

### 7. Annotation Queues

Annotation Queues — механизм для ручной проверки production traces. Это bridge между автоматической оценкой и human-in-the-loop.

#### Зачем нужны

Автоматические evaluators ловят очевидные проблемы: score out of range, empty response, format error. Но subtler issues — некорректная интерпретация рубрики, bias к определённым типам эссе, hallucinated feedback — требуют человеческой проверки.

Annotation queues создают workflow:

1. **Filter** — выбрать traces для проверки (random sample, low-confidence, flagged by rules)
2. **Assign** — распределить между reviewers
3. **Review** — reviewer видит trace: input, output, metadata
4. **Label** — reviewer проставляет labels по schema
5. **Export** — labeled traces → dataset → improve prompts

#### Создание queue

В LangSmith UI:

1. Project → Annotation Queues → New Queue
2. Имя: `assessment_review`
3. Фильтр: `tag = "assessment" AND has(metadata, "needs_review")`
4. Sampling: 10% от matching traces (или все, если volume небольшой)
5. Labeling schema: определить, какие labels reviewer проставляет

#### Labeling schemas

LangSmith поддерживает custom labeling schemas:

| Тип | Пример | Использование |
|-----|--------|---------------|
| Binary | Correct / Incorrect | Быстрая проверка "правильно/неправильно" |
| Categorical | Excellent / Good / Fair / Poor | Многоуровневая оценка |
| Numeric | 1-5 stars | Гранулярная оценка |
| Multi-label | [Accurate, Detailed, Helpful] | Несколько аспектов одновременно |

Для assessment-системы рекомендуемая schema:

```
Label: "assessment_quality"
  - score_accuracy: 1-5 (насколько оценка соответствует эталону)
  - feedback_quality: 1-5 (насколько feedback полезен студенту)
  - bias_detected: yes/no (заметен ли bias)
  - notes: free text (комментарии reviewer)
```

#### Export и feedback loop

Reviewed traces с labels экспортируются в dataset:

```python
client = Client()

runs = client.list_runs(
    project_name="ai-assessment",
    filter='has(feedback, "assessment_quality")',
)

good_examples = []
for run in runs:
    feedback = client.read_feedback(run_id=run.id)
    if feedback.score >= 4:
        good_examples.append({
            "inputs": run.inputs,
            "outputs": run.outputs,
        })

client.create_examples(
    inputs=[e["inputs"] for e in good_examples],
    outputs=[e["outputs"] for e in good_examples],
    dataset_name="assessment-golden-reviewed",
)
```

Этот dataset используется для evaluation новых версий промптов. Цикл замкнулся: production → annotation → dataset → evaluation → better prompts → production.

### 8. LangSmith vs Langfuse — практическое сравнение

Оба инструмента решают одну задачу — observability и evaluation LLM-приложений — но с разной философией и trade-offs.

#### Детальное сравнение фич

| Фича | LangSmith | Langfuse |
|------|-----------|----------|
| **Tracing** | | |
| Автоматический для LangChain | `LANGCHAIN_TRACING_V2=true` (zero-config) | Через `CallbackHandler` (3 строки кода) |
| Custom function tracing | `@traceable` decorator | `@observe` decorator |
| OpenAI SDK tracing | `wrap_openai()` | `observe(openai_client)` |
| Multi-framework | LangChain-first, другие через `@traceable` | Provider-agnostic по дизайну |
| **Prompts** | | |
| Prompt management | Hub — полноценный registry с versions | Встроенный prompt management в UI |
| Versioning | Git-like commits с hashes | Версии с labels (production, staging) |
| A/B testing промптов | Comparison experiments | Через manual switching |
| **Evaluation** | | |
| Evaluation API | `evaluate()` — мощный, declarative | Через scores API + custom code |
| Built-in evaluators | exact_match, embedding_distance, criteria | Нет built-in (всё custom) |
| LLM-as-judge | `LangChainStringEvaluator` | Custom через scores API |
| Comparison experiments | Нативная поддержка, side-by-side UI | Manual comparison |
| Datasets | First-class entity с splits | Datasets через UI/API |
| **Monitoring** | | |
| Dashboards | Built-in с latency, cost, quality | Built-in с latency, cost, scores |
| Rules/Automation | Rules с фильтрами и actions | Нет (через custom webhooks) |
| Annotation Queues | Нативная поддержка | Нативная поддержка |
| **Deployment** | | |
| Self-hosted | Нет | Да (Docker, K8s) |
| Data residency | US/EU cloud only | Полный контроль |
| SSO/RBAC | Enterprise tier | Open-source (free) |
| **Pricing** | | |
| Free tier | 5,000 traces/month | Unlimited (self-host) |
| Paid | $39+/user/month | Cloud: usage-based |

#### Когда выбрать LangSmith

- Проект полностью на LangChain — zero-config tracing, нативная интеграция
- Нужен мощный evaluation pipeline с comparison experiments
- Важен Hub для prompt management с versioning
- Команда < 5 человек (Plus tier экономичен)
- Data residency не является жёстким требованием

#### Когда выбрать Langfuse

- Нужен self-hosted (compliance, data residency, air-gapped environments)
- Multi-framework: часть на LangChain, часть на LlamaIndex, часть raw OpenAI
- Бюджет ограничен — open-source бесплатен при self-host
- Нужна интеграция с OpenTelemetry / существующей observability stack
- Команда > 10 человек (enterprise LangSmith дорог)

#### Совместное использование

Для production-grade системы можно использовать оба инструмента:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."

from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

langfuse_handler = LangfuseCallbackHandler()

result = await chain.ainvoke(
    {"student_work": "..."},
    config={"callbacks": [langfuse_handler]},
)
```

В этой конфигурации:
- LangSmith получает traces автоматически через env vars (для evaluation и Hub)
- Langfuse получает traces через callback handler (для production monitoring, self-hosted dashboard)

Типичное разделение:
- **LangSmith** — evaluation, prompt management (Hub), development & staging
- **Langfuse** — production tracing, cost monitoring, user feedback, self-hosted dashboard

---

## Справочник API

### langsmith.Client()

Основной клиент для программного взаимодействия с LangSmith API.

```python
from langsmith import Client

client = Client(
    api_key="lsv2_pt_...",
    api_url="https://api.smith.langchain.com",
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `api_key` | str | `LANGCHAIN_API_KEY` env | API-ключ LangSmith |
| `api_url` | str | `https://api.smith.langchain.com` | URL API (для enterprise может отличаться) |

Если `LANGCHAIN_API_KEY` установлена в env, клиент подхватывает ключ автоматически.

### Client().create_dataset()

Создание нового dataset.

```python
dataset = client.create_dataset(
    dataset_name="assessment-golden-v1",
    description="Golden dataset for essay assessment evaluation",
    data_type="kv",
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `dataset_name` | str | — | Уникальное имя dataset |
| `description` | str | `""` | Описание |
| `data_type` | str | `"kv"` | Тип данных: `"kv"` (key-value), `"llm"` (messages), `"chat"` |

Возвращает объект `Dataset` с полями `id`, `name`, `description`, `created_at`.

### Client().create_examples()

Добавление примеров в dataset.

```python
client.create_examples(
    inputs=[
        {"student_work": "Essay 1...", "rubric_id": "essay_default"},
        {"student_work": "Essay 2...", "rubric_id": "essay_default"},
    ],
    outputs=[
        {"overall_score": 85, "summary": "Strong essay..."},
        {"overall_score": 42, "summary": "Weak essay..."},
    ],
    dataset_name="assessment-golden-v1",
    splits=["test"],
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `inputs` | list[dict] | — | Входные данные для каждого примера |
| `outputs` | list[dict] | `None` | Ожидаемые выходы (reference answers) |
| `dataset_name` | str | `None` | Имя dataset (альтернатива `dataset_id`) |
| `dataset_id` | UUID | `None` | ID dataset |
| `metadata` | list[dict] | `None` | Метаданные для каждого примера |
| `splits` | list[str] | `None` | Split labels (e.g. `["test"]`, `["train"]`) |

Длины `inputs`, `outputs`, `metadata` должны совпадать.

### langsmith.evaluate()

Главная функция для запуска evaluation.

```python
from langsmith import evaluate

results = evaluate(
    target_function,
    data="assessment-golden-v1",
    evaluators=[score_accuracy, feedback_quality],
    summary_evaluators=[mean_absolute_error],
    experiment_prefix="assessment-v2",
    max_concurrency=5,
    num_repetitions=1,
    metadata={"model": "claude-sonnet-4-20250514", "prompt_version": "v2"},
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `target` | Callable | — | Функция `(inputs: dict) -> dict` |
| `data` | str / Dataset | — | Имя dataset, ID, или объект Dataset |
| `evaluators` | list[Callable] | `[]` | Per-example evaluators |
| `summary_evaluators` | list[Callable] | `[]` | Aggregate evaluators |
| `experiment_prefix` | str | `""` | Префикс имени эксперимента в UI |
| `max_concurrency` | int | `5` | Макс. параллельных вызовов target |
| `num_repetitions` | int | `1` | Кол-во повторов каждого example |
| `metadata` | dict | `None` | Метаданные эксперимента |

Возвращает `ExperimentResults` с доступом к individual results и aggregate scores.

### @traceable

Декоратор для трейсинга произвольных Python-функций.

```python
from langsmith import traceable

@traceable(
    name="assess_essay",
    run_type="chain",
    tags=["assessment", "v2"],
    metadata={"version": "2.0"},
)
async def assess_essay(student_work: str, rubric_id: str) -> dict:
    ...
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `name` | str | function name | Имя run в UI |
| `run_type` | str | `"chain"` | Тип: `"chain"`, `"tool"`, `"llm"`, `"retriever"` |
| `tags` | list[str] | `[]` | Теги для фильтрации |
| `metadata` | dict | `{}` | Метаданные |

Работает с sync и async функциями. Автоматически записывает inputs (аргументы функции), outputs (return value), errors, timing.

### hub.pull() / hub.push()

Управление промптами через LangChain Hub.

```python
from langchain import hub

prompt = hub.pull("my-org/assessment-prompt")

prompt_versioned = hub.pull("my-org/assessment-prompt:abc123def")

hub.push("my-org/assessment-prompt-v2", new_prompt)
```

**hub.pull()**:

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `owner_repo_commit` | str | — | `"owner/name"` или `"owner/name:commit_hash"` |
| `api_url` | str | `None` | Custom API URL |
| `api_key` | str | `None` | API key (по умолчанию из env) |

Возвращает объект промпта (обычно `ChatPromptTemplate`).

**hub.push()**:

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `owner_repo_commit` | str | — | `"owner/name"` |
| `object` | Runnable | — | Объект для публикации (prompt, chain) |
| `new_repo_is_public` | bool | `False` | Сделать публичным при первом push |

Каждый push создаёт новый commit. Возвращает URL промпта в Hub.

### langsmith.wrappers.wrap_openai()

Обёртка для трейсинга raw OpenAI SDK вызовов.

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Assess: ..."}],
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `client` | OpenAI / AsyncOpenAI | — | Инстанс OpenAI клиента |

Возвращает обёрнутый клиент с тем же API. Все вызовы `chat.completions.create()` записываются в LangSmith.

### EvaluationResult

Результат одного evaluator для одного example.

```python
from langsmith.evaluation import EvaluationResult

result = EvaluationResult(
    key="score_accuracy",
    score=0.85,
    comment="Predicted: 72, Expected: 65, Diff: 7",
)
```

| Поле | Тип | Описание |
|------|-----|----------|
| `key` | str | Имя метрики (отображается в UI) |
| `score` | float | Числовое значение (обычно 0.0-1.0) |
| `value` | str | Категориальное значение (альтернатива score) |
| `comment` | str | Текстовый комментарий |

---

## Практика: роутер `/api/v1/langsmith`

### Шаг 1. Схемы — `app/schemas/langsmith.py`

Pydantic-модели для request/response всех эндпоинтов:

```python
from pydantic import BaseModel, Field


class HubPullRequest(BaseModel):
    prompt_name: str = Field(description="Hub prompt identifier, e.g. 'my-org/assessment-prompt'")
    commit_hash: str | None = Field(default=None, description="Specific version hash")
    student_work: str = Field(description="Text to assess using the pulled prompt")
    rubric_id: str = Field(default="essay_default")


class HubPullResponse(BaseModel):
    prompt_name: str
    prompt_version: str
    assessment_result: dict
    trace_url: str | None = None


class DatasetExample(BaseModel):
    student_work: str
    expected_score: int = Field(ge=0, le=100)
    expected_summary: str


class CreateDatasetRequest(BaseModel):
    dataset_name: str = Field(description="Unique name for the dataset")
    description: str = Field(default="")
    examples: list[DatasetExample] = Field(min_length=1)


class CreateDatasetResponse(BaseModel):
    dataset_name: str
    dataset_id: str
    examples_count: int
    message: str


class EvaluatorConfig(BaseModel):
    name: str = Field(description="Evaluator name: 'score_accuracy', 'feedback_quality', 'embedding_distance'")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class EvaluateRequest(BaseModel):
    dataset_name: str
    experiment_prefix: str = Field(default="eval")
    evaluators: list[EvaluatorConfig] = Field(min_length=1)
    max_concurrency: int = Field(default=3, ge=1, le=20)


class EvaluationResultItem(BaseModel):
    example_id: str
    inputs: dict
    predicted: dict
    scores: dict[str, float]
    comments: dict[str, str]


class EvaluateResponse(BaseModel):
    experiment_name: str
    dataset_name: str
    total_examples: int
    aggregate_scores: dict[str, float]
    results: list[EvaluationResultItem]


class CompareRequest(BaseModel):
    dataset_name: str
    prompt_name_a: str = Field(description="First prompt Hub identifier")
    prompt_name_b: str = Field(description="Second prompt Hub identifier")
    evaluators: list[EvaluatorConfig] = Field(min_length=1)
    max_concurrency: int = Field(default=3, ge=1, le=20)


class CompareResultItem(BaseModel):
    example_id: str
    inputs: dict
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    winner: str


class CompareResponse(BaseModel):
    dataset_name: str
    prompt_a: str
    prompt_b: str
    total_examples: int
    wins_a: int
    wins_b: int
    ties: int
    aggregate_scores_a: dict[str, float]
    aggregate_scores_b: dict[str, float]
    results: list[CompareResultItem]
```

Обратите внимание:
- `HubPullRequest` позволяет указать конкретную версию промпта через `commit_hash` — это важно для reproducibility
- `EvaluatorConfig` абстрагирует выбор evaluator — клиент указывает имя, сервис маппит на реальную функцию
- `CompareResponse` включает `wins_a`, `wins_b`, `ties` — агрегат для быстрого принятия решений

### Шаг 2. Сервис — `app/services/langsmith_service.py`

Сервис инкапсулирует всю логику взаимодействия с LangSmith API. Роутер вызывает функции сервиса, не зная деталей реализации.

```python
import os
from langsmith import Client, traceable
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Run, Example
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.assessment import AssessmentResponse


FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert essay assessor. Evaluate the student work against standard academic criteria. Return a JSON with 'overall_score' (0-100) and 'summary' (2-3 sentences)."),
    ("human", "Assess this student work:\n\n{student_work}"),
])


def get_langsmith_client() -> Client:
    return Client()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("MODEL_NAME", "claude-sonnet-4-20250514"),
        temperature=0.0,
        max_tokens=4096,
    )


@traceable(name="pull_and_assess", run_type="chain", tags=["hub", "assessment"])
async def pull_and_assess(
    prompt_name: str,
    student_work: str,
    rubric_id: str,
    commit_hash: str | None = None,
) -> dict:
    full_name = f"{prompt_name}:{commit_hash}" if commit_hash else prompt_name

    try:
        prompt = hub.pull(full_name)
    except Exception:
        prompt = FALLBACK_PROMPT

    llm = get_llm()
    structured_llm = llm.with_structured_output(AssessmentResponse)
    chain = prompt | structured_llm

    result = await chain.ainvoke({
        "student_work": student_work,
        "rubric_id": rubric_id,
    })

    return {
        "overall_score": result.overall_score,
        "summary": result.summary,
        "strengths": result.strengths,
        "improvements": result.improvements,
        "criterion_scores": [
            {
                "criterion_name": cs.criterion_name,
                "score": cs.score,
                "max_score": cs.max_score,
                "feedback": cs.feedback,
            }
            for cs in result.criterion_scores
        ],
    }


def create_assessment_dataset(
    dataset_name: str,
    description: str,
    examples: list[dict],
) -> dict:
    client = get_langsmith_client()

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )

    inputs = []
    outputs = []
    for ex in examples:
        inputs.append({
            "student_work": ex["student_work"],
            "rubric_id": "essay_default",
        })
        outputs.append({
            "overall_score": ex["expected_score"],
            "summary": ex["expected_summary"],
        })

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_name=dataset_name,
    )

    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset_name,
        "examples_count": len(examples),
    }


def _build_score_accuracy_evaluator(threshold: float):
    def evaluator(run: Run, example: Example) -> EvaluationResult:
        predicted = run.outputs.get("overall_score", 0)
        expected = example.outputs.get("overall_score", 0)
        diff = abs(predicted - expected)
        max_diff = 100
        score = max(0.0, 1.0 - diff / max_diff)

        return EvaluationResult(
            key="score_accuracy",
            score=score,
            comment=f"Predicted: {predicted}, Expected: {expected}, Diff: {diff}",
        )
    return evaluator


def _build_feedback_quality_evaluator(threshold: float):
    def evaluator(run: Run, example: Example) -> EvaluationResult:
        summary = run.outputs.get("summary", "")
        word_count = len(summary.split())
        has_substance = word_count >= 10
        score = 1.0 if has_substance else 0.0

        return EvaluationResult(
            key="feedback_quality",
            score=score,
            comment=f"Summary word count: {word_count}, sufficient: {has_substance}",
        )
    return evaluator


EVALUATOR_REGISTRY = {
    "score_accuracy": _build_score_accuracy_evaluator,
    "feedback_quality": _build_feedback_quality_evaluator,
}


def _resolve_evaluators(configs: list[dict]) -> list:
    evaluators = []
    for config in configs:
        name = config["name"]
        threshold = config.get("threshold", 0.7)
        builder = EVALUATOR_REGISTRY.get(name)
        if not builder:
            continue
        evaluators.append(builder(threshold))
    return evaluators


def _build_target_function(prompt_name: str | None = None):
    def target(inputs: dict) -> dict:
        llm = get_llm()

        if prompt_name:
            try:
                prompt = hub.pull(prompt_name)
            except Exception:
                prompt = FALLBACK_PROMPT
        else:
            prompt = FALLBACK_PROMPT

        structured_llm = llm.with_structured_output(AssessmentResponse)
        chain = prompt | structured_llm
        result = chain.invoke(inputs)

        return {
            "overall_score": result.overall_score,
            "summary": result.summary,
        }

    return target


def run_evaluation(
    dataset_name: str,
    experiment_prefix: str,
    evaluator_configs: list[dict],
    max_concurrency: int = 3,
    prompt_name: str | None = None,
) -> dict:
    evaluators = _resolve_evaluators(evaluator_configs)
    if not evaluators:
        evaluators = [_build_score_accuracy_evaluator(0.7)]

    target = _build_target_function(prompt_name)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )

    result_items = []
    aggregate = {}

    for result in results:
        example_id = str(result["example"].id) if result.get("example") else "unknown"

        scores = {}
        comments = {}
        for eval_result in result.get("evaluation_results", {}).get("results", []):
            scores[eval_result.key] = eval_result.score
            comments[eval_result.key] = eval_result.comment or ""

        result_items.append({
            "example_id": example_id,
            "inputs": result.get("run", {}).inputs if result.get("run") else {},
            "predicted": result.get("run", {}).outputs if result.get("run") else {},
            "scores": scores,
            "comments": comments,
        })

        for key, score in scores.items():
            if key not in aggregate:
                aggregate[key] = []
            aggregate[key].append(score)

    aggregate_scores = {
        key: sum(values) / len(values)
        for key, values in aggregate.items()
        if values
    }

    return {
        "experiment_name": experiment_prefix,
        "dataset_name": dataset_name,
        "total_examples": len(result_items),
        "aggregate_scores": aggregate_scores,
        "results": result_items,
    }


def run_comparison(
    dataset_name: str,
    prompt_name_a: str,
    prompt_name_b: str,
    evaluator_configs: list[dict],
    max_concurrency: int = 3,
) -> dict:
    evaluators = _resolve_evaluators(evaluator_configs)
    if not evaluators:
        evaluators = [_build_score_accuracy_evaluator(0.7)]

    target_a = _build_target_function(prompt_name_a)
    target_b = _build_target_function(prompt_name_b)

    results_a = evaluate(
        target_a,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"compare-a-{prompt_name_a.split('/')[-1]}",
        max_concurrency=max_concurrency,
    )

    results_b = evaluate(
        target_b,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"compare-b-{prompt_name_b.split('/')[-1]}",
        max_concurrency=max_concurrency,
    )

    comparison_items = []
    wins_a = 0
    wins_b = 0
    ties = 0
    all_scores_a = {}
    all_scores_b = {}

    for res_a, res_b in zip(results_a, results_b):
        example_id = str(res_a["example"].id) if res_a.get("example") else "unknown"

        scores_a = {}
        scores_b = {}

        for er in res_a.get("evaluation_results", {}).get("results", []):
            scores_a[er.key] = er.score
        for er in res_b.get("evaluation_results", {}).get("results", []):
            scores_b[er.key] = er.score

        avg_a = sum(scores_a.values()) / len(scores_a) if scores_a else 0
        avg_b = sum(scores_b.values()) / len(scores_b) if scores_b else 0

        if avg_a > avg_b + 0.01:
            winner = "a"
            wins_a += 1
        elif avg_b > avg_a + 0.01:
            winner = "b"
            wins_b += 1
        else:
            winner = "tie"
            ties += 1

        comparison_items.append({
            "example_id": example_id,
            "inputs": res_a.get("run", {}).inputs if res_a.get("run") else {},
            "scores_a": scores_a,
            "scores_b": scores_b,
            "winner": winner,
        })

        for key, score in scores_a.items():
            all_scores_a.setdefault(key, []).append(score)
        for key, score in scores_b.items():
            all_scores_b.setdefault(key, []).append(score)

    agg_a = {k: sum(v) / len(v) for k, v in all_scores_a.items() if v}
    agg_b = {k: sum(v) / len(v) for k, v in all_scores_b.items() if v}

    return {
        "dataset_name": dataset_name,
        "prompt_a": prompt_name_a,
        "prompt_b": prompt_name_b,
        "total_examples": len(comparison_items),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "aggregate_scores_a": agg_a,
        "aggregate_scores_b": agg_b,
        "results": comparison_items,
    }
```

Ключевые архитектурные решения сервиса:

1. **EVALUATOR_REGISTRY** — маппинг имени evaluator на builder-функцию. Клиент отправляет `"score_accuracy"`, сервис создаёт нужный evaluator. Легко расширяется: добавляешь функцию и запись в dict.

2. **`_build_target_function()`** — фабрика target functions для evaluation. Принимает опциональный `prompt_name` и возвращает callable, совместимый с `evaluate()`. Замыкание захватывает prompt_name.

3. **Fallback prompt** — если Hub prompt не найден, используется встроенный шаблон. Production-код не должен падать из-за недоступности Hub.

4. **`@traceable`** на `pull_and_assess` — каждый вызов записывается в LangSmith, даже если трейсинг для остальных функций не включен.

### Шаг 3. Роутер — `app/api/v1/langsmith.py`

4 эндпоинта, каждый демонстрирует отдельную концепцию LangSmith:

```python
from fastapi import APIRouter, HTTPException

from app.schemas.langsmith import (
    HubPullRequest,
    HubPullResponse,
    CreateDatasetRequest,
    CreateDatasetResponse,
    EvaluateRequest,
    EvaluateResponse,
    EvaluationResultItem,
    CompareRequest,
    CompareResponse,
    CompareResultItem,
)
from app.services.langsmith_service import (
    pull_and_assess,
    create_assessment_dataset,
    run_evaluation,
    run_comparison,
)


router = APIRouter(prefix="/langsmith", tags=["langsmith"])


@router.post("/hub/pull")
async def hub_pull_and_assess(request: HubPullRequest) -> HubPullResponse:
    try:
        result = await pull_and_assess(
            prompt_name=request.prompt_name,
            student_work=request.student_work,
            rubric_id=request.rubric_id,
            commit_hash=request.commit_hash,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hub pull or assessment failed: {e}")

    version = request.commit_hash or "latest"

    return HubPullResponse(
        prompt_name=request.prompt_name,
        prompt_version=version,
        assessment_result=result,
    )


@router.post("/dataset")
async def create_dataset(request: CreateDatasetRequest) -> CreateDatasetResponse:
    examples = [
        {
            "student_work": ex.student_work,
            "expected_score": ex.expected_score,
            "expected_summary": ex.expected_summary,
        }
        for ex in request.examples
    ]

    try:
        result = create_assessment_dataset(
            dataset_name=request.dataset_name,
            description=request.description,
            examples=examples,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Dataset creation failed: {e}")

    return CreateDatasetResponse(
        dataset_name=result["dataset_name"],
        dataset_id=result["dataset_id"],
        examples_count=result["examples_count"],
        message=f"Dataset '{result['dataset_name']}' created with {result['examples_count']} examples",
    )


@router.post("/evaluate")
async def evaluate_dataset(request: EvaluateRequest) -> EvaluateResponse:
    evaluator_configs = [
        {"name": ev.name, "threshold": ev.threshold}
        for ev in request.evaluators
    ]

    try:
        result = run_evaluation(
            dataset_name=request.dataset_name,
            experiment_prefix=request.experiment_prefix,
            evaluator_configs=evaluator_configs,
            max_concurrency=request.max_concurrency,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {e}")

    return EvaluateResponse(
        experiment_name=result["experiment_name"],
        dataset_name=result["dataset_name"],
        total_examples=result["total_examples"],
        aggregate_scores=result["aggregate_scores"],
        results=[
            EvaluationResultItem(**item)
            for item in result["results"]
        ],
    )


@router.post("/compare")
async def compare_prompts(request: CompareRequest) -> CompareResponse:
    evaluator_configs = [
        {"name": ev.name, "threshold": ev.threshold}
        for ev in request.evaluators
    ]

    try:
        result = run_comparison(
            dataset_name=request.dataset_name,
            prompt_name_a=request.prompt_name_a,
            prompt_name_b=request.prompt_name_b,
            evaluator_configs=evaluator_configs,
            max_concurrency=request.max_concurrency,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Comparison failed: {e}")

    return CompareResponse(
        dataset_name=result["dataset_name"],
        prompt_a=result["prompt_a"],
        prompt_b=result["prompt_b"],
        total_examples=result["total_examples"],
        wins_a=result["wins_a"],
        wins_b=result["wins_b"],
        ties=result["ties"],
        aggregate_scores_a=result["aggregate_scores_a"],
        aggregate_scores_b=result["aggregate_scores_b"],
        results=[
            CompareResultItem(**item)
            for item in result["results"]
        ],
    )
```

**Как каждый эндпоинт связан с теорией:**

| Эндпоинт | Концепция из теории | Что демонстрирует |
|---|---|---|
| `POST /hub/pull` | §3 Hub, §2 Tracing | Загрузка промпта из Hub, assessment с трейсингом |
| `POST /dataset` | §4 Datasets | Создание dataset для evaluation через API |
| `POST /evaluate` | §5 Evaluation | Запуск evaluation с custom evaluators |
| `POST /compare` | §5 Comparison experiments | A/B-сравнение двух промптов |

### Шаг 4. Регистрация и тестирование

Добавьте роутер в `app/api/router.py`:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, prompts, langsmith

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(langsmith.router)
```

Обновите `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=ai-assessment-bootcamp
```

Запуск:

```bash
uvicorn app.main:app --reload
```

#### POST /langsmith/hub/pull — загрузка промпта и assessment

```bash
curl -s -X POST http://localhost:8000/api/v1/langsmith/hub/pull \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "my-org/essay-assessment",
    "student_work": "Climate change is a major threat. Rising temperatures cause ice to melt, leading to higher sea levels. Governments should implement carbon taxes and invest in renewable energy. Without action, future generations will suffer.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

Ожидаемый результат: `prompt_version: "latest"`, `assessment_result` содержит `overall_score`, `summary`, `strengths`, `improvements`. Если промпт не найден в Hub — используется fallback, assessment всё равно выполнится.

В LangSmith UI появится trace `pull_and_assess` с child runs для prompt template и LLM call.

#### POST /langsmith/dataset — создание dataset

```bash
curl -s -X POST http://localhost:8000/api/v1/langsmith/dataset \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "assessment-golden-test",
    "description": "Test golden dataset for bootcamp",
    "examples": [
      {
        "student_work": "Climate change is a major threat. Rising temperatures cause ice to melt, leading to higher sea levels. Governments should implement carbon taxes.",
        "expected_score": 55,
        "expected_summary": "Basic essay with clear thesis but lacking evidence and depth."
      },
      {
        "student_work": "The intersection of artificial intelligence and labor economics presents a nuanced challenge. While Frey and Osborne (2013) estimated 47% automation risk, subsequent analyses by Arntz et al. (2016) suggest only 9% of jobs are fully automatable.",
        "expected_score": 88,
        "expected_summary": "Strong academic essay with citations, nuanced analysis, and clear structure."
      },
      {
        "student_work": "AI is good. It helps people. The end.",
        "expected_score": 15,
        "expected_summary": "Extremely weak essay with no thesis, evidence, or analysis."
      }
    ]
  }' | python -m json.tool
```

Ожидаемый результат: `examples_count: 3`, `dataset_id` — UUID созданного dataset. В LangSmith UI → Datasets появится `assessment-golden-test` с 3 examples.

#### POST /langsmith/evaluate — запуск evaluation

```bash
curl -s -X POST http://localhost:8000/api/v1/langsmith/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "assessment-golden-test",
    "experiment_prefix": "bootcamp-eval-v1",
    "evaluators": [
      {"name": "score_accuracy", "threshold": 0.7},
      {"name": "feedback_quality", "threshold": 0.7}
    ],
    "max_concurrency": 2
  }' | python -m json.tool
```

Ожидаемый результат: `total_examples: 3`, `aggregate_scores` содержит `score_accuracy` и `feedback_quality` (средние значения по dataset). Каждый `results[i]` содержит `scores` и `comments` для конкретного example.

Evaluation занимает время: 3 examples × LLM call ≈ 10-20 секунд. В LangSmith UI → Experiments появится `bootcamp-eval-v1` с детальными результатами.

#### POST /langsmith/compare — A/B-сравнение промптов

```bash
curl -s -X POST http://localhost:8000/api/v1/langsmith/compare \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "assessment-golden-test",
    "prompt_name_a": "my-org/assessment-v1",
    "prompt_name_b": "my-org/assessment-v2",
    "evaluators": [
      {"name": "score_accuracy", "threshold": 0.7}
    ],
    "max_concurrency": 2
  }' | python -m json.tool
```

Ожидаемый результат: `wins_a`, `wins_b`, `ties` показывают, какой промпт лучше на каждом example. `aggregate_scores_a` vs `aggregate_scores_b` — общее сравнение. Каждый `results[i]` содержит `winner: "a"`, `"b"`, или `"tie"`.

Comparison занимает 2x времени evaluation (оба промпта прогоняются на всём dataset). В LangSmith UI → Experiments оба прогона видны рядом на одном dataset.

---

## Чеклист самопроверки

Ответь на эти вопросы **своими словами**:

- [ ] Что такое LangSmith и чем он отличается от Langfuse? Назови 4 главные функции LangSmith.
- [ ] Как включить автоматический трейсинг для LangChain-приложения? Какие env vars нужны?
- [ ] Что такое Run в модели данных LangSmith? Как организованы parent-child relationships?
- [ ] Для чего нужен `@traceable` декоратор? Чем он отличается от автоматического трейсинга?
- [ ] Как LangChain Hub решает проблему hardcoded промптов? Что даёт версионирование через commit hashes?
- [ ] Опиши структуру Dataset в LangSmith: что такое inputs, outputs, splits?
- [ ] Как работает `evaluate()` — опиши последовательность шагов от вызова до результата.
- [ ] Чем отличается custom evaluator от built-in? Напиши сигнатуру custom evaluator.
- [ ] Что такое comparison experiment и зачем он нужен при итерации над промптами?
- [ ] Когда стоит использовать LangSmith, а когда Langfuse? Назови 3 критерия выбора.

---

## Частые ошибки

### 1. Забыли установить LANGCHAIN_API_KEY

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

chain = prompt | llm
result = await chain.ainvoke({"student_work": "..."})
```

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."

chain = prompt | llm
result = await chain.ainvoke({"student_work": "..."})
```

Без `LANGCHAIN_API_KEY` трейсы не отправляются в LangSmith. Ошибки нет — traces молча теряются. Всегда проверяйте наличие ключа перед включением трейсинга.

### 2. Hardcoded prompt вместо Hub pull

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assessor..."),
    ("human", "{student_work}"),
])
chain = prompt | llm
```

```python
prompt = hub.pull("my-org/assessment-prompt:abc123")
chain = prompt | llm
```

Hardcoded промпт требует деплоя для каждого изменения. Hub позволяет менять промпт без деплоя кода и вести историю изменений. Используйте pinned version (`:commit_hash`) для production.

### 3. evaluate() без max_concurrency

```python
results = evaluate(
    target,
    data="large-dataset-500-examples",
    evaluators=[accuracy_evaluator],
)
```

```python
results = evaluate(
    target,
    data="large-dataset-500-examples",
    evaluators=[accuracy_evaluator],
    max_concurrency=5,
)
```

Default `max_concurrency` может запустить слишком много параллельных LLM-вызовов, вызывая rate limiting у провайдера. Для datasets > 20 examples всегда явно ограничивайте concurrency.

### 4. Dataset без reference outputs

```python
client.create_examples(
    inputs=[{"student_work": "Essay text..."}],
    dataset_name="my-dataset",
)

def evaluator(run: Run, example: Example) -> EvaluationResult:
    expected = example.outputs["overall_score"]
    ...
```

```python
client.create_examples(
    inputs=[{"student_work": "Essay text..."}],
    outputs=[{"overall_score": 75, "summary": "..."}],
    dataset_name="my-dataset",
)
```

Если `outputs` не указаны при создании examples, evaluator получит `example.outputs = None` и упадёт с `TypeError`. Всегда проверяйте наличие outputs в evaluator: `example.outputs.get("overall_score", 0)`.

### 5. Sync target function с async LLM call

```python
def target(inputs: dict) -> dict:
    result = await chain.ainvoke(inputs)
    return {"overall_score": result.overall_score}
```

```python
def target(inputs: dict) -> dict:
    result = chain.invoke(inputs)
    return {"overall_score": result.overall_score}
```

`evaluate()` по умолчанию вызывает target функцию синхронно. Если target помечена как `def` (не `async def`), внутри неё нельзя использовать `await`. Используйте `chain.invoke()` вместо `chain.ainvoke()` в sync target functions, или сделайте target async.

### 6. Сравнение на разных datasets

```python
results_a = evaluate(target_a, data="dataset-v1", ...)
results_b = evaluate(target_b, data="dataset-v2", ...)
```

```python
results_a = evaluate(target_a, data="dataset-v1", ...)
results_b = evaluate(target_b, data="dataset-v1", ...)
```

A/B-сравнение валидно только на одном dataset. Разные datasets имеют разные примеры — сравнение бессмысленно. Всегда используйте один и тот же dataset для обеих версий.

---

## Что читать дальше

- [LangSmith Documentation](https://docs.smith.langchain.com/) — полная документация платформы
- [LangSmith Tracing](https://docs.smith.langchain.com/observability) — настройка трейсинга
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — evaluation API и best practices
- [LangChain Hub](https://smith.langchain.com/hub) — реестр промптов
- [LangSmith Python SDK](https://docs.smith.langchain.com/reference/python) — API reference
- [LangSmith Pricing](https://www.langchain.com/pricing) — тарифы и лимиты
- [LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook) — примеры и рецепты
- [Langfuse vs LangSmith](https://langfuse.com/docs/langfuse-vs-langsmith) — сравнение от Langfuse

**Следующая тема:** [Тема 18: Ollama и локальные LLM](topic_18_local_llms.md)

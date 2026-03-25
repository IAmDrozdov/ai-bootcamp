# Тема 9: Evaluation

> **Пререквизиты:** [Тема 1–4](topic_01_prompt_engineering.md), [Тема 3 (Structured Output)](topic_03_structured_output.md), рекомендуется [Тема 5 (RAG)](topic_05_rag.md), [Тема 8 (Observability)](topic_08_observability.md)  
> **Что добавляем в проект:** роутер `app/api/v1/eval.py`, сервис `app/services/evaluation.py`, схемы `app/schemas/evaluation.py`, директория `data/golden/`  
> **Зависимости:** `langfuse`, `ragas`, `langsmith`, `numpy`, `scipy` (группа `eval` в pyproject.toml)

---

## Теория

### 1. Зачем evaluation в LLM-системах

Обычный софт тестируется бинарно: тест зелёный — работает, красный — сломано. В LLM-приложениях тесты на корректность формата (JSON парсится, поля существуют) необходимы, но недостаточны. Модель может вернуть валидный JSON с абсурдными оценками: 25/25 за эссе без тезиса или feedback "Good job" для работы с нулём цитат.

Evaluation отвечает на вопросы другого уровня:

- **Accuracy** — насколько оценки модели совпадают с экспертными?
- **Consistency** — при повторном прогоне с temperature=0 те же оценки или разброс?
- **Regression** — новый промпт лучше или хуже предыдущего?
- **Bias** — модель систематически завышает/занижает определённые критерии?

Evaluation — это CI/CD для промптов. Прежде чем менять промпт в production, прогони его на golden dataset и убедись, что метрики не деградировали. Без evaluation каждое изменение промпта — рулетка: кажется, что стало лучше, но без данных это subjectiveвпечатление.

Три уровня evaluation:

1. **Unit-level** — один пример → ответ корректен? Формат, границы значений, внутренняя согласованность (overall_score == sum)
2. **Dataset-level** — golden dataset → агрегированные метрики (MAE, correlation, within-N). Показывает общую картину
3. **A/B comparison** — два промпта / две модели → какой лучше на том же датасете?

В production-системе evaluation запускается:

- При каждом изменении промпта (pre-deploy check)
- Регулярно на выборке production-трафика (мониторинг дрифта качества)
- При смене версии модели у провайдера (новый релиз Claude/GPT)
- После изменения rubric-структуры (новые критерии → перекалибровка)

### 2. Офлайн и онлайн evaluation

**Офлайн evaluation** — прогон на заранее подготовленном golden dataset перед деплоем. Контролируемые условия, воспроизводимые результаты, temperature=0. Это основной инструмент для принятия решений "деплоить или нет".

**Онлайн evaluation** — анализ реальных ответов в production. Работает совместно с observability (Тема 8):

- **Langfuse scores**: привязка оценки качества к каждому trace. Например: после каждого assessment привязать MAE к trace
- **User feedback**: пользователь помечает ответ как "полезный" / "неполезный"
- **LLM-as-judge на выборке**: периодически прогонять judge на случайных 10% production-ответов
- **Drift detection**: если средний score за неделю упал — сигнал к расследованию

Офлайн evaluation отвечает "достаточно ли хорош новый промпт?", онлайн — "продолжает ли система работать хорошо в реальных условиях?"

### 3. Golden dataset

Golden dataset — набор данных с эталонными ответами, размеченными **экспертом** (не LLM). Это фундамент всей evaluation-системы.

Структура одной записи:

```json
{
  "id": "golden_001",
  "input": {
    "student_work": "Essay text...",
    "rubric_id": "essay_default"
  },
  "expected_output": {
    "overall_score": 72,
    "criterion_scores": [
      {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25}
    ]
  },
  "metadata": {
    "quality_level": "medium",
    "description": "Decent essay on climate policy"
  }
}
```

Требования к golden dataset:

| Параметр | Минимум | Оптимум | Почему |
|----------|---------|---------|--------|
| Размер | 20 | 50–100 | < 20 — слишком шумные метрики, статистически ненадёжно |
| Strong примеры | 5 | 15–25 | Работы на 75–95 баллов |
| Medium примеры | 5 | 15–25 | Работы на 50–74 балла |
| Weak примеры | 5 | 10–15 | Работы на 20–49 баллов |
| Edge cases | 5 | 10–15 | Короткие, off-topic, инъекции, mixed language |

Почему LLM-сгенерированные эталоны не подходят: LLM оценивает → LLM проверяет = circular evaluation. Модель согласна сама с собой, alignment будет искусственно высоким. Эталоны должны быть от человека-эксперта.

Версионирование: golden dataset хранится в Git. Каждая правка (добавление примеров, коррекция оценок) — отдельный коммит с описанием. Это позволяет отслеживать, как менялись эталоны и как это повлияло на метрики.

Калибровка: перед использованием стоит проверить согласованность между несколькими экспертами (inter-rater agreement). Если два эксперта дают разные оценки одному эссе — golden dataset ненадёжен.

### 4. Метрики для оценочных систем

Для числовых оценок используются четыре основные метрики:

| Метрика | Формула | Что показывает | Хорошее значение для нашего проекта |
|---------|---------|---------------|--------------------------------------|
| MAE | Σ\|predicted − actual\| / N | Средняя абсолютная ошибка в баллах | ≤ 3 для критерия (max 25), ≤ 5 для overall |
| Exact Match | count(pred == actual) / N | Доля точных совпадений | > 15% (полное совпадение необязательно) |
| Within-N | count(\|pred − actual\| ≤ N) / N | Доля оценок в пределах N баллов | Within-5 > 75% |
| Pearson r | ρ(predicted, actual) | Линейная корреляция | > 0.7 |

**MAE (Mean Absolute Error)** — основная метрика. Для критерия с max_score=25, MAE=3 означает "в среднем ошибка на 3 балла", что составляет 12% от шкалы. MAE=5 — ещё приемлемо. MAE=10 — система непригодна.

**Pearson correlation** — показывает, сохраняется ли порядок: если модель ставит выше тем, кто реально лучше — корреляция высокая, даже если абсолютные значения отличаются. Полезно, когда важен ranking, а не точные числа.

**Within-N** — практически полезная метрика. Within-5=80% означает "в 80% случаев оценка отклоняется не более чем на 5 баллов". Для педагогов это часто важнее, чем MAE.

**Exact Match** — самая строгая метрика. В нашем проекте 100% exact match маловероятен и не требуется. 15–20% — нормально для системы с 5 критериями.

Важно считать метрики **по каждому критерию отдельно**: модель может быть точной для "Thesis & Argument" (MAE=2), но слабой для "Critical Thinking" (MAE=6). Агрегированные метрики скрывают эти различия.

### 5. LLM-as-judge

Вместо ручной проверки каждого ответа используем другую LLM для оценки качества. Это масштабируемый подход для evaluation больших датасетов.

Три паттерна LLM-as-judge:

| Паттерн | Описание | Когда использовать |
|---------|---------|-------------------|
| **Pointwise** | Оцени один ответ по шкале 1–5 | Качество отдельного ответа |
| **Pairwise** | Какой из двух ответов лучше? | A/B сравнение |
| **Reference-based** | Насколько ответ совпадает с эталоном? | Оценка против golden dataset |

Для нашего проекта используем **reference-based**: сравниваем оценки системы с эталонными из golden dataset.

Judge получает:
1. **System output** — оценки и feedback от нашей assessment-системы
2. **Expected output** — эталонные оценки от эксперта
3. (Опционально) **Student work** — для проверки, обоснован ли feedback

Judge возвращает:
- `alignment_score` (0–100): общая степень совпадения
- `criterion_analysis`: по каждому критерию — expected vs actual + текстовый judgment
- `discrepancies`: список существенных расхождений (|diff| > 5)

Ограничения LLM-as-judge:

- **Self-bias** — модель может быть предвзята к ответам "своей" модели. Claude-judge оценивает Claude-subject выше, чем GPT-subject (и наоборот)
- **Position bias** — в pairwise-оценке первый ответ часто получает преимущество. Решение: прогонять дважды, меняя порядок
- **Verbosity bias** — длинные, подробные ответы оцениваются выше, даже если содержание слабее
- **Стоимость** — каждая judge-оценка = отдельный LLM-вызов. Для 100 записей golden dataset — 100 дополнительных вызовов

Рекомендации:
- Использовать разные модели для subject и judge (или хотя бы разные temperature)
- `temperature=0` для judge (максимальный детерминизм)
- Проверить judge на known-good и known-bad примерах — убедиться, что он различает качество

### 6. RAGAS метрики (для RAG-систем)

Если система использует RAG (Тема 5), помимо метрик из раздела 4 нужны метрики качества retrieval:

| Метрика | Что измеряет | Как вычисляется |
|---------|-------------|----------------|
| **Faithfulness** | Ответ основан на найденных документах, нет галлюцинаций | LLM проверяет каждое утверждение ответа — есть ли оно в контексте |
| **Answer Relevancy** | Ответ релевантен вопросу, не off-topic | LLM генерирует вопросы из ответа, сравнивает с оригинальным |
| **Context Precision** | Релевантные документы находятся в топе результатов | Позиция релевантных чанков в ranked-списке |
| **Context Recall** | Все необходимые документы найдены | LLM проверяет, покрывает ли контекст все факты эталонного ответа |

Faithfulness=0.3 означает, что 70% утверждений в ответе не подтверждаются контекстом — серьёзная проблема с галлюцинациями. Faithfulness > 0.8 — хороший результат.

### 7. A/B testing промптов

Два промпта → один golden dataset → сравнение метрик → выбор лучшего:

```
Prompt A (текущий):       MAE=4.2, Within-5=78%, r=0.72
Prompt B (с CoT):         MAE=3.1, Within-5=86%, r=0.85
Prompt C (few-shot x3):   MAE=3.5, Within-5=82%, r=0.80
```

Для надёжного A/B сравнения:

- **Минимум 20 примеров** (лучше 50+). На 3 примерах разница может быть случайной
- **Одна и та же golden dataset** для обоих промптов — иначе сравнение невалидно
- **temperature=0** для воспроизводимости результатов
- **Разница в MAE > 1.0** — практически значимая для нашего проекта
- **Trade-off analysis**: улучшение одной метрики не должно ухудшать другие. Если MAE упал, но correlation тоже — промпт стал менее предсказуемым
- **Per-criterion сравнение**: новый промпт может быть лучше для "Thesis" но хуже для "Critical Thinking"

### 8. Regression testing и CI/CD для промптов

При каждом изменении промпта — автоматический прогон на golden dataset:

```
Current production: MAE=3.1
New prompt candidate: MAE=3.8  ← REGRESSION! Don't deploy.
```

Правило: деплоим только если метрики не хуже текущих (или в пределах допустимого порога, например +0.5 MAE).

Интеграция в CI/CD:

1. Developer меняет промпт → push → CI запускается
2. CI прогоняет evaluation на golden dataset с temperature=0
3. Если MAE > threshold или регрессия > X% → CI pipeline fails
4. Reviewer видит метрики в PR → принимает решение → merge → deploy

Это превращает prompt engineering из "попробовал, вроде лучше" в data-driven процесс с гарантиями качества.

---

## Справочник API

### Langfuse — `score()`

Langfuse позволяет привязывать числовые и категориальные оценки к traces и observations для онлайн evaluation.

```python
from langfuse import Langfuse

langfuse = Langfuse()
```

**Параметры `langfuse.score()`:**

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|:---:|:---:|----------|
| `name` | `str` | да | — | Имя метрики (`"mae"`, `"alignment"`, `"accuracy"`) |
| `value` | `float` | да | — | Числовое значение оценки |
| `trace_id` | `str` | да | — | ID trace, к которому привязывается score |
| `observation_id` | `str` | нет | `None` | ID конкретного observation внутри trace |
| `comment` | `str` | нет | `None` | Текстовый комментарий к оценке |
| `data_type` | `str` | нет | `"NUMERIC"` | `"NUMERIC"`, `"BOOLEAN"` или `"CATEGORICAL"` |
| `id` | `str` | нет | auto-generated | Кастомный ID для score |

**Пример:**

```python
langfuse.score(
    trace_id="trace-abc-123",
    name="mae",
    value=3.5,
    comment="MAE for essay assessment against golden dataset",
    data_type="NUMERIC",
)

langfuse.score(
    trace_id="trace-abc-123",
    name="quality_tier",
    value="acceptable",
    data_type="CATEGORICAL",
)
```

---

### ragas — faithfulness, answer_relevancy, context_precision

Библиотека RAGAS предоставляет стандартизированные метрики для RAG-систем.

```bash
pip install ragas
```

**Импорт метрик:**

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate, EvaluationDataset, SingleTurnSample
```

**`SingleTurnSample` — контейнер для одной записи:**

| Поле | Тип | Описание |
|------|-----|----------|
| `user_input` | `str` | Вопрос / запрос пользователя |
| `response` | `str` | Ответ системы |
| `retrieved_contexts` | `list[str]` | Найденные RAG-документы |
| `reference` | `str` | Эталонный ответ |

**`evaluate()` — запуск evaluation:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `dataset` | `EvaluationDataset` | Датасет с samples |
| `metrics` | `list[Metric]` | Список метрик для вычисления |

**Пример:**

```python
sample = SingleTurnSample(
    user_input="Evaluate this essay on climate change",
    response="Score: 72. The essay presents a clear thesis...",
    retrieved_contexts=["Rubric: Thesis & Argument — max 25 points..."],
    reference="Score: 75. The essay demonstrates strong argumentation..."
)

dataset = EvaluationDataset(samples=[sample])
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(results)
```

---

### LangSmith — `evaluate()`

LangSmith предоставляет hosted evaluation с визуализацией результатов и сравнением экспериментов.

```python
from langsmith import evaluate
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `target` | `Callable` | Функция `(inputs) → outputs` для evaluation |
| `data` | `str` | Имя dataset в LangSmith |
| `evaluators` | `list[Callable]` | Список evaluator-функций |
| `experiment_prefix` | `str` | Префикс для именования эксперимента |
| `max_concurrency` | `int` | Максимальный параллелизм (default: 5) |

**Evaluator-функция принимает `run` и `example`, возвращает `dict`:**

```python
def predict(inputs: dict) -> dict:
    result = chain.invoke(inputs)
    return {"output": result}

def mae_evaluator(run, example):
    predicted = run.outputs["output"]["overall_score"]
    expected = example.outputs["overall_score"]
    return {"key": "mae", "score": abs(predicted - expected)}

evaluate(
    target=predict,
    data="golden-assessment-dataset",
    evaluators=[mae_evaluator],
    experiment_prefix="prompt-v2",
    max_concurrency=3,
)
```

---

### numpy / scipy — статистические функции

**MAE и Within-N с numpy:**

```python
import numpy as np

expected = np.array([18, 22, 15, 8, 20])
predicted = np.array([17, 21, 18, 10, 19])

diffs = np.abs(predicted - expected)
mae = float(np.mean(diffs))
exact_match = float(np.mean(diffs == 0))
within_3 = float(np.mean(diffs <= 3))
within_5 = float(np.mean(diffs <= 5))
```

**Pearson correlation с scipy:**

```python
from scipy.stats import pearsonr

correlation, p_value = pearsonr(expected, predicted)
```

| Возвращаемое значение | Тип | Описание |
|----------------------|-----|----------|
| `correlation` | `float` | Коэффициент Пирсона [−1, 1]. > 0.7 — сильная положительная корреляция |
| `p_value` | `float` | p-значение. < 0.05 — статистически значимая корреляция |

---

### Pydantic — `model_validate` / `model_dump`

Сериализация и десериализация golden dataset через Pydantic v2.

**`Model.model_validate(data)` — dict/JSON → Pydantic model с валидацией:**

```python
import json

with open("data/golden/dataset.json") as f:
    raw = json.load(f)

entries = [GoldenEntry.model_validate(item) for item in raw]
```

**`model.model_dump()` — Pydantic model → dict:**

```python
data = entry.model_dump()
json_str = entry.model_dump_json(indent=2)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|:---:|----------|
| `mode` | `str` | `"python"` | `"python"` (dict с Python-типами) или `"json"` (JSON-совместимый dict) |
| `include` | `set` | `None` | Включить только указанные поля |
| `exclude` | `set` | `None` | Исключить указанные поля |
| `by_alias` | `bool` | `False` | Использовать alias имена полей |

---

## Практика: роутер `/api/v1/eval`

Создаём evaluation API, интегрированный в FastAPI-проект. Четыре эндпоинта: загрузка golden dataset, прогон eval pipeline, LLM-as-judge, A/B тест промптов.

Архитектура:

```
POST /api/v1/eval/golden-dataset  →  сохранить golden entry
POST /api/v1/eval/run             →  прогнать pipeline, вернуть метрики
POST /api/v1/eval/judge           →  LLM-as-judge для одной записи
POST /api/v1/eval/ab-test         →  сравнить два промпта
```

### Шаг 1: Схемы (`app/schemas/evaluation.py`)

Создаём Pydantic-модели для всех данных evaluation. `GoldenEntry` описывает структуру golden dataset (раздел 3 теории). `EvalReport` содержит агрегированные метрики (раздел 4). `JudgeResponse` — структурированный вывод LLM-as-judge (раздел 5) и одновременно `with_structured_output`-схема.

```python
from pydantic import BaseModel, Field

from app.schemas.assessment import AssessmentResponse


class GoldenEntryInput(BaseModel):
    student_work: str
    rubric_id: str = "essay_default"


class ExpectedCriterionScore(BaseModel):
    criterion_name: str
    score: int
    max_score: int


class ExpectedOutput(BaseModel):
    overall_score: int
    criterion_scores: list[ExpectedCriterionScore]


class GoldenEntryMetadata(BaseModel):
    quality_level: str = Field(description="strong / medium / weak / edge_case")
    description: str = ""


class GoldenEntry(BaseModel):
    id: str
    input: GoldenEntryInput
    expected_output: ExpectedOutput
    metadata: GoldenEntryMetadata


class EvalRunRequest(BaseModel):
    rubric_id: str = "essay_default"


class CriterionMetrics(BaseModel):
    criterion_name: str
    mae: float
    exact_match_rate: float
    within_3_rate: float
    within_5_rate: float


class EvalReport(BaseModel):
    total_entries: int
    overall_mae: float
    overall_exact_match_rate: float
    overall_within_5_rate: float
    pearson_correlation: float
    criterion_metrics: list[CriterionMetrics]


class JudgeRequest(BaseModel):
    system_output: AssessmentResponse
    expected_output: ExpectedOutput


class CriterionJudgment(BaseModel):
    criterion_name: str
    expected_score: int
    actual_score: int
    difference: int
    judgment: str


class JudgeResponse(BaseModel):
    alignment_score: int = Field(ge=0, le=100)
    criterion_analysis: list[CriterionJudgment]
    overall_judgment: str
    discrepancies: list[str]


class ABTestRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    rubric_id: str = "essay_default"


class PromptMetrics(BaseModel):
    prompt_label: str
    overall_mae: float
    overall_within_5_rate: float
    avg_criterion_mae: float
    pearson_correlation: float


class ABTestResponse(BaseModel):
    prompt_a_metrics: PromptMetrics
    prompt_b_metrics: PromptMetrics
    winner: str
    details: list[dict]
```

### Шаг 2: Сервис (`app/services/evaluation.py`)

Вся evaluation-логика вынесена в сервис: хранение golden entries, вычисление метрик, judge chain, A/B test. Роутер вызывает сервис, не содержит бизнес-логики.

Golden entries хранятся in-memory (module-level list). Для production стоит заменить на БД, но для обучения и экспериментов этого достаточно.

Директория `data/golden/` — для хранения JSON-файлов golden dataset в Git. Загрузка через API (`POST /golden-dataset`) или напрямую из файлов.

```python
import numpy as np
from scipy.stats import pearsonr

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.schemas.assessment import AssessmentResponse
from app.schemas.evaluation import (
    CriterionMetrics,
    EvalReport,
    ExpectedOutput,
    GoldenEntry,
    JudgeResponse,
    PromptMetrics,
)
from app.schemas.rubric import Rubric

_golden_store: list[GoldenEntry] = []

JUDGE_SYSTEM_PROMPT = (
    "You are an expert meta-assessor. Compare a student work assessment "
    "against an expert reference assessment. For each criterion, evaluate "
    "whether the system's score aligns with the expected score. "
    "Identify any significant discrepancies and explain why they matter."
)


def add_golden_entry(entry: GoldenEntry) -> None:
    _golden_store.append(entry)


def get_golden_entries() -> list[GoldenEntry]:
    return list(_golden_store)


def format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


async def run_eval_pipeline(
    chain: Runnable,
    rubric: Rubric,
    entries: list[GoldenEntry],
) -> EvalReport:
    rubric_text = format_rubric(rubric)
    predictions: list[AssessmentResponse] = []

    for entry in entries:
        result = await chain.ainvoke({
            "student_work": entry.input.student_work,
            "rubric": rubric_text,
        })
        predictions.append(result)

    criterion_names = [c.name for c in rubric.criteria]
    criterion_metrics = []

    for cname in criterion_names:
        expected_scores = []
        predicted_scores = []
        for entry, pred in zip(entries, predictions):
            exp_cs = next(
                (cs for cs in entry.expected_output.criterion_scores
                 if cs.criterion_name == cname),
                None,
            )
            pred_cs = next(
                (cs for cs in pred.criterion_scores if cs.criterion_name == cname),
                None,
            )
            if exp_cs and pred_cs:
                expected_scores.append(exp_cs.score)
                predicted_scores.append(pred_cs.score)

        if not expected_scores:
            continue

        exp_arr = np.array(expected_scores)
        pred_arr = np.array(predicted_scores)
        diffs = np.abs(pred_arr - exp_arr)

        criterion_metrics.append(CriterionMetrics(
            criterion_name=cname,
            mae=round(float(np.mean(diffs)), 2),
            exact_match_rate=round(float(np.mean(diffs == 0)), 3),
            within_3_rate=round(float(np.mean(diffs <= 3)), 3),
            within_5_rate=round(float(np.mean(diffs <= 5)), 3),
        ))

    overall_expected = np.array([e.expected_output.overall_score for e in entries])
    overall_predicted = np.array([p.overall_score for p in predictions])
    overall_diffs = np.abs(overall_predicted - overall_expected)

    if len(overall_expected) >= 2:
        corr, _ = pearsonr(overall_expected.tolist(), overall_predicted.tolist())
    else:
        corr = 0.0

    return EvalReport(
        total_entries=len(entries),
        overall_mae=round(float(np.mean(overall_diffs)), 2),
        overall_exact_match_rate=round(float(np.mean(overall_diffs == 0)), 3),
        overall_within_5_rate=round(float(np.mean(overall_diffs <= 5)), 3),
        pearson_correlation=round(float(corr), 3),
        criterion_metrics=criterion_metrics,
    )


async def run_judge(
    system_output: AssessmentResponse,
    expected_output: ExpectedOutput,
    llm: ChatAnthropic,
) -> JudgeResponse:
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human",
         "System assessment:\n{system_output}\n\n"
         "Expected assessment (expert reference):\n{expected_output}\n\n"
         "Analyze the alignment between these two assessments."),
    ])
    judge_chain = prompt | llm.with_structured_output(JudgeResponse)
    return await judge_chain.ainvoke({
        "system_output": system_output.model_dump_json(indent=2),
        "expected_output": expected_output.model_dump_json(indent=2),
    })


async def run_ab_test(
    prompt_a: str,
    prompt_b: str,
    entries: list[GoldenEntry],
    rubric: Rubric,
    llm: ChatAnthropic,
) -> tuple[PromptMetrics, PromptMetrics]:
    def build_chain(system_prompt: str) -> Runnable:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please assess the following student work:\n\n{student_work}"),
        ])
        return prompt | llm.with_structured_output(AssessmentResponse)

    chain_a = build_chain(prompt_a)
    chain_b = build_chain(prompt_b)

    report_a = await run_eval_pipeline(chain_a, rubric, entries)
    report_b = await run_eval_pipeline(chain_b, rubric, entries)

    def to_prompt_metrics(label: str, report: EvalReport) -> PromptMetrics:
        avg_mae = (
            sum(cm.mae for cm in report.criterion_metrics)
            / len(report.criterion_metrics)
            if report.criterion_metrics else 0.0
        )
        return PromptMetrics(
            prompt_label=label,
            overall_mae=report.overall_mae,
            overall_within_5_rate=report.overall_within_5_rate,
            avg_criterion_mae=round(avg_mae, 2),
            pearson_correlation=report.pearson_correlation,
        )

    return to_prompt_metrics("A", report_a), to_prompt_metrics("B", report_b)
```

Связь с теорией: `run_eval_pipeline` вычисляет все метрики из раздела 4 (MAE, exact match, within-N, Pearson). `run_judge` реализует reference-based LLM-as-judge из раздела 5 — используем `with_structured_output(JudgeResponse)` для получения структурированного анализа. `run_ab_test` — A/B тестирование из раздела 7: два промпта прогоняются на одном датасете, результаты сравниваются по всем метрикам.

### Шаг 3: Роутер (`app/api/v1/eval.py`)

Роутер связывает HTTP-эндпоинты с сервисными функциями через dependency injection. Каждый эндпоинт: валидация входных данных → вызов сервиса → возврат результата.

```python
from fastapi import APIRouter, HTTPException

from app.dependencies import ChainDep, LLMDep, RubricStoreDep
from app.schemas.evaluation import (
    ABTestRequest,
    ABTestResponse,
    EvalReport,
    EvalRunRequest,
    GoldenEntry,
    JudgeRequest,
    JudgeResponse,
)
from app.services.evaluation import (
    add_golden_entry,
    get_golden_entries,
    run_ab_test,
    run_eval_pipeline,
    run_judge,
)

router = APIRouter(prefix="/eval", tags=["lesson-9-eval"])


@router.post("/golden-dataset")
async def upload_golden_entry(entry: GoldenEntry) -> GoldenEntry:
    add_golden_entry(entry)
    return entry


@router.post("/run")
async def run_evaluation(
    request: EvalRunRequest,
    chain: ChainDep,
    rubrics: RubricStoreDep,
) -> EvalReport:
    entries = get_golden_entries()
    if not entries:
        raise HTTPException(status_code=400, detail="No golden entries uploaded")
    rubric = rubrics.get(request.rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail=f"Rubric '{request.rubric_id}' not found")
    return await run_eval_pipeline(chain, rubric, entries)


@router.post("/judge")
async def judge_output(
    request: JudgeRequest,
    llm: LLMDep,
) -> JudgeResponse:
    return await run_judge(request.system_output, request.expected_output, llm)


@router.post("/ab-test")
async def ab_test(
    request: ABTestRequest,
    rubrics: RubricStoreDep,
    llm: LLMDep,
) -> ABTestResponse:
    entries = get_golden_entries()
    if not entries:
        raise HTTPException(status_code=400, detail="No golden entries uploaded")
    rubric = rubrics.get(request.rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail=f"Rubric '{request.rubric_id}' not found")

    metrics_a, metrics_b = await run_ab_test(
        request.prompt_a, request.prompt_b, entries, rubric, llm,
    )
    winner = "A" if metrics_a.overall_mae <= metrics_b.overall_mae else "B"

    return ABTestResponse(
        prompt_a_metrics=metrics_a,
        prompt_b_metrics=metrics_b,
        winner=winner,
        details=[
            {
                "metric": "overall_mae",
                "prompt_a": metrics_a.overall_mae,
                "prompt_b": metrics_b.overall_mae,
                "winner": "A" if metrics_a.overall_mae <= metrics_b.overall_mae else "B",
            },
            {
                "metric": "within_5_rate",
                "prompt_a": metrics_a.overall_within_5_rate,
                "prompt_b": metrics_b.overall_within_5_rate,
                "winner": "A" if metrics_a.overall_within_5_rate >= metrics_b.overall_within_5_rate else "B",
            },
            {
                "metric": "pearson_correlation",
                "prompt_a": metrics_a.pearson_correlation,
                "prompt_b": metrics_b.pearson_correlation,
                "winner": "A" if metrics_a.pearson_correlation >= metrics_b.pearson_correlation else "B",
            },
        ],
    )
```

### Шаг 4: Регистрация в `app/api/router.py`

Добавляем eval-роутер к существующему api_router:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, eval, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(eval.router)
```

Также создайте директорию `data/golden/` для хранения golden dataset файлов:

```bash
mkdir -p data/golden
```

### Шаг 5: Тестирование

Запускаем сервер и тестируем каждый эндпоинт.

```bash
uvicorn app.main:app --reload
```

**1. Загрузка golden entry:**

```bash
curl -X POST http://localhost:8000/api/v1/eval/golden-dataset \
  -H "Content-Type: application/json" \
  -d '{
    "id": "golden_001",
    "input": {
      "student_work": "Climate change represents one of the most significant challenges facing humanity today. Multiple studies from the IPCC and NASA confirm that global temperatures have risen by 1.1 degrees Celsius since pre-industrial times. This essay argues that carbon taxation is the most effective policy mechanism to reduce emissions, supported by evidence from the European Union Emissions Trading System and the British Columbia carbon tax. However, the economic impact on developing nations requires careful consideration and graduated implementation timelines.",
      "rubric_id": "essay_default"
    },
    "expected_output": {
      "overall_score": 72,
      "criterion_scores": [
        {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25},
        {"criterion_name": "Evidence & Support", "score": 19, "max_score": 25},
        {"criterion_name": "Structure & Organization", "score": 15, "max_score": 20},
        {"criterion_name": "Critical Thinking", "score": 12, "max_score": 20},
        {"criterion_name": "Language & Style", "score": 8, "max_score": 10}
      ]
    },
    "metadata": {
      "quality_level": "medium",
      "description": "Decent essay on climate policy with some citations"
    }
  }'
```

**2. Запуск evaluation pipeline:**

```bash
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"rubric_id": "essay_default"}'
```

Ответ содержит `EvalReport` с метриками по каждому критерию и overall. Для надёжных результатов загрузите 20+ golden entries перед запуском.

**3. LLM-as-judge:**

```bash
curl -X POST http://localhost:8000/api/v1/eval/judge \
  -H "Content-Type: application/json" \
  -d '{
    "system_output": {
      "overall_score": 68,
      "max_overall_score": 100,
      "criterion_scores": [
        {"criterion_name": "Thesis & Argument", "score": 16, "max_score": 25, "feedback": "Clear thesis but lacks depth in reasoning"},
        {"criterion_name": "Evidence & Support", "score": 17, "max_score": 25, "feedback": "Some citations provided but insufficient"},
        {"criterion_name": "Structure & Organization", "score": 14, "max_score": 20, "feedback": "Logical flow with minor gaps"},
        {"criterion_name": "Critical Thinking", "score": 13, "max_score": 20, "feedback": "Surface-level analysis"},
        {"criterion_name": "Language & Style", "score": 8, "max_score": 10, "feedback": "Academic tone maintained"}
      ],
      "summary": "Average essay with room for improvement in evidence and analysis",
      "strengths": ["Clear thesis statement", "Academic tone"],
      "improvements": ["Add more citations", "Deeper critical analysis"]
    },
    "expected_output": {
      "overall_score": 72,
      "criterion_scores": [
        {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25},
        {"criterion_name": "Evidence & Support", "score": 19, "max_score": 25},
        {"criterion_name": "Structure & Organization", "score": 15, "max_score": 20},
        {"criterion_name": "Critical Thinking", "score": 12, "max_score": 20},
        {"criterion_name": "Language & Style", "score": 8, "max_score": 10}
      ]
    }
  }'
```

**4. A/B тест промптов:**

Оба промпта должны содержать `{rubric}` как placeholder — сервис подставит текст рубрики при вызове chain.

```bash
curl -X POST http://localhost:8000/api/v1/eval/ab-test \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_a": "You are an expert academic assessor. Evaluate student work against the rubric:\n{rubric}\n\nProvide scores and specific feedback for each criterion. The overall_score is the sum of all criterion scores.",
    "prompt_b": "You are a strict academic evaluator. Analyze each criterion step by step:\n{rubric}\n\nFor each criterion: (1) quote specific evidence from the text, (2) identify strengths, (3) identify weaknesses, (4) assign a score. The overall_score must equal the sum of criterion scores.",
    "rubric_id": "essay_default"
  }'
```

Ответ содержит метрики для обоих промптов и winner — промпт с меньшим MAE.

---

## Чеклист самопроверки

- [ ] Зачем golden dataset? Почему LLM-сгенерированные эталоны = circular evaluation?
- [ ] Объясни MAE и Within-N. Какой MAE "хороший" для критерия с max=25?
- [ ] Три паттерна LLM-as-judge: pointwise, pairwise, reference-based — когда какой?
- [ ] Какие ограничения у LLM-as-judge? Назови три bias-а
- [ ] Почему A/B тест на 3 примерах ненадёжен?
- [ ] Что такое regression testing для промптов? Как интегрировать в CI/CD?
- [ ] Как Pydantic `model_validate`/`model_dump` помогают с golden dataset?
- [ ] Чем отличается офлайн evaluation от онлайн?

---

## Частые ошибки

### 1. Golden dataset, размеченный LLM

```python
expected = await llm.ainvoke("Score this essay")
```

LLM оценивает → LLM проверяет = circular evaluation. Alignment будет искусственно высоким — модель согласна сама с собой. Эталоны должны быть от человека-эксперта.

### 2. A/B тест на 3 примерах

```python
dataset = golden_dataset[:3]
```

На 3 примерах любая разница в метриках — шум, а не сигнал. Минимум 20 примеров, оптимально 50+.

### 3. Один LLM как judge и subject

```python
subject = ChatAnthropic(model="claude-sonnet-4-20250514")
judge = ChatAnthropic(model="claude-sonnet-4-20250514")
```

Self-bias: модель предвзята к собственным ответам. Используйте разные модели или хотя бы `temperature=0` для judge.

### 4. Метрики только по overall_score

```python
mae = abs(predicted.overall_score - expected.overall_score)
```

Агрегированная MAE скрывает проблемы по отдельным критериям. Модель может быть точной для "Thesis" (MAE=2) но слабой для "Critical Thinking" (MAE=8). Всегда считайте per-criterion.

### 5. Evaluation без temperature=0

```python
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
```

При `temperature > 0` модель даёт разные ответы на один вопрос. Метрики evaluation будут нестабильными. Используйте `temperature=0` для воспроизводимости.

---

## Что читать дальше

- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — hosted evaluation с визуализацией и сравнением экспериментов
- [Langfuse Scores](https://langfuse.com/docs/scores/overview) — привязка оценок к traces для онлайн мониторинга
- [RAGAS Documentation](https://docs.ragas.io/) — стандартизированные метрики для RAG-систем
- [Paper: "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023) — исследование надёжности и bias-ов LLM-as-judge
- [Anthropic Evaluation Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests) — рекомендации по evaluation от Anthropic

**Следующая тема:** [Тема 10: Production-паттерны](topic_10_production_patterns.md) — model routing, guardrails, semantic cache, cost optimization.

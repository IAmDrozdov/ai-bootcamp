# Тема 9: Evaluation

> **Пререквизиты:** [Тема 1–4](topic_01_prompt_engineering.md), [Тема 3 (Structured Output)](topic_03_structured_output.md), рекомендуется [Тема 5 (RAG)](topic_05_rag.md), [Тема 8 (Observability)](topic_08_observability.md)  
> **Зависимости:** `langchain-anthropic`, `langfuse`, `ragas`, `langsmith`, `numpy`, `scipy`

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

## Практика

Все примеры — самодостаточные: определяют данные и модели прямо в ячейке, не требуют внешних импортов из проекта.

### Пример 1: Golden dataset

Создаём Pydantic-модели для golden dataset и загружаем записи. Каждая запись содержит входные данные, эталонные оценки от эксперта и метаданные.

```python
import json
from pydantic import BaseModel, Field


class ExpectedCriterionScore(BaseModel):
    criterion_name: str
    score: int
    max_score: int


class ExpectedOutput(BaseModel):
    overall_score: int
    criterion_scores: list[ExpectedCriterionScore]


class GoldenEntryInput(BaseModel):
    student_work: str
    rubric_id: str = "essay_default"


class GoldenEntryMetadata(BaseModel):
    quality_level: str = Field(description="strong / medium / weak / edge_case")
    description: str = ""


class GoldenEntry(BaseModel):
    id: str
    input: GoldenEntryInput
    expected_output: ExpectedOutput
    metadata: GoldenEntryMetadata


golden_data = [
    {
        "id": "golden_001",
        "input": {
            "student_work": (
                "Climate change represents one of the most significant challenges "
                "facing humanity today. Multiple studies from the IPCC and NASA confirm "
                "that global temperatures have risen by 1.1 degrees Celsius since "
                "pre-industrial times. This essay argues that carbon taxation is the "
                "most effective policy mechanism to reduce emissions, supported by "
                "evidence from the EU ETS and the British Columbia carbon tax."
            ),
            "rubric_id": "essay_default",
        },
        "expected_output": {
            "overall_score": 72,
            "criterion_scores": [
                {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25},
                {"criterion_name": "Evidence & Support", "score": 19, "max_score": 25},
                {"criterion_name": "Structure & Organization", "score": 15, "max_score": 20},
                {"criterion_name": "Critical Thinking", "score": 12, "max_score": 20},
                {"criterion_name": "Language & Style", "score": 8, "max_score": 10},
            ],
        },
        "metadata": {"quality_level": "medium", "description": "Decent essay on climate policy"},
    },
    {
        "id": "golden_002",
        "input": {
            "student_work": "AI is cool. It does stuff. The end.",
            "rubric_id": "essay_default",
        },
        "expected_output": {
            "overall_score": 25,
            "criterion_scores": [
                {"criterion_name": "Thesis & Argument", "score": 5, "max_score": 25},
                {"criterion_name": "Evidence & Support", "score": 3, "max_score": 25},
                {"criterion_name": "Structure & Organization", "score": 8, "max_score": 20},
                {"criterion_name": "Critical Thinking", "score": 5, "max_score": 20},
                {"criterion_name": "Language & Style", "score": 4, "max_score": 10},
            ],
        },
        "metadata": {"quality_level": "weak", "description": "Minimal effort, no evidence"},
    },
]

entries = [GoldenEntry.model_validate(item) for item in golden_data]

for e in entries:
    print(f"{e.id}: overall={e.expected_output.overall_score}, "
          f"quality={e.metadata.quality_level}")
    for cs in e.expected_output.criterion_scores:
        print(f"  {cs.criterion_name}: {cs.score}/{cs.max_score}")

print(f"\nСериализация в JSON:\n{entries[0].model_dump_json(indent=2)[:200]}...")
```

### Пример 2: Eval pipeline — вычисление метрик

Имитируем прогон assessment-цепочки на golden dataset: для каждой записи получаем предсказания модели (здесь захардкожены для демонстрации), затем считаем MAE, exact match, within-N и Pearson correlation.

```python
import numpy as np
from scipy.stats import pearsonr
from pydantic import BaseModel


class CriterionResult(BaseModel):
    criterion_name: str
    expected: list[int]
    predicted: list[int]


class EvalReport(BaseModel):
    total_entries: int
    overall_mae: float
    overall_exact_match_rate: float
    overall_within_5_rate: float
    pearson_correlation: float
    criterion_reports: list[dict]


criterion_names = [
    "Thesis & Argument",
    "Evidence & Support",
    "Structure & Organization",
    "Critical Thinking",
    "Language & Style",
]

expected_overall = np.array([72, 25, 88, 55, 41])
predicted_overall = np.array([68, 30, 85, 60, 38])

expected_by_criterion = {
    "Thesis & Argument":        np.array([18, 5, 22, 14, 10]),
    "Evidence & Support":       np.array([19, 3, 23, 12, 8]),
    "Structure & Organization": np.array([15, 8, 18, 13, 10]),
    "Critical Thinking":        np.array([12, 5, 17, 10, 7]),
    "Language & Style":         np.array([8,  4,  8,  6, 6]),
}

predicted_by_criterion = {
    "Thesis & Argument":        np.array([16, 7, 21, 15, 9]),
    "Evidence & Support":       np.array([17, 5, 22, 14, 7]),
    "Structure & Organization": np.array([14, 9, 17, 12, 11]),
    "Critical Thinking":        np.array([13, 4, 16, 11, 6]),
    "Language & Style":         np.array([8,  5,  9,  8, 5]),
}

overall_diffs = np.abs(predicted_overall - expected_overall)
overall_mae = float(np.mean(overall_diffs))
overall_exact_match = float(np.mean(overall_diffs == 0))
overall_within_5 = float(np.mean(overall_diffs <= 5))
corr, p_value = pearsonr(expected_overall.tolist(), predicted_overall.tolist())

print(f"Overall MAE:          {overall_mae:.2f}")
print(f"Overall Exact Match:  {overall_exact_match:.1%}")
print(f"Overall Within-5:     {overall_within_5:.1%}")
print(f"Pearson correlation:  {corr:.3f} (p={p_value:.4f})")
print()

for cname in criterion_names:
    exp = expected_by_criterion[cname]
    pred = predicted_by_criterion[cname]
    diffs = np.abs(pred - exp)

    mae = float(np.mean(diffs))
    exact = float(np.mean(diffs == 0))
    w3 = float(np.mean(diffs <= 3))
    w5 = float(np.mean(diffs <= 5))

    print(f"{cname:30s}  MAE={mae:.2f}  exact={exact:.0%}  "
          f"within-3={w3:.0%}  within-5={w5:.0%}")
```

### Пример 3: LLM-as-judge

Reference-based LLM-as-judge: передаём системные и эталонные оценки judge-модели, получаем структурированный анализ alignment через `with_structured_output`.

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


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


JUDGE_SYSTEM_PROMPT = (
    "You are an expert meta-assessor. Compare a student work assessment "
    "against an expert reference assessment. For each criterion, evaluate "
    "whether the system's score aligns with the expected score. "
    "Identify any significant discrepancies (|diff| > 5) and explain why "
    "they matter."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("human",
     "System assessment:\n{system_output}\n\n"
     "Expected assessment (expert reference):\n{expected_output}\n\n"
     "Analyze the alignment between these two assessments."),
])

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
judge_chain = prompt | llm.with_structured_output(JudgeResponse)

system_output = {
    "overall_score": 68,
    "criterion_scores": [
        {"criterion_name": "Thesis & Argument", "score": 16, "max_score": 25,
         "feedback": "Clear thesis but lacks depth"},
        {"criterion_name": "Evidence & Support", "score": 17, "max_score": 25,
         "feedback": "Some citations but insufficient"},
        {"criterion_name": "Structure & Organization", "score": 14, "max_score": 20,
         "feedback": "Logical flow with minor gaps"},
        {"criterion_name": "Critical Thinking", "score": 13, "max_score": 20,
         "feedback": "Surface-level analysis"},
        {"criterion_name": "Language & Style", "score": 8, "max_score": 10,
         "feedback": "Academic tone maintained"},
    ],
}

expected_output = {
    "overall_score": 72,
    "criterion_scores": [
        {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25},
        {"criterion_name": "Evidence & Support", "score": 19, "max_score": 25},
        {"criterion_name": "Structure & Organization", "score": 15, "max_score": 20},
        {"criterion_name": "Critical Thinking", "score": 12, "max_score": 20},
        {"criterion_name": "Language & Style", "score": 8, "max_score": 10},
    ],
}

import json
result = judge_chain.invoke({
    "system_output": json.dumps(system_output, indent=2),
    "expected_output": json.dumps(expected_output, indent=2),
})

print(f"Alignment score: {result.alignment_score}/100")
print(f"Overall judgment: {result.overall_judgment}")
print()
for ca in result.criterion_analysis:
    print(f"  {ca.criterion_name}: expected={ca.expected_score} actual={ca.actual_score} "
          f"diff={ca.difference} — {ca.judgment}")
if result.discrepancies:
    print(f"\nDiscrepancies:")
    for d in result.discrepancies:
        print(f"  - {d}")
```

### Пример 4: A/B тестирование промптов

Два промпта прогоняются на одном golden dataset, метрики сравниваются. Здесь используем реальные LLM-вызовы: каждый промпт оценивает все записи, затем считаем MAE и выбираем winner.

```python
import json
import numpy as np
from scipy.stats import pearsonr
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class CriterionScore(BaseModel):
    criterion_name: str
    score: int
    max_score: int
    feedback: str


class AssessmentResult(BaseModel):
    overall_score: int
    criterion_scores: list[CriterionScore]
    summary: str


golden_entries = [
    {
        "student_work": (
            "Climate change represents one of the most significant challenges "
            "facing humanity today. Multiple studies from the IPCC confirm that "
            "global temperatures have risen by 1.1°C since pre-industrial times."
        ),
        "expected_overall": 72,
    },
    {
        "student_work": "AI is cool. It does stuff. The end.",
        "expected_overall": 25,
    },
    {
        "student_work": (
            "The French Revolution of 1789 fundamentally transformed European "
            "political structures. Through analysis of primary sources including "
            "the Declaration of the Rights of Man and contemporaneous parliamentary "
            "records, this essay demonstrates that economic inequality was the "
            "primary catalyst, while Enlightenment philosophy provided the "
            "intellectual framework for revolutionary action."
        ),
        "expected_overall": 88,
    },
]

RUBRIC_TEXT = """Rubric: Essay Assessment
- Thesis & Argument (max 25): Clear thesis with logical argumentation
- Evidence & Support (max 25): Use of citations and supporting evidence
- Structure & Organization (max 20): Logical flow and paragraph structure
- Critical Thinking (max 20): Depth of analysis and original insight
- Language & Style (max 10): Academic tone, grammar, vocabulary"""

prompt_a_text = (
    "You are an expert academic assessor. Evaluate student work against "
    "the rubric:\n{rubric}\n\nProvide scores and specific feedback for "
    "each criterion. The overall_score is the sum of all criterion scores."
)

prompt_b_text = (
    "You are a strict academic evaluator. Analyze each criterion step by "
    "step:\n{rubric}\n\nFor each criterion: (1) quote specific evidence "
    "from the text, (2) identify strengths, (3) identify weaknesses, "
    "(4) assign a score. The overall_score must equal the sum of criterion scores."
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Rubric:\n{rubric}\n\nStudent work:\n{student_work}"),
    ])
    return prompt | llm.with_structured_output(AssessmentResult)


chain_a = build_chain(prompt_a_text)
chain_b = build_chain(prompt_b_text)

expected_scores = np.array([e["expected_overall"] for e in golden_entries])

for label, chain in [("A", chain_a), ("B", chain_b)]:
    predicted = []
    for entry in golden_entries:
        result = chain.invoke({
            "rubric": RUBRIC_TEXT,
            "student_work": entry["student_work"],
        })
        predicted.append(result.overall_score)

    pred_arr = np.array(predicted)
    diffs = np.abs(pred_arr - expected_scores)
    mae = float(np.mean(diffs))
    within_5 = float(np.mean(diffs <= 5))

    if len(expected_scores) >= 2:
        corr, _ = pearsonr(expected_scores.tolist(), pred_arr.tolist())
    else:
        corr = 0.0

    print(f"Prompt {label}:  MAE={mae:.2f}  Within-5={within_5:.0%}  "
          f"Pearson r={corr:.3f}")
    for i, entry in enumerate(golden_entries):
        print(f"  [{entry['expected_overall']}] expected  vs  "
              f"[{predicted[i]}] predicted  (diff={diffs[i]})")
    print()
```

### Пример 5: Вычисление метрик (MAE, exact match, within-N, Pearson)

Функции для подсчёта всех основных evaluation-метрик. Работают с любыми числовыми массивами оценок.

```python
import numpy as np
from scipy.stats import pearsonr


def compute_metrics(expected: list[int], predicted: list[int]) -> dict:
    exp = np.array(expected)
    pred = np.array(predicted)
    diffs = np.abs(pred - exp)

    mae = float(np.mean(diffs))
    exact_match = float(np.mean(diffs == 0))
    within_3 = float(np.mean(diffs <= 3))
    within_5 = float(np.mean(diffs <= 5))

    if len(exp) >= 2:
        corr, p_value = pearsonr(exp.tolist(), pred.tolist())
    else:
        corr, p_value = 0.0, 1.0

    return {
        "mae": round(mae, 2),
        "exact_match_rate": round(exact_match, 3),
        "within_3_rate": round(within_3, 3),
        "within_5_rate": round(within_5, 3),
        "pearson_r": round(float(corr), 3),
        "p_value": round(float(p_value), 4),
    }


expected_thesis = [18, 5, 22, 14, 10, 20, 8, 16, 23, 12]
predicted_thesis = [16, 7, 21, 15, 9, 18, 10, 14, 22, 13]

expected_evidence = [19, 3, 23, 12, 8, 21, 6, 17, 24, 10]
predicted_evidence = [17, 5, 22, 14, 7, 19, 8, 15, 23, 11]

expected_overall = [72, 25, 88, 55, 41, 80, 30, 65, 90, 48]
predicted_overall = [68, 30, 85, 60, 38, 76, 34, 62, 87, 50]

for name, exp, pred in [
    ("Thesis & Argument", expected_thesis, predicted_thesis),
    ("Evidence & Support", expected_evidence, predicted_evidence),
    ("Overall Score", expected_overall, predicted_overall),
]:
    m = compute_metrics(exp, pred)
    print(f"{name}:")
    print(f"  MAE={m['mae']}  Exact={m['exact_match_rate']:.0%}  "
          f"Within-3={m['within_3_rate']:.0%}  Within-5={m['within_5_rate']:.0%}  "
          f"r={m['pearson_r']} (p={m['p_value']})")
```

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

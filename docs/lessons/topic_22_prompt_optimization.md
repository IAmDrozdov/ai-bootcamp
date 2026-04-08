# Тема 22: Автоматическая оптимизация промптов — GEPA и TensorZero

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 9](topic_09_evaluation.md)
> **Зависимости:** `gepa`, `tensorzero`, `langchain-anthropic`, `dspy`

---

## Теория

### 1. Зачем автоматизировать оптимизацию промптов

Ручной промпт-инжиниринг — это итеративный процесс: написал промпт → протестировал → поправил → снова протестировал. На каждую итерацию уходит от минут до часов. При этом:

- **Человек не может перебрать тысячи вариантов.** Один промпт-инженер тестирует 5-20 вариаций. Автоматический оптимизатор может проверить сотни.
- **Человек не видит паттерны ошибок.** На 50 тестовых примерах сложно заметить, что модель систематически ошибается в определённом типе задач. Алгоритм анализирует все ошибки.
- **Промпт — это не код.** Небольшое изменение формулировки может кардинально изменить поведение модели. "Пошаговый анализ" и "рассуждай шаг за шагом" — это разные инструкции для LLM.
- **A/B тесты промптов дороги.** Каждый вариант нужно прогнать на десятках примеров, каждый прогон стоит денег (API calls).

Два подхода к автоматизации:

| Характеристика | Gradient-free (GEPA) | RL / Fine-tuning |
|---|---|---|
| Что оптимизируется | Текст промпта | Веса модели |
| Доступ к модели | Только API (чёрный ящик) | Нужны веса или fine-tuning API |
| Количество evaluations | 100-500 | 5,000-25,000+ |
| Стоимость | $5-50 за оптимизацию | $50-500+ |
| Интерпретируемость | Высокая — промпт читаем | Низкая — изменения в весах |
| Скорость | Часы | Дни |

### 2. GEPA — Reflective Prompt Evolution

**GEPA** (Genetic-Pareto) — фреймворк для оптимизации любых текстовых артефактов (промптов, кода, конфигураций) через эволюционный поиск с LLM-рефлексией.

#### Ключевая идея

Традиционные оптимизаторы знают **что** кандидат провалился (низкий score), но не знают **почему**. GEPA использует LLM для анализа полных execution traces — сообщений об ошибках, логов рассуждений, профилировочных данных — чтобы диагностировать причины провала и предложить целенаправленные исправления.

#### Алгоритм

```
1. SELECT  — выбрать кандидата с Pareto-фронтира
2. EXECUTE — прогнать на минибатче, сохраняя полные traces
3. REFLECT — LLM читает traces, диагностирует ошибки
4. MUTATE  — сгенерировать улучшенного кандидата на основе диагностики
5. ACCEPT  — добавить в пул, если улучшился; обновить Pareto-фронтир
```

#### Pareto-фронтир

Вместо одного "лучшего" промпта GEPA поддерживает множество кандидатов — **Pareto-фронтир**. Каждый кандидат оптимален для разного подмножества задач. Это предотвращает "забывание" — улучшение на одних примерах за счёт деградации на других.

```
Score на задачах типа A
▲
│     ●  candidate_1 (хорош на A, слаб на B)
│   ● candidate_3 (баланс)
│ ●  candidate_2 (хорош на B, слаб на A)
└──────────────────────► Score на задачах типа B
```

#### Actionable Side Information (ASI)

ASI — ключевая концепция GEPA. Это диагностический фидбек, возвращаемый evaluator'ом: сообщения об ошибках, выводы, промежуточные результаты. ASI играет для текстовой оптимизации ту же роль, что градиент для числовой — указывает **направление** улучшения.

```python
import gepa.optimize_anything as oa

def evaluate(candidate: str) -> float:
    result = run_system(candidate)
    oa.log(f"Output: {result.output}")
    oa.log(f"Error: {result.error}")
    oa.log(f"Expected: {result.expected}")
    return result.score
```

LLM-рефлексия читает эти логи и понимает: "кандидат ошибается, потому что не учитывает edge case X → нужно добавить инструкцию Y".

#### Три API

| API | Что оптимизирует | Когда использовать |
|---|---|---|
| `gepa.optimize()` | System prompt | Простая оптимизация промпта для LLM-задачи |
| `gepa.optimize_anything()` | Любой текст | Код, конфигурации, agent architectures, SVG |
| `dspy.GEPA` | DSPy-программу | Сложные multi-step пайплайны |

### 3. TensorZero — LLM-платформа с оптимизацией

**TensorZero** — open-source платформа для LLM-приложений, объединяющая inference gateway, observability, evaluation и optimization в одном инструменте.

#### Архитектура

```
┌─────────────────────────────────────────────────┐
│                  Твоё приложение                 │
│              (Python, Node, HTTP)                │
└─────────────────┬───────────────────────────────┘
                  │ OpenAI-compatible API
┌─────────────────▼───────────────────────────────┐
│           TensorZero Gateway (Rust)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Routing  │ │ A/B test │ │ Fallbacks/Retry  │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │Templates │ │ Schemas  │ │  Observability   │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└───────┬──────────────┬──────────────┬───────────┘
        │              │              │
   ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
   │Anthropic│   │  OpenAI   │  │  Gemini │  ...
   └─────────┘   └───────────┘  └─────────┘
                  │
           ┌──────▼──────┐
           │  PostgreSQL  │  ← inference traces,
           │  (хранилище) │    feedback, metrics
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │ Optimization │  ← SFT, DPO, GEPA,
           │   Recipes    │    DICL, Best-of-N
           └─────────────┘
```

#### Ключевые концепции

**Function** — задача, которую решает LLM (генерация хайку, классификация, оценка эссе). Определяется в конфигурации с input/output схемами.

**Variant** — одна реализация функции: конкретный промпт + модель + параметры. У одной функции может быть несколько вариантов для A/B тестирования.

**Episode** — связанная последовательность inference-вызовов (multi-step workflow). Feedback привязывается к episode для end-to-end оптимизации.

**Feedback** — метрики и текстовый фидбек, собираемые после inference. Используются для оптимизации (fine-tuning, prompt evolution).

#### Конфигурация (`tensorzero.toml`)

TensorZero использует TOML-файл для декларативной конфигурации:

```toml
[functions.assess_essay]
type = "chat"

[functions.assess_essay.variants.claude_sonnet]
type = "chat_completion"
model = "anthropic::claude-sonnet-4-20250514"
system_template = "prompts/assess_essay/system.txt"
user_template = "prompts/assess_essay/user.txt"

[functions.assess_essay.variants.gpt4o]
type = "chat_completion"
model = "openai::gpt-4o"
system_template = "prompts/assess_essay/system.txt"
user_template = "prompts/assess_essay/user.txt"
weight = 0
```

Два варианта одной функции. `weight = 0` означает, что GPT-4o не получает трафик (можно включить для A/B теста, изменив weight).

#### Gateway — производительность

Gateway написан на Rust. Benchmark-результаты:

| Нагрузка | TensorZero P99 | LiteLLM P99 |
|---|---|---|
| 100 QPS | 0.33ms | 8.3ms |
| 1,000 QPS | 0.45ms | Не выдерживает |
| 10,000 QPS | 0.94ms | Не выдерживает |

Overhead gateway'я <1ms даже под экстремальной нагрузкой. Для сравнения, один LLM-вызов занимает 1,000-30,000ms.

#### Workflows оптимизации

TensorZero предоставляет готовые рецепты оптимизации:

| Категория | Метод | Описание |
|---|---|---|
| **Model** | SFT | Supervised fine-tuning на собранных данных |
| **Model** | DPO | Preference optimization (пары "лучше/хуже") |
| **Model** | RFT | Reinforcement fine-tuning |
| **Prompt** | GEPA | Эволюционная оптимизация промптов (встроен GEPA) |
| **Inference** | Best-of-N | Генерация N ответов, выбор лучшего |
| **Inference** | Mixture-of-N | Best-of-N через разные модели |
| **Inference** | DICL | Dynamic In-Context Learning — автоматический подбор few-shot примеров |

### 4. GEPA vs TensorZero — когда что

| Сценарий | GEPA | TensorZero |
|---|---|---|
| Быстро улучшить промпт | Да — `gepa.optimize()` | Нет (нужен deployed gateway) |
| Production LLM gateway | Нет (не gateway) | Да — основная функция |
| A/B тесты промптов | Нет (офлайн-оптимизация) | Да — встроено |
| Fine-tuning моделей | Нет | Да — SFT, DPO, RFT |
| Оптимизация кода/конфигов | Да — `optimize_anything()` | Нет (только промпты через GEPA) |
| Observability | Нет | Да — полный трейсинг |
| Inference-time оптимизация | Нет | Да — Best-of-N, DICL |

**Типичная стратегия:** используй GEPA для офлайн-оптимизации промптов (до деплоя), а TensorZero — для production inference с A/B тестами, мониторингом и дальнейшей оптимизацией (fine-tuning, DICL).

### 5. DSPy + GEPA — оптимизация сложных пайплайнов

DSPy — фреймворк для программирования LLM-пайплайнов как декларативных программ. GEPA интегрирован в DSPy как оптимизатор `dspy.GEPA`.

Преимущества DSPy + GEPA:

- **Оптимизация всей программы**, а не одного промпта. Если у тебя chain из 3 LLM-вызовов, GEPA оптимизирует все инструкции одновременно.
- **Автоматическая генерация промптов** из сигнатур. Ты описываешь `"question -> answer"`, DSPy генерирует промпт, GEPA его улучшает.
- **Модульность.** Замена модели, добавление Chain-of-Thought — одна строка кода.

```python
import dspy

class AssessmentPipeline(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("essay, rubric -> analysis")
        self.score = dspy.ChainOfThought("essay, analysis, rubric -> scores, feedback")

    def forward(self, essay, rubric):
        analysis = self.analyze(essay=essay, rubric=rubric)
        return self.score(essay=essay, analysis=analysis.analysis, rubric=rubric)
```

GEPA оптимизирует инструкции для обоих модулей (`analyze` и `score`) совместно, учитывая end-to-end метрику.

---

## API Reference

### gepa.optimize()

```python
gepa.optimize(
    seed_candidate: dict[str, str],
    trainset: list[dict],
    valset: list[dict] | None = None,
    task_lm: str = "openai/gpt-4o-mini",
    reflection_lm: str = "openai/gpt-4o",
    max_metric_calls: int = 100,
    candidate_selection_strategy: str = "pareto",
    stop_callbacks: list | None = None,
    display_progress_bar: bool = True,
) -> GEPAResult
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `seed_candidate` | `dict[str, str]` | — | Начальный промпт (ключ → текст) |
| `trainset` | `list[dict]` | — | Обучающие примеры |
| `valset` | `list[dict]` | `None` | Валидационные примеры (для generalization) |
| `task_lm` | `str` | `"openai/gpt-4o-mini"` | Модель для выполнения задачи |
| `reflection_lm` | `str` | `"openai/gpt-4o"` | Модель для рефлексии (рекомендуется сильная) |
| `max_metric_calls` | `int` | `100` | Бюджет evaluations |
| `candidate_selection_strategy` | `str` | `"pareto"` | Стратегия выбора: `pareto`, `current_best`, `epsilon_greedy` |

**Возвращает** `GEPAResult`:

```python
result.best_candidate     # dict[str, str] — лучший промпт
result.best_score         # float — score лучшего кандидата
result.pareto_frontier    # list — кандидаты на Pareto-фронтире
result.history            # list — история оптимизации
```

### gepa.optimize_anything()

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

optimize_anything(
    seed_candidate: str | None = None,
    evaluator: Callable[[str], float | tuple[float, dict]],
    objective: str | None = None,
    config: GEPAConfig = GEPAConfig(),
) -> GEPAResult
```

| Параметр | Тип | Описание |
|---|---|---|
| `seed_candidate` | `str \| None` | Начальный текстовый артефакт (или None — GEPA сгенерирует) |
| `evaluator` | `Callable` | Функция оценки: возвращает score или (score, side_info) |
| `objective` | `str \| None` | Описание цели оптимизации |
| `config` | `GEPAConfig` | Конфигурация (бюджет, модели, стратегии) |

### dspy.GEPA

```python
optimizer = dspy.GEPA(
    metric: Callable,
    reflection_lm: dspy.LM | None = None,
    max_metric_calls: int = 150,
    auto: str = "light",
    num_threads: int = 8,
    track_stats: bool = False,
)

optimized = optimizer.compile(
    student: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example] | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `metric` | `Callable` | Метрика с feedback (score + текстовый фидбек) |
| `reflection_lm` | `dspy.LM` | Модель для рефлексии |
| `auto` | `str` | Автоконфигурация: `"light"`, `"medium"`, `"heavy"` |

### TensorZero Python Client

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        function_name="my_function",
        input={
            "messages": [{"role": "user", "content": {"text": "..."}}]
        },
    )

    client.feedback(
        inference_id=response.inference_id,
        metric_name="quality_score",
        value=0.85,
    )
```

Альтернативно, TensorZero совместим с OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

response = client.chat.completions.create(
    model="tensorzero::function_name::assess_essay",
    messages=[{"role": "user", "content": "Оцени эссе: ..."}],
)
```

---

## Практика

### Пример 1: Оптимизация промпта с GEPA

Оптимизируем system prompt для задачи оценки эссе. GEPA эволюционирует промпт на обучающей выборке.

```python
import gepa

trainset = [
    {"question": "Evaluate this essay on AI impact:\n\nAI is changing the world. It helps people.", "answer": "3"},
    {"question": "Evaluate this essay on AI impact:\n\nThe intersection of artificial intelligence and labor economics presents a nuanced challenge. While Frey and Osborne (2013) estimated 47% automation risk, subsequent analyses by Arntz et al. (2016) suggest only 9% of jobs are fully automatable.", "answer": "9"},
    {"question": "Evaluate this essay on AI impact:\n\nAI is good. Some jobs will go. The end.", "answer": "1"},
    {"question": "Evaluate this essay on AI impact:\n\nArtificial intelligence represents a paradigm shift in how we approach knowledge work. Studies from MIT show that AI tools increase productivity by 40% for writing tasks, yet this gain comes with risks: over-reliance on generated content, erosion of critical thinking, and widening inequality.", "answer": "8"},
    {"question": "Evaluate this essay on AI impact:\n\nMachine learning is a subset of AI. Deep learning is a subset of machine learning. Neural networks have layers.", "answer": "2"},
]

valset = [
    {"question": "Evaluate this essay on AI impact:\n\nThe rapid advancement of generative AI has created unprecedented challenges for education. While tools like ChatGPT demonstrate impressive language capabilities, their integration into academic settings requires careful consideration of pedagogical implications.", "answer": "7"},
    {"question": "Evaluate this essay on AI impact:\n\nAI robots will take over everything soon.", "answer": "1"},
]

seed_prompt = {
    "system_prompt": (
        "You are an essay evaluator. Score the essay from 1 to 10. "
        "Return only the numeric score."
    )
}

result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o",
    max_metric_calls=50,
    display_progress_bar=True,
)

print("Исходный промпт:")
print(seed_prompt["system_prompt"])
print()
print("Оптимизированный промпт:")
print(result.best_candidate["system_prompt"])
print(f"\nBest score: {result.best_score:.2f}")
print(f"Кандидатов на Pareto-фронтире: {len(result.pareto_frontier)}")
```

### Пример 2: optimize_anything — оптимизация кода

GEPA может оптимизировать не только промпты, но и любой текстовый артефакт. Оптимизируем функцию Python:

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

test_cases = [
    ([3, 1, 4, 1, 5, 9, 2, 6], [1, 1, 2, 3, 4, 5, 6, 9]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    (list(range(100, 0, -1)), list(range(1, 101))),
]

def evaluate(candidate: str) -> tuple[float, dict]:
    import time
    try:
        namespace = {}
        exec(candidate, namespace)
        sort_fn = namespace.get("sort_list")
        if sort_fn is None:
            oa.log("Error: function 'sort_list' not found")
            return 0.0, {"error": "function not found"}

        passed = 0
        total_time = 0
        for inp, expected in test_cases:
            start = time.perf_counter()
            result = sort_fn(inp.copy())
            elapsed = time.perf_counter() - start
            total_time += elapsed
            if result == expected:
                passed += 1
            else:
                oa.log(f"FAIL: sort_list({inp}) = {result}, expected {expected}")

        score = passed / len(test_cases)
        oa.log(f"Passed: {passed}/{len(test_cases)}, Time: {total_time*1000:.1f}ms")
        return score, {"passed": passed, "time_ms": total_time * 1000}

    except Exception as e:
        oa.log(f"Runtime error: {e}")
        return 0.0, {"error": str(e)}

seed = """
def sort_list(lst):
    return sorted(lst)
"""

result = optimize_anything(
    seed_candidate=seed,
    evaluator=evaluate,
    objective="Write an efficient sort_list function. Optimize for correctness and speed.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=30)),
)

print("Лучший кандидат:")
print(result.best_candidate)
print(f"Score: {result.best_score:.2f}")
```

### Пример 3: DSPy + GEPA — оптимизация пайплайна

Оптимизируем multi-step пайплайн через DSPy. GEPA оптимизирует инструкции для каждого шага совместно.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class EssayAssessor(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("essay -> analysis: str")
        self.score = dspy.ChainOfThought("essay, analysis -> score: int, feedback: str")

    def forward(self, essay):
        analysis = self.analyze(essay=essay)
        return self.score(essay=essay, analysis=analysis.analysis)


def assessment_metric(example, pred, trace=None):
    try:
        pred_score = int(pred.score)
    except (ValueError, AttributeError):
        return dspy.Prediction(
            score=0.0,
            feedback=f"Could not parse score from prediction: {pred.score}"
        )

    expected = int(example.expected_score)
    diff = abs(pred_score - expected)

    if diff == 0:
        score = 1.0
        feedback = f"Exact match: predicted {pred_score} == expected {expected}"
    elif diff <= 1:
        score = 0.7
        feedback = f"Close: predicted {pred_score}, expected {expected} (diff={diff})"
    elif diff <= 2:
        score = 0.4
        feedback = f"Off by {diff}: predicted {pred_score}, expected {expected}. Review scoring criteria."
    else:
        score = 0.0
        feedback = f"Way off: predicted {pred_score}, expected {expected} (diff={diff}). Likely misunderstanding the rubric."

    return dspy.Prediction(score=score, feedback=feedback)


trainset = [
    dspy.Example(
        essay="AI is transforming society in profound ways. Studies show 47% job automation risk.",
        expected_score="6",
    ).with_inputs("essay"),
    dspy.Example(
        essay="AI good. Robots bad. The end.",
        expected_score="2",
    ).with_inputs("essay"),
    dspy.Example(
        essay="The intersection of machine learning and healthcare presents both promise and peril. "
              "Wang et al. (2021) demonstrated 94% diagnostic accuracy for skin lesions, "
              "yet Obermeyer et al. (2019) revealed racial bias in predictive algorithms.",
        expected_score="9",
    ).with_inputs("essay"),
]

optimizer = dspy.GEPA(
    metric=assessment_metric,
    reflection_lm=dspy.LM("openai/gpt-4o"),
    auto="light",
    num_threads=4,
    track_stats=True,
)

optimized = optimizer.compile(EssayAssessor(), trainset=trainset)

print("Оптимизированные инструкции (analyze):")
print(optimized.analyze.signature.instructions)
print()
print("Оптимизированные инструкции (score):")
print(optimized.score.signature.instructions)

test_essay = "Climate change is real. We should do something about it."
result = optimized(essay=test_essay)
print(f"\nТест: score={result.score}, feedback={result.feedback}")
```

### Пример 4: TensorZero — настройка gateway

Конфигурация TensorZero для задачи оценки эссе с двумя вариантами и A/B тестированием.

**Файл `config/tensorzero.toml`:**

```toml
[functions.assess_essay]
type = "chat"

[functions.assess_essay.variants.claude_sonnet]
type = "chat_completion"
model = "anthropic::claude-sonnet-4-20250514"
system_template = "prompts/assess/system.txt"
user_template = "prompts/assess/user.txt"
weight = 0.8

[functions.assess_essay.variants.gpt4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_template = "prompts/assess/system_optimized.txt"
user_template = "prompts/assess/user.txt"
weight = 0.2

[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"

[metrics.user_feedback]
type = "comment"
level = "inference"
```

**Файл `prompts/assess/system.txt`:**

```
You are an expert essay evaluator. Score the essay from 1 to 10.
Consider: thesis clarity, evidence usage, argument structure, critical thinking.
Return a JSON with "score" (int) and "feedback" (string).
```

**Файл `prompts/assess/system_optimized.txt`** — сюда поместим промпт, оптимизированный GEPA из примера 1.

**Docker Compose:**

```bash
docker compose up -d
```

**Использование через Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

response = client.chat.completions.create(
    model="tensorzero::function_name::assess_essay",
    messages=[
        {"role": "user", "content": "Evaluate this essay:\n\nAI is transforming society..."}
    ],
)

print(f"Model used: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Episode ID: {response.episode_id}")
```

Gateway автоматически распределяет трафик: 80% на Claude Sonnet, 20% на GPT-4o Mini (с оптимизированным промптом). Все inference-вызовы логируются в PostgreSQL.

### Пример 5: TensorZero — сбор feedback и оптимизация

После сбора inference-данных отправляем feedback для дальнейшей оптимизации.

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        function_name="assess_essay",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": {"text": "Evaluate: AI is changing the world in many ways..."},
                }
            ]
        },
    )

    print(f"Inference ID: {response.inference_id}")
    print(f"Response: {response.content}")

    client.feedback(
        inference_id=response.inference_id,
        metric_name="quality_score",
        value=0.85,
    )

    client.feedback(
        inference_id=response.inference_id,
        metric_name="user_feedback",
        value="Good assessment, but missed the lack of citations.",
    )

    print("Feedback отправлен")
```

Собранные данные используются для:
- **A/B анализа** — какой вариант получает лучшие scores
- **Fine-tuning** — SFT на лучших inference+feedback парах
- **DICL** — автоматический подбор few-shot примеров из истории

### Пример 6: Полный цикл — GEPA → TensorZero

Типичный workflow: оптимизируем промпт офлайн через GEPA, затем деплоим через TensorZero.

```python
import gepa

trainset = [
    {"question": "Score 1-10: AI is good.", "answer": "2"},
    {"question": "Score 1-10: The rapid advancement of AI poses fundamental questions about the nature of work. Brynjolfsson and McAfee (2014) argue for a 'race against the machine', while Autor (2015) suggests complementarity.", "answer": "9"},
    {"question": "Score 1-10: Machine learning uses data to learn patterns. It can be supervised or unsupervised.", "answer": "4"},
    {"question": "Score 1-10: AI will destroy all jobs and humanity is doomed.", "answer": "2"},
    {"question": "Score 1-10: The ethical implications of AI in healthcare extend beyond accuracy metrics. Consideration of patient autonomy, informed consent for AI-assisted diagnosis, and equitable access to AI-powered tools are critical dimensions.", "answer": "8"},
]

seed = {"system_prompt": "Score the essay from 1 to 10."}

result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o",
    max_metric_calls=100,
)

optimized_prompt = result.best_candidate["system_prompt"]
print("Оптимизированный промпт:")
print(optimized_prompt)

with open("prompts/assess/system_optimized.txt", "w") as f:
    f.write(optimized_prompt)
print("\nПромпт сохранён → теперь обнови weight в tensorzero.toml и перезапусти gateway")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | GEPA: seed → evolve → Pareto frontier |
| 2 | optimize_anything: ASI, evaluator с диагностикой |
| 3 | DSPy + GEPA: multi-step оптимизация, feedback-driven metric |
| 4 | TensorZero: functions, variants, A/B routing |
| 5 | TensorZero: feedback loop, данные для оптимизации |
| 6 | Полный цикл: офлайн-оптимизация → production deploy |

---

## Чеклист самопроверки

- [ ] Чем GEPA отличается от RL для оптимизации промптов? Назови 3 ключевых различия.
- [ ] Что такое Pareto-фронтир и зачем он нужен при оптимизации промптов?
- [ ] Что такое ASI (Actionable Side Information)? Почему это эффективнее, чем просто score?
- [ ] В чём разница между `gepa.optimize()` и `gepa.optimize_anything()`?
- [ ] Что такое Function и Variant в TensorZero? Как они связаны с A/B тестами?
- [ ] Какие методы оптимизации предоставляет TensorZero помимо GEPA?
- [ ] Зачем в DSPy метрика должна возвращать текстовый feedback, а не просто score?
- [ ] Опиши типичный workflow: GEPA для офлайн-оптимизации → TensorZero для production.
- [ ] Когда стоит использовать fine-tuning вместо prompt optimization? Назови 2 сценария.
- [ ] Что такое DICL (Dynamic In-Context Learning) и чем оно отличается от статических few-shot примеров?

---

## Частые ошибки

### 1. Слабая модель для рефлексии

```python
result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o-mini",
)
```

```python
result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o",
)
```

`reflection_lm` анализирует ошибки и предлагает исправления. Для этого нужна сильная модель. Использование той же модели для рефлексии и задачи — как просить студента проверить собственную контрольную.

### 2. Evaluator без диагностики

```python
def evaluate(candidate: str) -> float:
    return run_system(candidate).score
```

```python
def evaluate(candidate: str) -> float:
    result = run_system(candidate)
    oa.log(f"Output: {result.output}")
    oa.log(f"Expected: {result.expected}")
    oa.log(f"Error: {result.error}")
    return result.score
```

Без ASI (Actionable Side Information) GEPA знает только score, но не понимает **почему** кандидат провалился. Добавь `oa.log()` с максимумом полезной информации.

### 3. Слишком маленький бюджет

```python
result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    max_metric_calls=10,
)
```

10 evaluations — слишком мало для эволюционного поиска. Минимум 50-100 для простых задач, 150-300 для сложных. Но больше 500 обычно даёт diminishing returns.

### 4. Метрика DSPy без feedback

```python
def metric(example, pred, trace=None):
    return 1.0 if pred.answer == example.answer else 0.0
```

```python
def metric(example, pred, trace=None):
    correct = pred.answer == example.answer
    score = 1.0 if correct else 0.0
    feedback = (
        f"Correct!" if correct
        else f"Wrong: got '{pred.answer}', expected '{example.answer}'"
    )
    return dspy.Prediction(score=score, feedback=feedback)
```

GEPA использует текстовый feedback для рефлексии. Без него оптимизатор работает "вслепую" — знает score, но не причину ошибки.

### 5. TensorZero: feedback без inference_id

```python
client.feedback(
    metric_name="quality",
    value=0.9,
)
```

```python
client.feedback(
    inference_id=response.inference_id,
    metric_name="quality",
    value=0.9,
)
```

Feedback должен быть привязан к конкретному inference. Без `inference_id` TensorZero не может связать метрику с вариантом, и данные бесполезны для оптимизации.

---

## Что читать дальше

- [GEPA Documentation](https://gepa-ai.github.io/gepa/) — полная документация GEPA
- [GEPA Quick Start](https://gepa-ai.github.io/gepa/guides/quickstart/) — быстрый старт
- [GEPA Paper (arXiv)](https://arxiv.org/abs/2507.19457) — академическая статья
- [optimize_anything Blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) — API для оптимизации любых артефактов
- [TensorZero Documentation](https://www.tensorzero.com/docs/) — полная документация
- [TensorZero Quickstart](https://www.tensorzero.com/docs/gateway/tutorial/) — быстрый старт
- [TensorZero Optimization Recipes](https://www.tensorzero.com/docs/optimization) — SFT, DPO, GEPA, DICL
- [DSPy + GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/) — пошаговые notebooks
- [TensorZero GitHub](https://github.com/tensorzero/tensorzero) — исходный код и примеры

**Предыдущая тема:** [Тема 21: AI Testing & QA](topic_21_testing_qa.md) — тестирование AI-систем.
**Следующая тема:** [Тема 23: Databricks](topic_23_databricks.md) — MLflow, Model Serving, Vector Search.

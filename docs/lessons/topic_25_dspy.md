# Тема 25: DSPy — программирование LLM вместо промптинга

> **Пререквизиты:** [Тема 1](topic_01_prompt_engineering.md), [Тема 9](topic_09_evaluation.md)
> **Зависимости:** `dspy`, `litellm`

---

## Теория

### 1. Парадигма DSPy: программы вместо промптов

DSPy (Declarative Self-improving Language Programs) — фреймворк от Stanford NLP, который меняет парадигму работы с LLM. Вместо написания промптов ты пишешь **программы**: определяешь сигнатуры (что на входе → что на выходе), собираешь модули в пайплайн, а оптимизатор автоматически генерирует и улучшает промпты.

Ключевая идея: **промпт — это скомпилированный артефакт, а не написанный вручную код.**

Аналогия с обычным программированием:

| Традиционный ML | Традиционный Prompt Engineering | DSPy |
|---|---|---|
| Писать архитектуру модели | Писать промпт | Писать сигнатуру |
| Обучать на данных | Тестировать вручную | Компилировать с оптимизатором |
| Градиенты оптимизируют веса | Человек правит промпт | Оптимизатор генерирует промпт |

Преимущества DSPy:

- **Разделение логики и реализации.** Ты описываешь `"essay -> score, feedback"` — что делать. DSPy решает — как (какой промпт, какие примеры, какая цепочка рассуждений).
- **Автоматическая оптимизация.** Оптимизатор перебирает варианты промптов, few-shot примеров, chain-of-thought стратегий на обучающей выборке.
- **Модульность.** Программа — композиция модулей. Замена модели, добавление CoT, смена стратегии — одна строка.
- **Воспроизводимость.** Оптимизированная программа сериализуется — можно загрузить, поделиться, задеплоить.

### 2. Signatures — декларативный контракт

Signature — описание "что делает" модуль. Входные и выходные поля с описаниями. DSPy генерирует промпт автоматически.

Инлайн-формат:

```python
import dspy

classify = dspy.Predict("sentence -> sentiment: str")
result = classify(sentence="DSPy is amazing!")
print(result.sentiment)
```

Класс-формат (для сложных задач):

```python
import dspy

class AssessEssay(dspy.Signature):
    """Оцени студенческое эссе по академическим критериям."""

    essay: str = dspy.InputField(desc="Текст студенческого эссе")
    rubric: str = dspy.InputField(desc="Критерии оценивания")
    score: int = dspy.OutputField(desc="Оценка от 1 до 10")
    feedback: str = dspy.OutputField(desc="Развёрнутый фидбек с конкретными рекомендациями")
    strengths: list[str] = dspy.OutputField(desc="Сильные стороны работы")
```

DSPy берёт docstring, имена и описания полей — и генерирует system prompt + user prompt автоматически. Ты никогда не пишешь "You are an expert..." — это делает компилятор.

### 3. Modules — строительные блоки

Модули — предопределённые стратегии вызова LLM:

| Модуль | Что делает | Когда использовать |
|---|---|---|
| `dspy.Predict` | Простой вызов LLM | Простые задачи, классификация |
| `dspy.ChainOfThought` | Добавляет пошаговое рассуждение | Сложные задачи, требующие логики |
| `dspy.ProgramOfThought` | Генерирует и выполняет код | Математика, вычисления |
| `dspy.ReAct` | Reason + Act цикл с tools | Агентские задачи с инструментами |
| `dspy.MultiChainComparison` | Несколько цепочек → выбор лучшей | Когда нужна надёжность |
| `dspy.Refine` | Итеративное улучшение ответа | Когда первый ответ часто неточный |

Каждый модуль — обёртка над одной и той же signature. Разница — в стратегии промпта:

```python
predict = dspy.Predict("essay -> score: int, feedback: str")

cot = dspy.ChainOfThought("essay -> score: int, feedback: str")

react = dspy.ReAct("essay -> score: int, feedback: str", tools=[word_counter, citation_checker])
```

`Predict` просто спрашивает. `ChainOfThought` добавляет "Let's think step by step" и поле `reasoning`. `ReAct` добавляет tool-calling loop.

### 4. Programs — композиция модулей

Программа — класс, наследующий `dspy.Module`, с несколькими модулями внутри:

```python
import dspy

class EssayAssessor(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("essay -> topic, argument_quality, evidence_quality")
        self.score = dspy.ChainOfThought(
            "essay, topic, argument_quality, evidence_quality -> score: int, feedback: str"
        )

    def forward(self, essay: str):
        analysis = self.analyze(essay=essay)
        return self.score(
            essay=essay,
            topic=analysis.topic,
            argument_quality=analysis.argument_quality,
            evidence_quality=analysis.evidence_quality,
        )
```

Два модуля: `analyze` разбирает эссе, `score` выставляет оценку на основе анализа. Каждый модуль — отдельный вызов LLM с автоматически сгенерированным промптом.

### 5. Optimizers — автоматическая оптимизация

Оптимизатор берёт программу + обучающие примеры + метрику → улучшает промпты, примеры и стратегии.

Основные оптимизаторы:

| Оптимизатор | Что оптимизирует | Когда использовать |
|---|---|---|
| `BootstrapFewShot` | Подбирает few-shot примеры | Быстрый старт, мало данных |
| `BootstrapFewShotWithRandomSearch` | Few-shot + random search | Немного больше бюджета |
| `MIPRO` | Инструкции + few-shot совместно | Сложные задачи |
| `MIPROv2` | Улучшенный MIPRO | Production-качество |
| `GEPA` | Эволюционный поиск с рефлексией | Когда нужен максимум качества |
| `BootstrapFinetune` | Fine-tuning маленькой модели | Снижение стоимости inference |

```python
import dspy

trainset = [
    dspy.Example(
        essay="AI is transforming society...",
        score=7,
        feedback="Good analysis but lacks specific evidence."
    ).with_inputs("essay"),
    dspy.Example(
        essay="Machine learning is a subset of AI.",
        score=2,
        feedback="Too superficial, no argument or analysis."
    ).with_inputs("essay"),
]

def assessment_metric(example, pred, trace=None):
    score_diff = abs(int(pred.score) - example.score)
    if score_diff == 0: return 1.0
    if score_diff <= 1: return 0.7
    if score_diff <= 2: return 0.4
    return 0.0

optimizer = dspy.MIPROv2(
    metric=assessment_metric,
    auto="light",
    num_threads=4,
)

optimized = optimizer.compile(EssayAssessor(), trainset=trainset)
```

После `compile()` программа содержит оптимизированные промпты и few-shot примеры. Их можно сохранить:

```python
optimized.save("optimized_assessor.json")

loaded = EssayAssessor()
loaded.load("optimized_assessor.json")
```

### 6. Metrics — как измерять качество

Метрика — функция `(example, prediction, trace) -> float`. DSPy использует её для оптимизации:

```python
def essay_metric(example, pred, trace=None):
    score_correct = abs(int(pred.score) - example.score) <= 1
    has_feedback = len(pred.feedback) > 50
    
    if trace is not None:
        return score_correct and has_feedback
    
    return score_correct
```

Важный паттерн: `trace is not None` означает, что метрика вызывается во время оптимизации (для отбора few-shot примеров). В этом случае метрика должна быть строже. Когда `trace is None` — это evaluation, метрика может быть мягче.

LLM-as-judge метрика:

```python
class JudgeFeedback(dspy.Signature):
    """Оцени качество фидбека: конкретность, полезность, конструктивность."""
    essay: str = dspy.InputField()
    feedback: str = dspy.InputField()
    quality: float = dspy.OutputField(desc="0.0-1.0")

judge = dspy.Predict(JudgeFeedback)

def feedback_quality_metric(example, pred, trace=None):
    result = judge(essay=example.essay, feedback=pred.feedback)
    return float(result.quality)
```

### 7. Retrieval в DSPy

DSPy имеет встроенную поддержку retrieval для RAG:

```python
import dspy

class RAGAssessor(dspy.Module):
    def __init__(self, num_examples=3):
        self.retrieve = dspy.Retrieve(k=num_examples)
        self.assess = dspy.ChainOfThought(
            "essay, similar_examples -> score: int, feedback: str"
        )

    def forward(self, essay: str):
        similar = self.retrieve(essay)
        examples_text = "\n---\n".join(
            f"Essay: {doc[:200]}..." for doc in similar.passages
        )
        return self.assess(essay=essay, similar_examples=examples_text)
```

Retriever конфигурируется глобально:

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

retriever = ChromadbRM(
    collection_name="essays",
    persist_directory="./chroma_db",
    k=3,
)

dspy.configure(rm=retriever)
```

### 8. DSPy Assertions — runtime constraints

Assertions — инварианты, которые должен соблюдать LLM. При нарушении — автоматический retry с фидбеком:

```python
import dspy

class StrictAssessor(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought("essay, rubric -> score: int, feedback: str")

    def forward(self, essay: str, rubric: str):
        result = self.assess(essay=essay, rubric=rubric)

        dspy.Assert(
            1 <= int(result.score) <= 10,
            f"Score must be 1-10, got {result.score}",
        )

        dspy.Suggest(
            len(result.feedback) >= 100,
            "Feedback should be detailed (at least 100 characters)",
        )

        return result
```

`Assert` — жёсткое ограничение (retry при нарушении). `Suggest` — мягкое (warning, LLM старается, но не гарантирует).

---

## Справочник API

### dspy.configure

```python
dspy.configure(
    lm: LM | None = None,
    rm: Retrieve | None = None,
    adapter: Adapter | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `lm` | `LM` | Модель по умолчанию |
| `rm` | `Retrieve` | Retriever по умолчанию |

### dspy.LM

```python
lm = dspy.LM(
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    cache: bool = True,
)
```

Форматы model: `"openai/gpt-4o"`, `"anthropic/claude-sonnet-4-20250514"`, `"ollama_chat/llama3.1"`.

### dspy.Predict / ChainOfThought / ReAct

```python
predict = dspy.Predict(signature: str | type[Signature], **config)
cot = dspy.ChainOfThought(signature, **config)
react = dspy.ReAct(signature, tools: list[callable], max_iters: int = 5, **config)
```

### dspy.Example

```python
example = dspy.Example(
    field1="value1",
    field2="value2",
).with_inputs("field1")
```

`.with_inputs()` помечает поля как входные (остальные — выходные / labels).

### dspy.Evaluate

```python
evaluator = dspy.Evaluate(
    devset: list[Example],
    metric: callable,
    num_threads: int = 1,
    display_progress: bool = True,
    display_table: int = 0,
    provide_traceback: bool = False,
)

score = evaluator(program)
```

### Оптимизаторы

```python
optimizer = dspy.MIPROv2(
    metric: callable,
    auto: str = "light",
    num_threads: int = 6,
    num_candidates: int = 10,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
)

optimized = optimizer.compile(
    student: Module,
    trainset: list[Example],
    valset: list[Example] | None = None,
)
```

| `auto` | Бюджет | Описание |
|---|---|---|
| `"light"` | ~100 calls | Быстро, для экспериментов |
| `"medium"` | ~300 calls | Баланс |
| `"heavy"` | ~1000 calls | Максимальное качество |

### Сохранение / загрузка

```python
program.save("path.json")

loaded = MyModule()
loaded.load("path.json")
```

---

## Практика

### Пример 1: Базовый Predict и ChainOfThought

Простое сравнение: прямой вызов vs chain-of-thought.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
dspy.configure(lm=lm)

predict = dspy.Predict("essay -> score: int, feedback: str")
cot = dspy.ChainOfThought("essay -> score: int, feedback: str")

essay = (
    "The intersection of artificial intelligence and labor economics presents "
    "a nuanced challenge. Frey and Osborne (2013) estimated 47% automation risk."
)

result_predict = predict(essay=essay)
print(f"Predict: score={result_predict.score}, feedback={result_predict.feedback[:100]}...")

result_cot = cot(essay=essay)
print(f"CoT: score={result_cot.score}, feedback={result_cot.feedback[:100]}...")
print(f"Reasoning: {result_cot.reasoning[:200]}...")
```

### Пример 2: Typed Signatures

Сигнатуры как классы с подробными описаниями.

```python
import dspy

class AssessEssay(dspy.Signature):
    """Оцени студенческое эссе по академическим стандартам.
    Учитывай: тезис, аргументацию, доказательную базу, структуру и язык."""

    essay: str = dspy.InputField(desc="Полный текст студенческого эссе")
    rubric: str = dspy.InputField(desc="Критерии оценивания с максимальными баллами")
    score: int = dspy.OutputField(desc="Итоговая оценка от 1 до 10")
    feedback: str = dspy.OutputField(desc="Развёрнутый фидбек (минимум 3 предложения)")
    strengths: list[str] = dspy.OutputField(desc="Список сильных сторон работы")
    weaknesses: list[str] = dspy.OutputField(desc="Список слабых сторон работы")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

assessor = dspy.ChainOfThought(AssessEssay)

result = assessor(
    essay="AI is transforming society in profound ways...",
    rubric="Thesis: 0-3, Evidence: 0-3, Structure: 0-2, Language: 0-2",
)

print(f"Score: {result.score}")
print(f"Strengths: {result.strengths}")
print(f"Weaknesses: {result.weaknesses}")
```

### Пример 3: Multi-step Program

Программа из двух модулей: анализ → оценка.

```python
import dspy

class EssayAssessor(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(
            "essay -> topic: str, argument_quality: str, evidence_quality: str, structure_quality: str"
        )
        self.score = dspy.ChainOfThought(
            "essay, topic, argument_quality, evidence_quality, structure_quality "
            "-> score: int, feedback: str, grade: str"
        )

    def forward(self, essay: str):
        analysis = self.analyze(essay=essay)
        return self.score(
            essay=essay,
            topic=analysis.topic,
            argument_quality=analysis.argument_quality,
            evidence_quality=analysis.evidence_quality,
            structure_quality=analysis.structure_quality,
        )

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

assessor = EssayAssessor()
result = assessor(essay="The rapid advancement of AI poses fundamental questions...")

print(f"Topic: {result.topic}")
print(f"Score: {result.score}/10 ({result.grade})")
print(f"Feedback: {result.feedback}")
```

### Пример 4: Оптимизация с MIPROv2

Автоматическая оптимизация промптов на обучающей выборке.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

trainset = [
    dspy.Example(essay="AI is good. It helps people.", score=2).with_inputs("essay"),
    dspy.Example(
        essay="The intersection of AI and labor economics presents nuanced challenges. "
              "Frey and Osborne (2013) estimated 47% automation risk.",
        score=8,
    ).with_inputs("essay"),
    dspy.Example(essay="Machine learning uses data to learn patterns.", score=3).with_inputs("essay"),
    dspy.Example(
        essay="The ethical implications of AI in healthcare extend beyond accuracy. "
              "Patient autonomy, informed consent, and equitable access are critical.",
        score=8,
    ).with_inputs("essay"),
    dspy.Example(essay="AI will destroy all jobs.", score=1).with_inputs("essay"),
    dspy.Example(
        essay="Generative AI's impact on education requires nuanced analysis. "
              "While tools like ChatGPT demonstrate impressive capabilities (OpenAI, 2023), "
              "their integration demands careful pedagogical consideration.",
        score=7,
    ).with_inputs("essay"),
]

valset = [
    dspy.Example(essay="AI robots will take over.", score=1).with_inputs("essay"),
    dspy.Example(
        essay="The rapid advancement of generative AI has created unprecedented "
              "challenges for education and workforce development.",
        score=6,
    ).with_inputs("essay"),
]

def metric(example, pred, trace=None):
    diff = abs(int(pred.score) - example.score)
    if diff == 0: return 1.0
    if diff <= 1: return 0.7
    if diff <= 2: return 0.3
    return 0.0

assessor = EssayAssessor()

print("Before optimization:")
evaluate = dspy.Evaluate(devset=valset, metric=metric, display_progress=True)
baseline_score = evaluate(assessor)
print(f"Baseline: {baseline_score:.2f}")

optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=4)
optimized = optimizer.compile(assessor, trainset=trainset, valset=valset)

print("\nAfter optimization:")
optimized_score = evaluate(optimized)
print(f"Optimized: {optimized_score:.2f}")

optimized.save("essay_assessor_optimized.json")
```

### Пример 5: Assertions — runtime constraints

Жёсткие и мягкие ограничения на выход LLM.

```python
import dspy

class GuardedAssessor(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought("essay -> score: int, feedback: str")

    def forward(self, essay: str):
        result = self.assess(essay=essay)

        dspy.Assert(
            1 <= int(result.score) <= 10,
            f"Score must be between 1 and 10, got {result.score}",
        )

        dspy.Assert(
            len(result.feedback.split()) >= 20,
            "Feedback must be at least 20 words for meaningful assessment",
        )

        dspy.Suggest(
            any(word in result.feedback.lower() for word in ["because", "however", "although", "потому", "однако"]),
            "Feedback should include reasoning connectors for better argumentation",
        )

        return result

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

assessor = GuardedAssessor()

result = assessor(essay="AI is transforming the world in many ways...")
print(f"Score: {result.score}")
print(f"Feedback: {result.feedback}")
```

### Пример 6: ReAct с tools

Агент с инструментами через DSPy ReAct.

```python
import dspy
import re

def count_words(text: str) -> str:
    """Подсчитывает количество слов в тексте."""
    count = len(text.split())
    return f"Текст содержит {count} слов"

def check_citations(text: str) -> str:
    """Находит академические цитирования в формате (Author, Year)."""
    citations = re.findall(r'\(([^)]+,\s*\d{4})\)', text)
    if not citations:
        return "Цитирования не найдены"
    return f"Найдено {len(citations)} цитирований: {', '.join(citations)}"

def analyze_vocabulary(text: str) -> str:
    """Анализирует сложность лексики: средняя длина слова, уникальные слова."""
    words = text.lower().split()
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)
    return f"Средняя длина слова: {avg_len:.1f}, уникальность лексики: {unique_ratio:.2f}"

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

react_assessor = dspy.ReAct(
    "essay -> score: int, feedback: str",
    tools=[count_words, check_citations, analyze_vocabulary],
    max_iters=5,
)

result = react_assessor(
    essay=(
        "The rapid advancement of AI (Russell, 2019) poses fundamental questions. "
        "Brynjolfsson and McAfee (2014) argue for automation, while Autor (2015) "
        "suggests complementarity between human skills and AI."
    )
)

print(f"Score: {result.score}")
print(f"Feedback: {result.feedback}")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | Predict vs ChainOfThought — разные стратегии |
| 2 | Typed Signatures — описательные контракты |
| 3 | Multi-step Program — композиция модулей |
| 4 | MIPROv2 — автоматическая оптимизация |
| 5 | Assertions — runtime constraints |
| 6 | ReAct — агент с tools |

---

## Чеклист самопроверки

- [ ] Чем DSPy отличается от LangChain? В чём философское различие подходов?
- [ ] Что такое Signature в DSPy? Чем инлайн-формат отличается от класса?
- [ ] Чем `Predict` отличается от `ChainOfThought`? Когда использовать каждый?
- [ ] Что делает оптимизатор? Что именно он оптимизирует (промпт, примеры, стратегию)?
- [ ] В чём разница между `BootstrapFewShot` и `MIPROv2`?
- [ ] Зачем нужен `trace is not None` в метрике? Когда метрика используется для обучения vs evaluation?
- [ ] Что такое `dspy.Assert` и `dspy.Suggest`? Чем они отличаются?
- [ ] Как сохранить и загрузить оптимизированную программу?
- [ ] Как DSPy интегрируется с retrieval (RAG)?
- [ ] Зачем нужен `.with_inputs()` при создании Examples?

---

## Частые ошибки

### 1. Забыли configure LM

```python
import dspy

predict = dspy.Predict("text -> summary")
result = predict(text="Hello world")
```

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("text -> summary")
result = predict(text="Hello world")
```

Без `dspy.configure(lm=...)` DSPy не знает, какую модель использовать. Вызов модуля упадёт с ошибкой.

### 2. Example без with_inputs

```python
trainset = [
    dspy.Example(essay="text", score=5),
]
```

```python
trainset = [
    dspy.Example(essay="text", score=5).with_inputs("essay"),
]
```

Без `.with_inputs()` оптимизатор не знает, какие поля — входные, а какие — ожидаемые выходы. Оптимизация будет некорректной.

### 3. Метрика возвращает bool вместо float

```python
def metric(example, pred, trace=None):
    return pred.score == example.score
```

```python
def metric(example, pred, trace=None):
    diff = abs(int(pred.score) - example.score)
    if diff == 0: return 1.0
    if diff <= 1: return 0.7
    return 0.0
```

Бинарная метрика (True/False) слишком грубая для оптимизатора. Градуированная метрика даёт больше сигнала — оптимизатор различает "почти правильно" и "совсем не то".

### 4. Слишком маленький trainset

```python
trainset = [
    dspy.Example(essay="text1", score=5).with_inputs("essay"),
    dspy.Example(essay="text2", score=8).with_inputs("essay"),
]

optimizer = dspy.MIPROv2(metric=metric, auto="heavy")
```

2 примера — слишком мало для оптимизации. `auto="heavy"` потратит бюджет впустую. Минимум 5-10 примеров для `"light"`, 20+ для `"medium"`, 50+ для `"heavy"`.

### 5. Не сохранили оптимизированную программу

```python
optimized = optimizer.compile(program, trainset=trainset)
result = optimized(essay="test")
```

```python
optimized = optimizer.compile(program, trainset=trainset)
optimized.save("optimized_program.json")

loaded = MyProgram()
loaded.load("optimized_program.json")
result = loaded(essay="test")
```

Оптимизация стоит десятки-сотни API-вызовов. Результат нужно сохранить. Без `save()` при перезапуске скрипта вся оптимизация потеряется.

---

## Что читать дальше

- [DSPy Documentation](https://dspy.ai/) — полная документация
- [DSPy Signatures](https://dspy.ai/learn/programming/signatures/) — как описывать задачи
- [DSPy Modules](https://dspy.ai/learn/programming/modules/) — Predict, ChainOfThought, ReAct
- [DSPy Optimizers](https://dspy.ai/learn/optimization/optimizers/) — BootstrapFewShot, MIPROv2, GEPA
- [DSPy Assertions](https://dspy.ai/learn/programming/assertions/) — runtime constraints
- [DSPy Evaluation](https://dspy.ai/learn/evaluation/) — метрики и evaluation
- [DSPy Paper (arXiv)](https://arxiv.org/abs/2310.03714) — академическая статья
- [DSPy GitHub](https://github.com/stanfordnlp/dspy) — исходный код

**Предыдущая тема:** [Тема 24: PydanticAI](topic_24_pydantic_ai.md) — type-safe агенты.
**Следующая тема:** [Тема 26: LiteLLM](topic_26_litellm.md) — единый API для 100+ LLM.

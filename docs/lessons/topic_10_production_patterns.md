# Тема 10: Production-паттерны

> **Пререквизиты:** все предыдущие темы, особенно [Тема 2 (LCEL)](topic_02_langchain_lcel.md), [Тема 3 (Structured Output)](topic_03_structured_output.md), [Тема 8 (Observability)](topic_08_observability.md), [Тема 9 (Evaluation)](topic_09_evaluation.md)  
> **Зависимости:** `langchain-openai` (для embeddings в semantic cache), `numpy` (для cosine similarity), `langchain-anthropic`

---

## Теория

### 1. Model routing — выбор модели по задаче

Не каждая задача требует самой мощной (и дорогой) модели. Model routing — автоматический выбор модели на основе характеристик запроса.

```
Короткое эссе (< 200 слов), простая рубрика → Claude Haiku ($0.80/M input)
Длинное эссе (> 500 слов), детальная рубрика → Claude Sonnet ($3.00/M input)
```

Разница в стоимости: **~4x по input, ~4x по output**. При 1000 оценок в день model routing экономит сотни долларов в месяц, если 60% запросов — простые.

Три подхода к routing:

| Подход | Как работает | Когда использовать |
|--------|-------------|-------------------|
| **Rule-based** | Правила по длине текста, количеству параграфов, наличию цитат | Простые, предсказуемые паттерны сложности |
| **LLM-classifier** | Дешёвая модель (Haiku) классифицирует сложность | Когда rule-based недостаточно, но overhead classifier оправдан |
| **Cascading** | Сначала дешёвая модель, если результат некачественный → дорогая | Когда нужна гарантия качества при минимальной стоимости |

Rule-based routing — самый простой и дешёвый. Не требует дополнительных LLM-вызовов. Для нашего проекта — оптимальный выбор.

Сигналы сложности для assessment:

- **Длина текста**: < 200 слов → simple, ≥ 300 слов → complex
- **Количество параграфов**: < 3 → simple, ≥ 4 → complex
- **Наличие цитат**: `(Author, 2024)`, `[1]`, `et al.` → complex
- **Количество критериев в рубрике**: ≤ 3 → simple, > 3 → complex

Scoring: каждый сигнал = +1 к complexity_score. Если score ≥ 2 — complex, иначе — simple.

Мониторинг routing-решений обязателен. Без метрик невозможно понять, работает ли routing правильно: достаточно ли качество Haiku для простых задач? Не маршрутизируются ли сложные работы на дешёвую модель?

### 2. Semantic caching — кэширование по смыслу

Обычный кэш: точное совпадение ключа → вернуть кэшированный результат. Проблема: студенты часто отправляют почти идентичные работы (исправили опечатку, переформулировали предложение). Обычный кэш это пропускает.

Semantic cache: **похожий** запрос → вернуть кэшированный результат:

```
Запрос 1: "Evaluate this essay about climate change and its impact on polar bears"
Запрос 2: "Assess this essay on climate change effects on polar bear populations"
→ Cosine similarity = 0.96 → cache hit!
```

Механизм работы:

1. **Embedding** — преобразование текста запроса в числовой вектор через embedding-модель
2. **Search** — поиск среди закэшированных векторов по cosine similarity
3. **Threshold check** — если similarity > 0.95 → вернуть кэшированный результат
4. **Cache miss** — вызвать LLM, сохранить результат + embedding в кэш

Выбор threshold:

| Threshold | Поведение | Риск |
|-----------|----------|------|
| 0.99 | Почти exact match | Мало cache hits, минимальная экономия |
| 0.95 | Семантически идентичные запросы | Оптимальный баланс для assessment |
| 0.90 | Похожие, но не идентичные запросы | Могут вернуться нерелевантные результаты |
| 0.80 | Широкое совпадение | Высокий риск ошибок |

Для assessment 0.95 — оптимальный threshold. Ниже — риск вернуть оценку за другое эссе.

Ключ кэша должен включать не только текст работы, но и rubric_id. Одно и то же эссе, оценённое по разным рубрикам, должно давать разные результаты.

Cache invalidation: при изменении промпта или модели весь кэш инвалидируется (оценки были получены старой конфигурацией). Простое решение — добавить версию промпта к ключу кэша.

### 3. Guardrails — валидация LLM output

LLM может вернуть структурно-валидный, но логически-некорректный результат:

- Оценку выше max_score критерия (22/20)
- overall_score ≠ сумме criterion scores (70 ≠ 18+17+15+12+8)
- Пустой feedback при ненулевой оценке
- Нулевые strengths или improvements
- Оскорбительный контент в feedback

Guardrails — **программная валидация** после каждого LLM-вызова. Это последний барьер перед тем, как результат попадёт к пользователю.

Уровни валидации:

| Уровень | Что проверяем | Пример |
|---------|--------------|--------|
| **Структурный** | Типы, диапазоны, суммы | `score >= 0`, `score <= max_score`, `overall == sum` |
| **Полнота** | Все поля заполнены | `feedback != ""`, `len(strengths) >= 1` |
| **Согласованность** | Критерии совпадают с рубрикой | Имена критериев из ответа == имена из рубрики |
| **Семантический** | Содержательность feedback | Feedback ссылается на текст работы (advanced) |

Стратегия при ошибках: **retry с correction context**. Не просто повторяем вызов (та же ошибка повторится), а добавляем в промпт описание ошибок из предыдущей попытки:

```
Attempt 1: overall_score=70, sum=68 → error: "overall_score (70) != sum (68)"
Attempt 2: prompt += "Fix: overall_score must equal sum of criteria" → success
```

Максимум 3 попытки. После — возвращаем ошибку клиенту. Бесконечные retry = бесконечные расходы.

Pydantic validators (`field_validator`, `model_validator`) — ещё один слой защиты. Они валидируют данные при десериализации, до того как они попадут в бизнес-логику. Разница: guardrails-сервис может retry с исправленным контекстом, Pydantic — только reject.

### 4. Retry и fallback стратегии

LLM API подвержены transient errors:

| Ошибка | HTTP код | Стратегия |
|--------|----------|-----------|
| Rate limit | 429 | Exponential backoff с jitter |
| Server error | 500/503 | Retry через 1–2 секунды |
| Timeout | — | Retry с уменьшенным max_tokens или fallback |
| Model overloaded | 529 | Backoff или fallback на другую модель |

**Exponential backoff с jitter:**

```
Attempt 1: fail → wait 1.0 + random(0, 0.5) секунд
Attempt 2: fail → wait 2.0 + random(0, 1.0) секунд
Attempt 3: fail → wait 4.0 + random(0, 2.0) секунд
```

Jitter (случайная добавка) предотвращает **thundering herd** — когда все клиенты retry-ят одновременно после восстановления сервиса.

**Fallback** — при невозможности использовать основную модель, переключаемся на альтернативную:

```
Primary: Claude Sonnet → RateLimitError
Fallback 1: Claude Haiku → success (дешевле, но работает)
Fallback 2: GPT-4o-mini → success (другой провайдер)
```

В LangChain retry и fallback — встроенные методы Runnable: `.with_retry()` и `.with_fallbacks()`. Не нужно писать цикл вручную.

**Circuit breaker** (продвинутый паттерн): если N последовательных запросов к API failed — перестаём пытаться на X секунд, сразу используем fallback. Предотвращает лавину запросов к нерабочему сервису.

### 5. Rate limiting и бюджетирование

Проблемы без rate limiting:

- DDoS или баг → тысячи запросов к LLM API → счёт на тысячи долларов
- Провайдер возвращает 429 Too Many Requests → сервис деградирует
- Один пользователь монополизирует ресурсы → остальные ждут

Три уровня защиты:

| Уровень | Механизм | Пример |
|---------|----------|--------|
| **Per-user** | Max N оценок в час/день на пользователя | 50 assessments/час |
| **Global** | Max M запросов к API в минуту | 100 requests/min |
| **Budget cap** | Max $X в день | $50/день, при превышении → fallback на дешёвую модель |

Budget cap — самый важный для LLM-систем. В отличие от обычных API, где запрос стоит доли цента, один LLM-вызов может стоить $0.01–$0.10. При массовом использовании расходы растут быстро.

### 6. Cost optimization

Стоимость одного LLM-вызова:

```
cost = (input_tokens × input_price + output_tokens × output_price) / 1_000_000
```

Факторы, влияющие на стоимость:

| Фактор | Влияние | Как оптимизировать |
|--------|---------|-------------------|
| Модель | 4–10x разница между моделями | Model routing (раздел 1) |
| Длина промпта | Прямо пропорционально | Убрать few-shot для простых задач |
| max_tokens | Output стоит дороже input | Уменьшить для коротких ответов |
| Кэширование | Кэш hit = $0 | Semantic cache (раздел 2) |
| Повторные запросы | Guardrails retry = 2–3x стоимость | Улучшить промпт, чтобы retry реже |

Пример расчёта для нашего проекта:

```
Sonnet: 2000 input tokens × $3/M + 500 output tokens × $15/M = $0.0135/assessment
Haiku:  2000 input tokens × $0.80/M + 500 output tokens × $4/M = $0.0036/assessment

При 1000 assessments/день:
- Всё на Sonnet: $13.50/день = $405/месяц
- Routing (60% Haiku, 40% Sonnet): $7.56/день = $227/месяц
- Routing + Cache (30% hit rate): $5.29/день = $159/месяц
```

---

## Справочник API

### LangChain — `CacheBackedEmbeddings`

Кэширует результаты embedding-вызовов, чтобы не пересчитывать embedding для уже виденного текста.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
```

**`CacheBackedEmbeddings.from_bytes_store()` — фабричный метод:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `underlying_embeddings` | `Embeddings` | Базовая embedding-модель (`OpenAIEmbeddings`, `HuggingFaceEmbeddings`) |
| `document_embedding_cache` | `ByteStore` | Хранилище для кэша (`InMemoryByteStore`, `RedisStore`) |
| `namespace` | `str` | Namespace для изоляции кэшей разных моделей |

**Пример:**

```python
store = InMemoryByteStore()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace="assessment_embeddings",
)

vectors = cached_embeddings.embed_documents(["Hello world", "Hello world"])
```

Второй вызов для `"Hello world"` возвращает результат из кэша без вызова OpenAI API.

---

### LangChain — `InMemoryCache`

Кэш LLM-ответов по точному совпадению входа. Простое in-memory решение для разработки.

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

После установки кэша все LLM-вызовы через LangChain автоматически кэшируются. Повторный вызов с тем же prompt → ответ из кэша, без обращения к API.

Ограничения: in-memory = теряется при перезапуске, не подходит для production с несколькими инстансами. Для production — `RedisCache` или `SQLiteCache`.

---

### LangChain — `.with_retry()`

Метод любого `Runnable`. Оборачивает вызов в retry-логику с exponential backoff.

```python
from anthropic import RateLimitError, APIStatusError

chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    retry_if_exception_type=(RateLimitError, APIStatusError),
)
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|:---:|----------|
| `stop_after_attempt` | `int` | `3` | Максимальное количество попыток (включая первую) |
| `wait_exponential_jitter` | `bool` | `False` | Exponential backoff (2^attempt секунд) + случайный jitter |
| `retry_if_exception_type` | `tuple[type, ...]` | `(Exception,)` | Типы исключений, при которых делать retry |

**Как работает:**

```
Attempt 1: invoke() → RateLimitError → wait ~1s
Attempt 2: invoke() → RateLimitError → wait ~2s + jitter
Attempt 3: invoke() → success → return result
```

Если все попытки исчерпаны — пробрасывает последнее исключение.

---

### LangChain — `.with_fallbacks()`

Метод любого `Runnable`. При ошибке основного chain переключается на fallback.

```python
from anthropic import RateLimitError

primary = prompt | sonnet_llm.with_structured_output(AssessmentResponse)
fallback = prompt | haiku_llm.with_structured_output(AssessmentResponse)

chain = primary.with_fallbacks(
    fallbacks=[fallback],
    exceptions_to_handle=(RateLimitError,),
)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `fallbacks` | `list[Runnable]` | Список fallback chains, пробуются по порядку |
| `exceptions_to_handle` | `tuple[type, ...]` | Типы исключений, при которых переключаться на fallback |

Fallback chains пробуются по порядку. Если все fail — пробрасывается последнее исключение. Можно комбинировать с `.with_retry()`:

```python
chain = (
    primary
    .with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
    .with_fallbacks(fallbacks=[fallback])
)
```

Сначала retry основного chain (2 попытки), затем fallback.

---

### LangChain — `RunnableLambda`

Оборачивает обычную Python-функцию в `Runnable`, позволяя использовать её в LCEL-цепочках с оператором `|`.

```python
from langchain_core.runnables import RunnableLambda
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `func` | `Callable` | Синхронная функция |
| `afunc` | `Callable` | Асинхронная функция (опционально) |

**Пример — routing:**

```python
def classify_and_route(input_data: dict) -> dict:
    word_count = len(input_data["student_work"].split())
    input_data["complexity"] = "complex" if word_count > 300 else "simple"
    return input_data

classifier = RunnableLambda(classify_and_route)
chain = classifier | assessment_chain
```

**Асинхронный вариант:**

```python
async def async_classify(input_data: dict) -> dict:
    input_data["complexity"] = await determine_complexity(input_data)
    return input_data

classifier = RunnableLambda(func=sync_fallback, afunc=async_classify)
```

---

### Pydantic — `field_validator`, `model_validator`

Валидаторы Pydantic v2 для guardrails на уровне схемы.

**`@field_validator` — валидация одного поля:**

```python
from pydantic import BaseModel, field_validator

class ScoreGuard(BaseModel):
    overall_score: int
    max_overall_score: int

    @field_validator("overall_score")
    @classmethod
    def score_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("overall_score must be >= 0")
        return v
```

| Параметр декоратора | Описание |
|---------------------|----------|
| `*fields` | Имена полей, к которым применяется валидатор |
| `mode` | `"before"` (до преобразования типа) или `"after"` (после, по умолчанию) |

**`@model_validator` — валидация всей модели (cross-field):**

```python
from pydantic import BaseModel, model_validator

class AssessmentGuard(BaseModel):
    overall_score: int
    criterion_scores: list[CriterionScore]

    @model_validator(mode="after")
    def score_sum_matches(self) -> "AssessmentGuard":
        total = sum(cs.score for cs in self.criterion_scores)
        if self.overall_score != total:
            raise ValueError(
                f"overall_score ({self.overall_score}) != sum of criteria ({total})"
            )
        return self
```

| Параметр декоратора | Описание |
|---------------------|----------|
| `mode` | `"before"` (получает сырые данные как dict) или `"after"` (получает инстанс модели) |

`field_validator` — для constraints одного поля (диапазоны, форматы). `model_validator` — для cross-field constraints (суммы, зависимости между полями).

---

## Практика

Пять самостоятельных примеров — по одному на каждый production-паттерн из теории. Каждый пример можно запустить в Jupyter-ноутбуке как есть.

### Пример 1: Model routing — выбор модели по сложности

Rule-based routing: анализируем характеристики текста, считаем complexity score, выбираем дешёвую или дорогую модель.

```python
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

MODEL_PRICING = {
    "claude-haiku-4-20250414": {"input_per_m": 0.80, "output_per_m": 4.00},
    "claude-sonnet-4-20250514": {"input_per_m": 3.00, "output_per_m": 15.00},
}


class AssessmentResult(BaseModel):
    score: int = Field(description="Overall score 0-100")
    feedback: str = Field(description="Brief feedback")


def classify_complexity(text: str) -> tuple[str, dict]:
    word_count = len(text.split())
    paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
    has_citations = bool(re.search(r"\(\w+,?\s*\d{4}\)|\[\d+\]|et al\.", text))

    signals = {
        "word_count": word_count,
        "paragraphs": paragraph_count,
        "has_citations": has_citations,
    }

    complexity_score = 0
    if word_count >= 300:
        complexity_score += 1
    if paragraph_count >= 4:
        complexity_score += 1
    if has_citations:
        complexity_score += 1

    return ("complex" if complexity_score >= 2 else "simple"), signals


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])
    return (input_tokens * pricing["input_per_m"] + output_tokens * pricing["output_per_m"]) / 1_000_000


simple_essay = "Social media is bad for society. It makes people sad."

complex_essay = """The proliferation of artificial intelligence in educational settings
presents both transformative opportunities and significant ethical challenges.

Recent research by Smith et al. (2024) demonstrates that AI-powered tutoring
systems can improve student outcomes by 15-20% in standardized assessments.
The meta-analysis conducted by Johnson and Lee (2023) across 47 institutions
corroborates these findings.

However, the implementation of such systems raises concerns about data privacy,
algorithmic bias, and the potential erosion of critical thinking skills.

This essay argues that a balanced framework incorporating AI tools while
preserving human pedagogical judgment offers the most promising path forward."""

for label, essay in [("Simple", simple_essay), ("Complex", complex_essay)]:
    complexity, signals = classify_complexity(essay)
    model = "claude-haiku-4-20250414" if complexity == "simple" else "claude-sonnet-4-20250514"

    approx_input = len(essay.split()) * 2 + 300
    cost = estimate_cost(model, approx_input, 200)

    print(f"\n{'='*50}")
    print(f"{label} essay:")
    print(f"  Signals: {signals}")
    print(f"  Complexity: {complexity}")
    print(f"  Model: {model}")
    print(f"  Estimated cost: ${cost:.6f}")

    llm = ChatAnthropic(model=model, max_tokens=1024)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an essay assessor. Provide a score 0-100 and brief feedback."),
        ("human", "Assess this essay:\n\n{text}"),
    ])

    chain = prompt | llm.with_structured_output(AssessmentResult)
    result = chain.invoke({"text": essay})
    print(f"  Score: {result.score}, Feedback: {result.feedback[:80]}...")
```

### Пример 2: Semantic caching с embeddings

In-memory semantic cache: embedding запроса → cosine similarity с кэшированными → threshold check. Похожие запросы возвращают кэшированный результат без повторного вызова LLM.

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings


class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.threshold = threshold
        self._cache: list[tuple[str, list[float], str]] = []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm)

    def get(self, key: str) -> tuple[str | None, float]:
        if not self._cache:
            return None, 0.0

        query_vec = self.embeddings.embed_query(key)
        best_score = 0.0
        best_result = None

        for cached_key, cached_vec, cached_result in self._cache:
            similarity = self._cosine_similarity(query_vec, cached_vec)
            if similarity > best_score:
                best_score = similarity
                best_result = cached_result

        if best_score >= self.threshold:
            return best_result, best_score
        return None, best_score

    def set(self, key: str, value: str) -> None:
        vec = self.embeddings.embed_query(key)
        self._cache.append((key, vec, value))


cache = SemanticCache(threshold=0.95)

rubric_id = "essay_v1"
text_1 = "Evaluate this essay about climate change and its impact on polar bears"
text_2 = "Assess this essay on climate change effects on polar bear populations"
text_3 = "Review this paper about machine learning in healthcare"

cache.set(f"{rubric_id}:{text_1}", "Score: 75, Good analysis of climate impact")

for label, query in [("Similar query", text_2), ("Different query", text_3)]:
    result, similarity = cache.get(f"{rubric_id}:{query}")
    print(f"\n{label}:")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Cache hit: {result is not None}")
    if result:
        print(f"  Cached result: {result}")

cache.set(f"other_rubric:{text_1}", "Score: 60, Different rubric criteria")
result_same, _ = cache.get(f"{rubric_id}:{text_1}")
result_other, _ = cache.get(f"other_rubric:{text_1}")
print(f"\nSame text, different rubrics:")
print(f"  Rubric '{rubric_id}': {result_same}")
print(f"  Rubric 'other_rubric': {result_other}")
```

### Пример 3: Guardrails — валидация LLM output с retry

Валидация structured output и retry с correction context. При невалидном результате описание ошибок добавляется в промпт — модель исправляет себя.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, model_validator


class CriterionScore(BaseModel):
    name: str
    score: int
    max_score: int
    feedback: str

    @field_validator("score")
    @classmethod
    def score_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("score must be >= 0")
        return v


class Assessment(BaseModel):
    criterion_scores: list[CriterionScore]
    overall_score: int
    summary: str
    strengths: list[str]
    improvements: list[str]

    @model_validator(mode="after")
    def check_score_sum(self) -> "Assessment":
        total = sum(cs.score for cs in self.criterion_scores)
        if self.overall_score != total:
            raise ValueError(
                f"overall_score ({self.overall_score}) != sum ({total})"
            )
        return self


def validate_assessment(result: Assessment, max_scores: dict[str, int]) -> list[str]:
    errors = []
    for cs in result.criterion_scores:
        if cs.name in max_scores and cs.score > max_scores[cs.name]:
            errors.append(f"{cs.name}: score {cs.score} exceeds max {max_scores[cs.name]}")
        if not cs.feedback.strip():
            errors.append(f"{cs.name}: empty feedback")
    if not result.summary.strip():
        errors.append("empty summary")
    if not result.strengths:
        errors.append("no strengths listed")
    if not result.improvements:
        errors.append("no improvements listed")
    total = sum(cs.score for cs in result.criterion_scores)
    if result.overall_score != total:
        errors.append(f"overall_score ({result.overall_score}) != sum ({total})")
    return errors


llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=2048)

rubric = "Criteria: Content (max 40), Structure (max 30), Language (max 30)"
max_scores = {"Content": 40, "Structure": 30, "Language": 30}
student_work = "Climate change is a pressing global issue affecting every continent."

correction = ""
max_retries = 3

for attempt in range(1, max_retries + 1):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an essay assessor. Use EXACTLY these criteria with their max scores. "
                   "overall_score MUST equal the sum of criterion scores.\n\n{rubric}"),
        ("human", "Assess this work:\n\n{student_work}\n\n{correction}"),
    ])

    chain = prompt | llm.with_structured_output(Assessment)

    result = chain.invoke({
        "student_work": student_work,
        "rubric": rubric,
        "correction": correction,
    })

    errors = validate_assessment(result, max_scores)

    print(f"\nAttempt {attempt}:")
    print(f"  Overall: {result.overall_score}")
    print(f"  Criteria: {[(cs.name, cs.score) for cs in result.criterion_scores]}")

    if not errors:
        print(f"  ✓ Validation passed on attempt {attempt}")
        break

    print(f"  Errors: {errors}")
    correction = (
        f"IMPORTANT: Fix these validation errors: {'; '.join(errors)}. "
        f"overall_score must equal sum of criterion scores. "
        f"All scores must be within [0, max_score]."
    )
else:
    print(f"\n✗ Validation failed after {max_retries} attempts")
```

### Пример 4: Rate limiting и retry-стратегии

Exponential backoff с jitter, fallback на альтернативную модель, `.with_retry()` и `.with_fallbacks()` из LangChain.

```python
import random
import time

from anthropic import RateLimitError, APIStatusError
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class SimpleScore(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str


prompt = ChatPromptTemplate.from_messages([
    ("system", "Score the essay 0-100 with brief feedback."),
    ("human", "{text}"),
])


sonnet = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)
haiku = ChatAnthropic(model="claude-haiku-4-20250414", max_tokens=1024)

primary = prompt | sonnet.with_structured_output(SimpleScore)
fallback = prompt | haiku.with_structured_output(SimpleScore)

chain_with_retry = primary.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    retry_if_exception_type=(RateLimitError, APIStatusError),
)
print("✓ Chain with retry (3 attempts, exponential backoff + jitter)")

chain_with_fallback = primary.with_fallbacks(
    fallbacks=[fallback],
    exceptions_to_handle=(RateLimitError,),
)
print("✓ Chain with fallback (Sonnet → Haiku)")

robust_chain = (
    primary
    .with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
    .with_fallbacks(fallbacks=[fallback])
)
print("✓ Robust chain (retry 2x → fallback to Haiku)")

result = robust_chain.invoke({"text": "Social media affects modern communication patterns."})
print(f"\nResult: score={result.score}, feedback={result.feedback[:80]}...")


print("\n--- Manual backoff demo ---")
def call_with_backoff(func, max_attempts=3, base_delay=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay * attempt)
            print(f"  Attempt {attempt} failed: {type(e).__name__}. Retry in {delay:.1f}s")
            time.sleep(delay)


print("Exponential backoff pattern:")
print("  Attempt 1 fail → wait ~1.0 + random(0, 1.0)s")
print("  Attempt 2 fail → wait ~2.0 + random(0, 2.0)s")
print("  Attempt 3 fail → wait ~4.0 + random(0, 3.0)s")
```

### Пример 5: Cost optimization — расчёт и сравнение стоимости

Расчёт стоимости для разных конфигураций: модель, длина промпта, max_tokens. Оценка экономии от routing и caching.

```python
MODEL_PRICING = {
    "claude-haiku-4-20250414": {"input_per_m": 0.80, "output_per_m": 4.00},
    "claude-sonnet-4-20250514": {"input_per_m": 3.00, "output_per_m": 15.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    pricing = MODEL_PRICING[model]
    input_cost = input_tokens * pricing["input_per_m"] / 1_000_000
    output_cost = output_tokens * pricing["output_per_m"] / 1_000_000
    return {
        "model": model.split("-")[1].capitalize(),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total": input_cost + output_cost,
    }


configs = [
    ("claude-sonnet-4-20250514", 2000, 500),
    ("claude-sonnet-4-20250514", 2000, 1500),
    ("claude-haiku-4-20250414", 2000, 500),
    ("claude-haiku-4-20250414", 2000, 1500),
]

print("Per-assessment cost comparison:")
print(f"{'Model':<10} {'Input tok':>10} {'Output tok':>10} {'Cost':>12}")
print("-" * 45)

for model, inp, out in configs:
    est = estimate_cost(model, inp, out)
    print(f"{est['model']:<10} {inp:>10} {out:>10} ${est['total']:>10.6f}")


daily_assessments = 1000
haiku_ratio = 0.6

sonnet_cost = estimate_cost("claude-sonnet-4-20250514", 2000, 500)["total"]
haiku_cost = estimate_cost("claude-haiku-4-20250414", 2000, 500)["total"]

all_sonnet_daily = sonnet_cost * daily_assessments
routed_daily = (haiku_cost * daily_assessments * haiku_ratio +
                sonnet_cost * daily_assessments * (1 - haiku_ratio))

cache_hit_rate = 0.30
routed_cached_daily = routed_daily * (1 - cache_hit_rate)

print(f"\n--- Daily cost ({daily_assessments} assessments/day) ---")
print(f"All Sonnet:              ${all_sonnet_daily:.2f}/day = ${all_sonnet_daily * 30:.0f}/month")
print(f"Routing (60% Haiku):     ${routed_daily:.2f}/day = ${routed_daily * 30:.0f}/month")
print(f"Routing + Cache (30%):   ${routed_cached_daily:.2f}/day = ${routed_cached_daily * 30:.0f}/month")
print(f"\nSavings with all optimizations: {(1 - routed_cached_daily / all_sonnet_daily) * 100:.0f}%")
```

---

## Чеклист самопроверки

- [ ] Назови три подхода к model routing (rule-based, classifier, cascading). Когда какой?
- [ ] Как работает semantic cache? Что означает cosine similarity threshold 0.95?
- [ ] Какие проверки должны быть в guardrails для assessment? Назови минимум 5
- [ ] Что такое exponential backoff с jitter? Зачем jitter?
- [ ] Как `.with_retry()` и `.with_fallbacks()` комбинируются в LangChain?
- [ ] Формула стоимости LLM-вызова: какие факторы влияют?
- [ ] Почему ключ semantic cache должен включать rubric_id?
- [ ] `field_validator` vs `model_validator` — в чём разница, когда какой?

---

## Частые ошибки

### 1. Cache без учёта рубрики

```python
cache_key = embed(student_work)
```

Одно эссе + разные рубрики = разные оценки. Ключ кэша должен включать rubric_id. Иначе вернётся оценка по неправильной рубрике.

```python
cache_key = embed(f"{rubric_id}:{student_work}")
```

### 2. Guardrails только на overall_score

```python
if result.overall_score < 0 or result.overall_score > 100:
    retry()
```

Проверка только границ overall_score пропускает массу ошибок: per-criterion score > max, пустой feedback, несовпадение суммы. Используйте полную валидацию всех полей.

### 3. Retry без backoff

```python
for _ in range(3):
    try:
        return await call_api()
    except:
        pass
```

Мгновенные retry усиливают проблему: сервер уже перегружен, а вы шлёте ещё больше запросов. Используйте exponential backoff.

### 4. Model routing без мониторинга

```python
model = route(student_work)
result = await chain.ainvoke(data)
```

Без логирования routing-решений невозможно понять: достаточно ли Haiku для простых задач? Не маршрутизируются ли сложные работы неправильно? Всегда возвращайте routing metadata.

### 5. Semantic cache с низким threshold

```python
cache = SemanticCache(similarity_threshold=0.80)
```

Threshold 0.80 слишком низкий — вернётся оценка за совершенно другое эссе, просто на похожую тему. Для assessment 0.95 — минимально допустимый threshold.

---

## Что читать дальше

- [LangChain Caching](https://python.langchain.com/docs/how_to/llm_caching/) — встроенное кэширование LLM-вызовов
- [LangChain Fallbacks](https://python.langchain.com/docs/how_to/fallbacks/) — паттерн fallback в LCEL
- [Guardrails AI](https://www.guardrailsai.com/) — фреймворк валидации LLM output
- [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits) — лимиты и рекомендации для Claude API
- [Redis Vector Search](https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/) — production semantic cache на Redis
- [Anthropic Pricing](https://docs.anthropic.com/en/docs/about-claude/models) — актуальные цены моделей для cost optimization

**Поздравляю!** Вы прошли все 10 тем AI Engineering курса. Следующий шаг — объединить все паттерны в единую production-ready assessment system: routing + guardrails + cache + evaluation + observability.

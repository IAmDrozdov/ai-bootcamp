# Тема 10: Production-паттерны

> **Пререквизиты:** все предыдущие темы, особенно [Тема 2 (LCEL)](topic_02_langchain_lcel.md), [Тема 3 (Structured Output)](topic_03_structured_output.md), [Тема 8 (Observability)](topic_08_observability.md), [Тема 9 (Evaluation)](topic_09_evaluation.md)  
> **Что добавляем в проект:** роутер `app/api/v1/production.py`, сервисы `app/services/model_router.py`, `app/services/guardrails.py`, `app/services/semantic_cache.py`, схемы `app/schemas/production.py`  
> **Зависимости:** `langchain-openai` (для embeddings в semantic cache), `numpy` (для cosine similarity), `redis`, `langchain-redis` (опционально, для production cache)

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

## Практика: роутер `/api/v1/production`

Создаём production-ready API с четырьмя паттернами: model routing, guardrails, semantic cache, cost analysis. Каждый эндпоинт демонстрирует один паттерн из теории.

Архитектура:

```
POST /api/v1/production/routed-assess   →  routing по сложности
POST /api/v1/production/guarded-assess  →  assessment + валидация + retry
POST /api/v1/production/cached-assess   →  semantic cache + assessment
POST /api/v1/production/cost-analysis   →  расчёт стоимости конфигураций
```

### Шаг 1: Схемы (`app/schemas/production.py`)

Каждый эндпоинт имеет собственную пару Request/Response. Response-модели расширяют `AssessmentResponse` метаданными: routing info, validation attempts, cache status.

```python
from pydantic import BaseModel, Field

from app.schemas.assessment import AssessmentResponse


class RoutedAssessRequest(BaseModel):
    student_work: str
    rubric_id: str = "essay_default"


class RoutingMetadata(BaseModel):
    complexity: str = Field(description="simple or complex")
    model_used: str
    estimated_cost: float


class RoutedAssessResponse(BaseModel):
    assessment: AssessmentResponse
    routing: RoutingMetadata


class GuardedAssessRequest(BaseModel):
    student_work: str
    rubric_id: str = "essay_default"


class GuardedAssessResponse(BaseModel):
    assessment: AssessmentResponse
    validation_attempts: int


class CachedAssessRequest(BaseModel):
    student_work: str
    rubric_id: str = "essay_default"


class CachedAssessResponse(BaseModel):
    assessment: AssessmentResponse
    cache_hit: bool
    similarity_score: float


class CostConfig(BaseModel):
    model: str
    max_tokens: int
    prompt_length: int


class CostEstimate(BaseModel):
    model: str
    max_tokens: int
    prompt_length: int
    estimated_input_cost: float
    estimated_output_cost: float
    total_cost_per_assessment: float


class CostAnalysisRequest(BaseModel):
    configurations: list[CostConfig]


class CostAnalysisResponse(BaseModel):
    estimates: list[CostEstimate]
```

Связь с теорией: `RoutingMetadata` содержит данные для мониторинга routing-решений (раздел 1). `CachedAssessResponse.cache_hit` и `similarity_score` — метрики semantic cache (раздел 2). `GuardedAssessResponse.validation_attempts` показывает, сколько retry потребовалось guardrails (раздел 3).

### Шаг 2: Сервис model routing (`app/services/model_router.py`)

Rule-based routing: анализируем характеристики текста и рубрики, присваиваем complexity score, выбираем модель. Также включает estimate стоимости вызова.

```python
import re

from langchain_anthropic import ChatAnthropic

from app.chains.assessment_chain import build_assessment_chain
from app.config import Settings
from app.schemas.assessment import AssessmentResponse
from app.schemas.rubric import Rubric

MODEL_PRICING = {
    "claude-haiku-4-20250414": {"input_per_m": 0.80, "output_per_m": 4.00},
    "claude-sonnet-4-20250514": {"input_per_m": 3.00, "output_per_m": 15.00},
}


def format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


def classify_complexity(student_work: str, rubric: Rubric) -> str:
    word_count = len(student_work.split())
    paragraph_count = len([p for p in student_work.split("\n\n") if p.strip()])
    has_citations = bool(re.search(r"\(\w+,?\s*\d{4}\)|\[\d+\]|et al\.", student_work))
    criteria_count = len(rubric.criteria)

    complexity_score = 0
    if word_count >= 300:
        complexity_score += 1
    if paragraph_count >= 4:
        complexity_score += 1
    if has_citations:
        complexity_score += 1
    if criteria_count > 3:
        complexity_score += 1

    return "complex" if complexity_score >= 2 else "simple"


def create_routed_llm(complexity: str, config: Settings) -> ChatAnthropic:
    model = "claude-haiku-4-20250414" if complexity == "simple" else config.model_name
    return ChatAnthropic(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.anthropic_api_key,
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])
    return (input_tokens * pricing["input_per_m"] + output_tokens * pricing["output_per_m"]) / 1_000_000


async def route_and_assess(
    student_work: str,
    rubric: Rubric,
    config: Settings,
) -> tuple[AssessmentResponse, str, str, float]:
    complexity = classify_complexity(student_work, rubric)
    llm = create_routed_llm(complexity, config)
    chain = build_assessment_chain(llm)

    rubric_text = format_rubric(rubric)
    result = await chain.ainvoke({"student_work": student_work, "rubric": rubric_text})

    approx_input_tokens = len(student_work.split()) * 2 + 500
    approx_output_tokens = 500
    cost = estimate_cost(llm.model, approx_input_tokens, approx_output_tokens)

    return result, complexity, llm.model, cost
```

Связь с теорией: `classify_complexity` реализует rule-based routing (раздел 1) — считает complexity_score по четырём сигналам. `estimate_cost` использует формулу из раздела 6 (cost optimization). `route_and_assess` возвращает routing metadata для мониторинга.

### Шаг 3: Сервис guardrails (`app/services/guardrails.py`)

Валидация LLM output + retry с correction context. `validate_assessment` проверяет все constraints из раздела 3 теории. `assess_with_guardrails` — retry-цикл: при невалидном результате добавляет описание ошибок в промпт и повторяет (до 3 раз).

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse
from app.schemas.rubric import Rubric


class GuardrailError(Exception):
    def __init__(self, errors: list[str], attempts: int):
        self.errors = errors
        self.attempts = attempts
        super().__init__(f"Validation failed after {attempts} attempts: {errors}")


def format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)


def validate_assessment(result: AssessmentResponse, rubric: Rubric) -> list[str]:
    errors = []

    total = sum(cs.score for cs in result.criterion_scores)
    if result.overall_score != total:
        errors.append(
            f"overall_score ({result.overall_score}) != sum of criterion scores ({total})"
        )

    max_total = sum(cs.max_score for cs in result.criterion_scores)
    if result.max_overall_score != max_total:
        errors.append(
            f"max_overall_score ({result.max_overall_score}) != sum of max_scores ({max_total})"
        )

    for cs in result.criterion_scores:
        if cs.score < 0:
            errors.append(f"{cs.criterion_name}: score ({cs.score}) is negative")
        if cs.score > cs.max_score:
            errors.append(f"{cs.criterion_name}: score ({cs.score}) exceeds max ({cs.max_score})")
        if not cs.feedback.strip():
            errors.append(f"{cs.criterion_name}: empty feedback")

    if not result.summary.strip():
        errors.append("empty summary")
    if not result.strengths:
        errors.append("no strengths listed")
    if not result.improvements:
        errors.append("no improvements listed")

    return errors


async def assess_with_guardrails(
    student_work: str,
    rubric: Rubric,
    llm: ChatAnthropic,
    max_retries: int = 3,
) -> tuple[AssessmentResponse, int]:
    structured_llm = llm.with_structured_output(AssessmentResponse)
    rubric_text = format_rubric(rubric)
    correction = ""

    for attempt in range(1, max_retries + 1):
        prompt = ChatPromptTemplate.from_messages([
            ("system", ASSESSMENT_SYSTEM_PROMPT),
            (
                "human",
                "Please assess the following student work:\n\n"
                "{student_work}\n\n{correction}",
            ),
        ]).partial(
            few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
            few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
        )
        chain = prompt | structured_llm

        result = await chain.ainvoke({
            "student_work": student_work,
            "rubric": rubric_text,
            "correction": correction,
        })

        errors = validate_assessment(result, rubric)
        if not errors:
            return result, attempt

        correction = (
            f"IMPORTANT: Your previous assessment had validation errors: "
            f"{'; '.join(errors)}. "
            f"Ensure overall_score equals the sum of criterion scores, "
            f"all scores are within [0, max_score], and all fields are non-empty."
        )

    raise GuardrailError(errors, max_retries)
```

Связь с теорией: `validate_assessment` реализует все четыре уровня валидации из раздела 3 (структурный, полнота, согласованность). Retry с correction context — ключевой паттерн: модель получает описание своих ошибок и исправляет их. `{correction}` — пустая строка при первой попытке, описание ошибок при повторных.

### Шаг 4: Сервис semantic cache (`app/services/semantic_cache.py`)

In-memory semantic cache с embedding-based similarity search. Использует OpenAI embeddings для vectorизации и numpy cosine similarity для поиска. Ключ кэша = `rubric_id:student_work` — одно эссе с разными рубриками кэшируется отдельно.

Для production замените in-memory на Redis + HNSW index. Для обучения in-memory достаточно.

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.schemas.assessment import AssessmentResponse


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.threshold = similarity_threshold
        self._cache: list[tuple[list[float], AssessmentResponse]] = []

    async def get(
        self, student_work: str, rubric_id: str,
    ) -> tuple[AssessmentResponse | None, float]:
        if not self._cache:
            return None, 0.0

        cache_key = f"{rubric_id}:{student_work}"
        query_vec = np.array(await self.embeddings.aembed_query(cache_key))

        best_score = 0.0
        best_result = None

        for cached_vec_list, cached_result in self._cache:
            cached_vec = np.array(cached_vec_list)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)
            if norm_product == 0:
                continue
            similarity = float(np.dot(query_vec, cached_vec) / norm_product)
            if similarity > best_score:
                best_score = similarity
                best_result = cached_result

        if best_score >= self.threshold:
            return best_result, best_score
        return None, best_score

    async def set(
        self, student_work: str, rubric_id: str, result: AssessmentResponse,
    ) -> None:
        cache_key = f"{rubric_id}:{student_work}"
        embedding = await self.embeddings.aembed_query(cache_key)
        self._cache.append((embedding, result))


_cache_instance: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
```

Связь с теорией: реализует полный цикл semantic caching из раздела 2 — embedding → similarity search → threshold check. `rubric_id` в ключе кэша предотвращает ошибку "cache без учёта рубрики". Singleton-паттерн (`get_semantic_cache`) гарантирует единый кэш для всего приложения.

### Шаг 5: Роутер (`app/api/v1/production.py`)

Роутер связывает HTTP-эндпоинты с тремя сервисами. Каждый эндпоинт демонстрирует один production-паттерн. Dependency injection через `Depends()` — тот же паттерн, что и в основном assessment-роутере.

```python
from fastapi import APIRouter, HTTPException

from app.dependencies import LLMDep, RubricStoreDep, SettingsDep
from app.schemas.production import (
    CachedAssessRequest,
    CachedAssessResponse,
    CostAnalysisRequest,
    CostAnalysisResponse,
    CostEstimate,
    GuardedAssessRequest,
    GuardedAssessResponse,
    RoutedAssessRequest,
    RoutedAssessResponse,
    RoutingMetadata,
)
from app.services.guardrails import GuardrailError, assess_with_guardrails
from app.services.model_router import (
    MODEL_PRICING,
    route_and_assess,
)
from app.services.semantic_cache import get_semantic_cache

router = APIRouter(prefix="/production", tags=["lesson-10-production"])


@router.post("/routed-assess")
async def routed_assessment(
    request: RoutedAssessRequest,
    rubrics: RubricStoreDep,
    settings: SettingsDep,
) -> RoutedAssessResponse:
    rubric = rubrics.get(request.rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail=f"Rubric '{request.rubric_id}' not found")

    result, complexity, model, cost = await route_and_assess(
        request.student_work, rubric, settings,
    )

    return RoutedAssessResponse(
        assessment=result,
        routing=RoutingMetadata(
            complexity=complexity,
            model_used=model,
            estimated_cost=round(cost, 6),
        ),
    )


@router.post("/guarded-assess")
async def guarded_assessment(
    request: GuardedAssessRequest,
    rubrics: RubricStoreDep,
    llm: LLMDep,
) -> GuardedAssessResponse:
    rubric = rubrics.get(request.rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail=f"Rubric '{request.rubric_id}' not found")

    try:
        result, attempts = await assess_with_guardrails(
            request.student_work, rubric, llm,
        )
    except GuardrailError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Validation failed after {e.attempts} attempts: {e.errors}",
        )

    return GuardedAssessResponse(assessment=result, validation_attempts=attempts)


@router.post("/cached-assess")
async def cached_assessment(
    request: CachedAssessRequest,
    rubrics: RubricStoreDep,
    llm: LLMDep,
) -> CachedAssessResponse:
    rubric = rubrics.get(request.rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail=f"Rubric '{request.rubric_id}' not found")

    cache = get_semantic_cache()
    cached, similarity = await cache.get(request.student_work, request.rubric_id)

    if cached:
        return CachedAssessResponse(
            assessment=cached,
            cache_hit=True,
            similarity_score=round(similarity, 4),
        )

    from app.chains.assessment_chain import build_assessment_chain
    from app.services.model_router import format_rubric

    chain = build_assessment_chain(llm)
    rubric_text = format_rubric(rubric)
    result = await chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
    })

    await cache.set(request.student_work, request.rubric_id, result)

    return CachedAssessResponse(
        assessment=result,
        cache_hit=False,
        similarity_score=round(similarity, 4),
    )


@router.post("/cost-analysis")
async def cost_analysis(request: CostAnalysisRequest) -> CostAnalysisResponse:
    estimates = []
    for config in request.configurations:
        pricing = MODEL_PRICING.get(config.model, MODEL_PRICING["claude-sonnet-4-20250514"])
        input_tokens = config.prompt_length + 500
        output_tokens = min(config.max_tokens, 1500)

        input_cost = input_tokens * pricing["input_per_m"] / 1_000_000
        output_cost = output_tokens * pricing["output_per_m"] / 1_000_000

        estimates.append(CostEstimate(
            model=config.model,
            max_tokens=config.max_tokens,
            prompt_length=config.prompt_length,
            estimated_input_cost=round(input_cost, 6),
            estimated_output_cost=round(output_cost, 6),
            total_cost_per_assessment=round(input_cost + output_cost, 6),
        ))

    return CostAnalysisResponse(estimates=estimates)
```

### Шаг 6: Регистрация в `app/api/router.py`

Добавляем production-роутер:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, eval, production, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(eval.router)
api_router.include_router(production.router)
```

### Шаг 7: Тестирование

```bash
uvicorn app.main:app --reload
```

**1. Model routing:**

Короткое эссе → ожидаем routing на Haiku:

```bash
curl -X POST http://localhost:8000/api/v1/production/routed-assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Social media is bad for society. Everyone knows this. It makes people sad and distracted. We should use it less.",
    "rubric_id": "essay_default"
  }'
```

Длинное эссе с цитатами → ожидаем routing на Sonnet:

```bash
curl -X POST http://localhost:8000/api/v1/production/routed-assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The proliferation of artificial intelligence in educational settings presents both transformative opportunities and significant ethical challenges that demand careful examination.\n\nRecent research by Smith et al. (2024) demonstrates that AI-powered tutoring systems can improve student outcomes by 15-20% in standardized assessments. The meta-analysis conducted by Johnson and Lee (2023) across 47 institutions corroborates these findings, noting particular effectiveness in STEM disciplines.\n\nHowever, the implementation of such systems raises concerns about data privacy, algorithmic bias, and the potential erosion of critical thinking skills. As Nguyen (2024) argues, over-reliance on AI feedback may create a generation of students who cannot self-assess their work.\n\nThis essay argues that a balanced framework incorporating AI tools while preserving human pedagogical judgment offers the most promising path forward. Drawing on case studies from three universities that have successfully integrated AI assessment tools, I will demonstrate that the key lies not in choosing between human and artificial intelligence, but in designing systems where each complements the other.",
    "rubric_id": "essay_default"
  }'
```

Проверьте поле `routing` в ответе — `complexity` и `model_used` должны отличаться.

**2. Guardrails:**

```bash
curl -X POST http://localhost:8000/api/v1/production/guarded-assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is real. We need to act now. The polar ice caps are melting and sea levels are rising. Scientists agree that human activity is the main cause.",
    "rubric_id": "essay_default"
  }'
```

Обратите внимание на `validation_attempts` в ответе. Значение 1 — модель дала валидный ответ с первой попытки. Значение 2–3 — были retry с correction context.

**3. Semantic cache:**

Первый запрос — cache miss:

```bash
curl -X POST http://localhost:8000/api/v1/production/cached-assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The impact of renewable energy on global economics is significant. Solar and wind power have become increasingly cost-competitive with fossil fuels.",
    "rubric_id": "essay_default"
  }'
```

Повторный запрос (идентичный или почти идентичный) — cache hit:

```bash
curl -X POST http://localhost:8000/api/v1/production/cached-assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The impact of renewable energy on global economics is significant. Solar and wind power have become increasingly cost-competitive with fossil fuels.",
    "rubric_id": "essay_default"
  }'
```

Проверьте `cache_hit: true` и `similarity_score` ≈ 1.0 во втором ответе.

**4. Cost analysis:**

```bash
curl -X POST http://localhost:8000/api/v1/production/cost-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "configurations": [
      {"model": "claude-sonnet-4-20250514", "max_tokens": 4096, "prompt_length": 2000},
      {"model": "claude-sonnet-4-20250514", "max_tokens": 2048, "prompt_length": 2000},
      {"model": "claude-haiku-4-20250414", "max_tokens": 4096, "prompt_length": 2000},
      {"model": "claude-haiku-4-20250414", "max_tokens": 2048, "prompt_length": 1000}
    ]
  }'
```

Ответ содержит estimated cost для каждой конфигурации — используйте для выбора оптимального баланса цена/качество.

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

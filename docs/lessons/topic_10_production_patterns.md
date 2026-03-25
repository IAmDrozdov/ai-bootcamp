# Тема 10: Production-паттерны

> **Пререквизиты:** все предыдущие темы, особенно [Тема 2 (LCEL)](topic_02_langchain_lcel.md), [Тема 3 (Structured Output)](topic_03_structured_output.md), [Тема 8 (Observability)](topic_08_observability.md)
> **Где в проекте:** `app/services/`, `app/chains/`
> **Зависимости:** `redis`, `langchain-redis` (для кэширования)

---

## Теория

### 1. Model routing — выбор модели по задаче

Не каждая задача требует самой мощной (и дорогой) модели. Model routing — это автоматический выбор модели на основе сложности запроса:

```
Короткое эссе (< 200 слов), простая рубрика → Claude Haiku ($0.25/M input)
Длинное эссе (> 500 слов), детальная рубрика → Claude Sonnet ($3/M input)
```

Разница в стоимости: **12x**. При 1000 оценок в день model routing может сэкономить сотни долларов в месяц.

Подходы к routing:
- **Rule-based** — по длине текста, типу рубрики, пороговым значениям
- **Classifier** — LLM-классификатор определяет сложность (сам вызов дешёвый, haiku)
- **Cascading** — сначала дешёвая модель, если результат некачественный → дорогая

### 2. Semantic caching — кэширование по смыслу

Обычный кэш: точное совпадение ключа → вернуть кэшированный результат.

Semantic cache: **похожий** запрос → вернуть кэшированный результат:

```
Запрос 1: "Evaluate this essay about climate change and its impact on polar bears"
Запрос 2: "Assess this essay on climate change effects on polar bear populations"
→ Cosine similarity = 0.95 → cache hit!
```

Механизм:
1. Embedding запроса
2. Поиск в vector store среди закэшированных запросов
3. Если similarity > threshold → вернуть кэш
4. Иначе → вызвать LLM, сохранить результат в кэш

### 3. Guardrails — валидация LLM output

LLM может вернуть:
- Оценку выше max_score
- overall_score ≠ сумме criterion scores
- Пустой feedback
- Оскорбительный контент
- Факты, не основанные на student work

Guardrails — это **программная валидация** после каждого LLM-вызова:

```python
def validate_assessment(result: AssessmentResponse, rubric: Rubric) -> list[str]:
    errors = []
    total = sum(cs.score for cs in result.criterion_scores)
    if result.overall_score != total:
        errors.append(f"overall_score {result.overall_score} != sum {total}")
    for cs in result.criterion_scores:
        if cs.score > cs.max_score:
            errors.append(f"{cs.criterion_name}: {cs.score} > max {cs.max_score}")
        if not cs.feedback.strip():
            errors.append(f"{cs.criterion_name}: empty feedback")
    return errors
```

Стратегия при ошибках: **retry с уточнённым промптом** — добавить ошибку в контекст и попросить модель исправить.

### 4. Rate limiting — контроль расходов

Проблемы без rate limiting:
- DDoS или баг → тысячи запросов к LLM API → огромный счёт
- Провайдер (Anthropic, OpenAI) возвращает 429 Too Many Requests
- Один пользователь монополизирует ресурсы

Реализация:
- **Per-user rate limit** — max N оценок в час/день на пользователя
- **Global rate limit** — max M запросов к API в минуту
- **Budget cap** — max $X в день, при превышении — отказ или fallback на дешёвую модель

### 5. Retry strategies

LLM API подвержены transient errors:
- **Rate limit (429)** — слишком много запросов → exponential backoff
- **Server error (500/503)** — временная проблема → retry через 1-2 секунды
- **Timeout** — долгий ответ → retry или fallback

Exponential backoff с jitter:

```python
import random

for attempt in range(max_retries):
    try:
        return await chain.ainvoke(data)
    except RateLimitError:
        wait = min(2 ** attempt + random.uniform(0, 1), max_wait)
        await asyncio.sleep(wait)
```

Jitter (случайная добавка) предотвращает **thundering herd** — когда все клиенты ретраят одновременно.

---

## Практические задания

### Задание 1: Model routing

**Цель:** автоматически выбирать модель на основе сложности задачи.

**Файлы:** `app/services/model_router.py`, `experiments/t10_routing.py`

**Критерии успеха:**
- Classifier определяет сложность: short/simple → haiku, long/complex → sonnet
- Экономия ≥ 30% при смешанной нагрузке (vs всегда sonnet)
- Качество haiku-оценок приемлемо для простых работ

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Implement model routing for the assessment system.

1. Create app/services/model_router.py:
   - Function classify_complexity(student_work: str, rubric: Rubric) -> str:
     Returns "simple" or "complex" based on:
     - Word count: <200 = simple, >=200 = complex
     - Paragraph count: <3 = simple
     - Rubric criteria count: <=3 = simple
     - Has citations (regex check): complex
   
   - Function get_model_for_complexity(complexity: str, config: Settings) -> ChatAnthropic:
     "simple" → ChatAnthropic(model="claude-haiku-4-20250414", ...)
     "complex" → ChatAnthropic(model="claude-sonnet-4-20250514", ...)

   - Function build_routed_chain(config: Settings) -> Runnable:
     Uses RunnableLambda to classify, then routes to appropriate model chain

2. Create experiments/t10_routing.py:
   - Test with 6 essays: 3 short/simple, 3 long/complex
   - For each: print complexity classification, model used, score, cost estimate
   - Compare total cost: routed vs all-sonnet
   - Compare quality: routed vs all-sonnet scores (should be close)

Run with: python -m experiments.t10_routing
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/services/model_router.py:

1. CLASSIFICATION: Are criteria for simple vs complex sensible?
2. TWO MODELS: Are haiku and sonnet actually used for different complexities?
3. COST SAVINGS: Is the cost comparison clear?
4. QUALITY: Is there evidence that haiku is "good enough" for simple tasks?
5. FLEXIBILITY: Can classification rules be easily adjusted?

Common mistakes:
- Only using word count (ignores other complexity signals)
- Not measuring quality difference between models
- Hardcoded model names instead of using config

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Guardrails

**Цель:** добавить валидацию LLM output с retry на невалидные результаты.

**Файлы:** `app/services/guardrails.py`, `experiments/t10_guardrails.py`

**Критерии успеха:**
- Валидация: scores в допустимых границах, overall = sum, feedback не пустой
- При невалидном результате: retry с добавлением ошибки в промпт
- Максимум 3 попытки, потом — ошибка

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Implement output guardrails for assessment.

1. Create app/services/guardrails.py:
   - Function validate_assessment(result: AssessmentResponse, rubric: Rubric) -> list[str]:
     Checks:
     - Each criterion score <= max_score and >= 0
     - overall_score == sum of criterion scores
     - max_overall_score == sum of max_scores
     - All criterion names match rubric criteria
     - No empty feedback strings
     - summary is non-empty
     - strengths has >= 1 item
     - improvements has >= 1 item
     Returns list of error messages (empty = valid)

   - Function assess_with_guardrails(chain, input_data, rubric, max_retries=3) -> AssessmentResponse:
     Loop:
       result = await chain.ainvoke(input_data)
       errors = validate_assessment(result, rubric)
       if not errors: return result
       else: add errors to input as correction context, retry
     After max_retries: raise GuardrailError with last errors

2. Create experiments/t10_guardrails.py:
   - Run normal assessment: validate output
   - Intentionally provoke invalid output (e.g., temperature=1.0, very short prompt)
   - Show guardrails catching and correcting issues
   - Print: attempt number, errors found, final result

Run with: python -m experiments.t10_guardrails
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/services/guardrails.py:

1. VALIDATION RULES: Are all the specified checks implemented?
2. RETRY LOGIC: Does it retry with error context added to the prompt?
3. MAX RETRIES: Is there a limit to prevent infinite loops?
4. ERROR REPORTING: Are validation errors clear and specific?
5. INTEGRATION: Can it be plugged into the existing assessment flow?

Common mistakes:
- Only checking overall_score, not per-criterion
- Retry without adding error context (same mistake will repeat)
- No max_retries limit
- Swallowing errors silently instead of logging

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: Semantic cache

**Цель:** кэшировать оценки похожих работ для экономии.

**Файлы:** `app/services/semantic_cache.py`, `experiments/t10_cache.py`

**Критерии успеха:**
- Похожие работы (cosine similarity > 0.95) получают кэшированный результат
- Cache hit не вызывает LLM = $0 стоимость
- Cache miss → вызов LLM + сохранение в кэш
- Измеримая экономия

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Implement semantic caching for assessments.

1. Create app/services/semantic_cache.py:
   - Class SemanticCache:
     - __init__(embedding_model, similarity_threshold=0.95)
     - Uses in-memory Chroma vector store for simplicity
     - async def get(self, student_work: str) -> AssessmentResponse | None:
       Embed student_work, search for similar cached entries
       If similarity > threshold: return cached AssessmentResponse
       Else: return None
     - async def set(self, student_work: str, result: AssessmentResponse):
       Store embedding + serialized result in vector store

   - Function build_cached_chain(chain, cache) -> Runnable:
     Check cache → if hit return → else invoke chain → store in cache → return

2. Create experiments/t10_cache.py:
   - Initialize semantic cache
   - Run assessment on essay A → cache miss, LLM called
   - Run assessment on essay A again (identical) → cache hit
   - Run assessment on essay A' (slightly rephrased) → cache hit (semantic match)
   - Run assessment on essay B (different topic) → cache miss
   - Print: cache hits/misses, time saved, cost saved

Run with: python -m experiments.t10_cache
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/services/semantic_cache.py:

1. EMBEDDING: Is the cache using embeddings for semantic matching?
2. THRESHOLD: Is similarity threshold configurable and sensible (0.95)?
3. SERIALIZATION: Can AssessmentResponse be stored and retrieved correctly?
4. INTEGRATION: Can it wrap the existing chain transparently?
5. METRICS: Are cache hits/misses tracked?

Common mistakes:
- Exact string matching instead of semantic matching
- Threshold too low (returns irrelevant cached results)
- Not serializing/deserializing AssessmentResponse correctly
- Cache grows without bounds (no eviction policy)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: Cost optimization

**Цель:** найти баланс цена/качество для оценок.

**Файлы:** `experiments/t10_cost_optimization.py`

**Критерии успеха:**
- Измерить стоимость одной оценки для разных конфигураций
- Сравнить: sonnet vs haiku, long prompt vs short, high vs low max_tokens
- Рекомендация: оптимальная конфигурация

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t10_cost_optimization.py — find optimal price/quality balance.

1. Define configurations to test:
   a. Baseline: sonnet, full prompt, max_tokens=4096
   b. Cheaper model: haiku, full prompt, max_tokens=4096
   c. Shorter prompt: sonnet, reduced prompt (no few-shot examples), max_tokens=4096
   d. Lower max_tokens: sonnet, full prompt, max_tokens=2048
   e. Optimized: haiku, reduced prompt, max_tokens=2048
   f. Full optimization: haiku + routing + cache

2. For each config:
   - Run on 5 golden dataset entries
   - Measure: input tokens, output tokens, cost, latency
   - Measure quality: MAE vs golden dataset expected scores

3. Print comparison table:
   Config          Cost/eval   Latency   MAE    Quality
   Baseline        $0.016      8.2s      4.2    Good
   Cheaper model   $0.003      2.1s      6.8    Acceptable
   Shorter prompt  $0.012      7.5s      5.1    Good
   Lower tokens    $0.014      6.0s      4.5    Good
   Optimized       $0.002      1.5s      7.2    Acceptable
   Full optimize   $0.001      0.5s      5.5    Good (with routing)

4. Recommendation: which config for which use case
   - High-stakes assessment: baseline
   - Bulk screening: cheaper model
   - Production default: routing + cache

Run with: python -m experiments.t10_cost_optimization
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t10_cost_optimization.py:

1. CONFIGURATIONS: Are multiple meaningful configs tested?
2. METRICS: Cost, latency, and quality all measured?
3. GOLDEN DATASET: Quality measured against expected scores?
4. COMPARISON TABLE: Clear side-by-side comparison?
5. RECOMMENDATION: Actionable advice on which config to use when?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Назови 3 стратегии model routing. Когда какая подходит?
- [ ] Как работает semantic cache? Что такое cosine similarity threshold?
- [ ] Какие проверки должны быть в guardrails для assessment? Назови 5.
- [ ] Что такое exponential backoff с jitter? Зачем jitter?
- [ ] Как посчитать стоимость одного LLM-вызова? Какие факторы влияют?
- [ ] Rate limiting: per-user vs global vs budget cap. Когда каждый нужен?

---

## Частые ошибки

### 1. Cache без учёта рубрики

```python
# Плохо: кэшируется только по тексту работы
cache_key = embed(student_work)

# Правильно: ключ = работа + рубрика (разные рубрики → разные оценки)
cache_key = embed(student_work + rubric.name)
```

### 2. Guardrails только на границах

```python
# Плохо: валидация только overall_score
if result.overall_score < 0 or result.overall_score > 100:
    retry()

# Хорошо: глубокая валидация
errors = validate_assessment(result, rubric)  # checks ALL fields
```

### 3. Retry без backoff

```python
# Плохо: мгновенные ретраи = amplify the problem
for _ in range(3):
    try: return await call_api()
    except: pass

# Хорошо: exponential backoff
for attempt in range(3):
    try: return await call_api()
    except RateLimitError:
        await asyncio.sleep(2 ** attempt + random.random())
```

### 4. Model routing без мониторинга

```python
# Плохо: routing без метрик — не знаешь, как он работает
model = route(complexity)

# Хорошо: логировать routing decisions
model, complexity = route(complexity)
logger.info(f"Routed to {model.model_name} for complexity={complexity}")
langfuse.score(trace_id=..., name="routing", value=complexity)
```

---

## Что читать дальше

- [LangChain Caching](https://python.langchain.com/docs/how_to/llm_caching/) — кэширование LLM
- [LangChain Fallbacks](https://python.langchain.com/docs/how_to/fallbacks/) — паттерн fallback
- [Guardrails AI](https://www.guardrailsai.com/) — фреймворк валидации
- [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits) — лимиты Claude API
- [Redis Semantic Cache](https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/) — Redis для vector search

**Поздравляю!** Ты прошёл все 10 тем AI Engineering курса. Следующий шаг — объединить все знания в production-ready assessment system.

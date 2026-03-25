# Тема 8: Observability

> **Пререквизиты:** [Тема 1-4](topic_01_prompt_engineering.md), рекомендуется [Тема 6](topic_06_langgraph_agents.md)
> **Где в проекте:** `app/services/`, `app/config.py`
> **Зависимости:** `langfuse` (группа `eval`)

---

## Теория

### 1. Зачем трейсинг для LLM

LLM-вызовы кардинально отличаются от обычных API-вызовов:
- **Недетерминизм** — один и тот же input может давать разные output.
- **Высокая стоимость** — каждый вызов стоит деньги (токены). Неоптимальный промпт = переплата.
- **Latency** — 2-30 секунд на вызов. Нужно видеть, где bottleneck.
- **Качество** — "работает" и "работает хорошо" — разные вещи. Нужно измерять.

Трейсинг для LLM-приложений — это как APM (Application Performance Monitoring) для обычных сервисов, но с фокусом на:
- **Промпты** — какой текст отправлен модели
- **Токены** — сколько input/output, какая стоимость
- **Latency** — время каждого шага в цепочке
- **Качество** — scores, user feedback, evaluation results

### 2. Langfuse — open-source LLM observability

Langfuse — open-source платформа для трейсинга LLM-приложений. Основные абстракции:

| Сущность | Описание |
|----------|----------|
| **Trace** | Полный запрос пользователя (от входа до ответа) |
| **Span** | Шаг внутри trace (preprocessing, retrieval, formatting) |
| **Generation** | LLM вызов (промпт → ответ, токены, стоимость, latency) |
| **Score** | Оценка качества (автоматическая или от пользователя) |

Иерархия: Trace → Span → Generation. Trace содержит несколько spans, span может содержать generation.

### 3. Callback handlers в LangChain

LangChain вызывает callback-функции на каждом этапе:

```python
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

handler = LangfuseCallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
)

result = await chain.ainvoke(
    {"student_work": "..."},
    config={"callbacks": [handler]},
)
```

Что происходит:
1. Chain начинает выполнение → `on_chain_start` → Langfuse создаёт Trace
2. Prompt рендерится → Langfuse записывает rendered prompt
3. LLM вызывается → `on_llm_start`/`on_llm_end` → Langfuse записывает Generation (prompt, completion, tokens, latency)
4. Output парсится → `on_chain_end` → Langfuse закрывает Trace

Всё это происходит **автоматически** — подключил callback, и каждый вызов трейсится.

### 4. Cost tracking

Langfuse автоматически считает стоимость по модели и количеству токенов:

```
Generation: claude-sonnet-4-20250514
  Input tokens:  1,250 → $0.00375
  Output tokens:   800 → $0.01200
  Total cost:           $0.01575
```

Это позволяет:
- Видеть стоимость **каждой** оценки
- Отслеживать расходы за день/неделю/месяц
- Обнаруживать аномалии (слишком длинные промпты, избыточные вызовы)

### 5. Prompt management

Langfuse хранит **версии промптов** — можно менять промпт в UI без деплоя кода:

```python
from langfuse import Langfuse

langfuse = Langfuse()
prompt = langfuse.get_prompt("assessment-prompt", version=3)
system_text = prompt.compile(rubric="...", few_shot_good="...")
```

Преимущества:
- **Версионирование** — откат к предыдущей версии одним кликом
- **A/B testing** — разные версии промпта для разных пользователей
- **Не-инженеры** — product manager может редактировать промпт без PR

---

## Практические задания

### Задание 1: Langfuse setup

**Цель:** подключить Langfuse к проекту и увидеть первый trace.

**Файлы:** `app/services/langfuse_handler.py`, обновление `app/config.py`

**Критерии успеха:**
- Langfuse запущен (cloud или Docker)
- Callback handler подключён к chain
- В Langfuse dashboard виден trace с prompt, completion, tokens

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Set up Langfuse integration for the assessment project.

1. Update app/config.py — add Langfuse settings:
   - langfuse_public_key: str = ""
   - langfuse_secret_key: str = ""
   - langfuse_host: str = "https://cloud.langfuse.com"
   - langfuse_enabled: bool = False

2. Update .env.example with new Langfuse variables

3. Create app/services/langfuse_handler.py:
   - Function create_langfuse_handler(config: Settings) -> CallbackHandler | None:
     If langfuse_enabled and keys are set: return LangfuseCallbackHandler
     Otherwise: return None
   - Function get_langfuse_config(config: Settings) -> dict:
     Returns {"callbacks": [handler]} if handler exists, else {}

4. Modify app/dependencies.py:
   - Add get_langfuse_config dependency
   - Pass langfuse callbacks to chain invocations

5. Create experiments/t8_langfuse_setup.py:
   - Run one assessment with Langfuse callback
   - Print: trace URL, token count, cost
   - Verify trace appears in Langfuse dashboard

Sign up at https://cloud.langfuse.com (free tier) or run Docker:
docker run -d --name langfuse -p 3000:3000 langfuse/langfuse

Run with: python -m experiments.t8_langfuse_setup
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the Langfuse integration:

1. CONFIG: Are Langfuse keys and host in Settings?
2. HANDLER: Is LangfuseCallbackHandler created correctly?
3. OPTIONAL: Does the app work normally when Langfuse is disabled?
4. INTEGRATION: Is the handler passed to chain via config={"callbacks": [handler]}?
5. EXPERIMENT: Can you verify traces appear in Langfuse?

Common mistakes:
- Hardcoded Langfuse keys (should be in .env)
- App crashes when Langfuse is disabled or unreachable
- Handler not passed to async calls (ainvoke needs config param)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Трейсинг цепочки

**Цель:** увидеть полную цепочку вызовов в Langfuse dashboard.

**Файлы:** `experiments/t8_tracing.py`

**Критерии успеха:**
- Trace показывает: prompt rendering → LLM call → parsing
- Видны: input/output каждого шага, latency, токены
- Можно кликнуть на generation и увидеть полный промпт

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t8_tracing.py demonstrating full chain tracing.

1. Run 3 assessments with different essays and Langfuse tracing:
   - Each assessment should have a unique trace name
   - Add metadata to traces: essay_length, rubric_name, timestamp
   - Use handler.trace(name=..., metadata=...) for custom trace attributes

2. For each assessment, print:
   - Trace ID and URL
   - Input tokens, output tokens, cost
   - Total latency
   - Scores per criterion (as Langfuse scores)

3. Add Langfuse scores programmatically:
   langfuse.score(
       trace_id=trace_id,
       name="overall_score",
       value=result.overall_score / result.max_overall_score,
   )

4. Print summary: average cost per assessment, average latency

After running, check Langfuse dashboard:
- All 3 traces should be visible
- Each trace should show the full chain
- Scores should be attached to traces

Run with: python -m experiments.t8_tracing
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t8_tracing.py:

1. TRACE NAMING: Are traces named descriptively (not just "LangChain")?
2. METADATA: Is useful metadata attached (essay length, rubric, timestamp)?
3. SCORES: Are Langfuse scores added programmatically?
4. MULTIPLE TRACES: Are 3 separate traces created?
5. COST SUMMARY: Is there a summary of costs?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: Prompt management

**Цель:** перенести промпт в Langfuse и загружать динамически.

**Файлы:** `experiments/t8_prompt_management.py`, модификация `app/services/`

**Критерии успеха:**
- Assessment prompt создан в Langfuse через API
- Chain загружает промпт из Langfuse вместо хардкода
- Изменение промпта в Langfuse UI отражается в оценках без деплоя

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Implement Langfuse prompt management.

1. Create experiments/t8_prompt_management.py:

   a. Upload the current assessment prompt to Langfuse:
      langfuse.create_prompt(
          name="assessment-system-prompt",
          prompt=ASSESSMENT_SYSTEM_PROMPT,
          config={"temperature": 0.3, "model": "claude-sonnet-4-20250514"},
          labels=["production"],
      )

   b. Create a function that loads prompt from Langfuse:
      def get_langfuse_prompt(name: str, version: int | None = None) -> str:
          prompt = langfuse.get_prompt(name, version=version)
          return prompt.compile(rubric=rubric, few_shot_good=..., few_shot_bad=...)

   c. Build chain using Langfuse prompt instead of hardcoded:
      system_text = get_langfuse_prompt("assessment-system-prompt")
      prompt = ChatPromptTemplate.from_messages([
          ("system", system_text),
          ("human", "Assess: {student_work}"),
      ])

   d. Demonstrate:
      - Run assessment with version 1 of prompt
      - Update prompt in Langfuse (add a tweak)
      - Run assessment with version 2
      - Compare results

Run with: python -m experiments.t8_prompt_management
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the prompt management implementation:

1. UPLOAD: Is the prompt correctly created in Langfuse?
2. VERSIONING: Can you load specific prompt versions?
3. COMPILE: Is prompt.compile() used with variables?
4. DYNAMIC: Does changing the prompt in Langfuse change the chain behavior?
5. FALLBACK: What happens if Langfuse is down? Is there a fallback to hardcoded prompt?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: Cost dashboard

**Цель:** построить view стоимости оценок.

**Файлы:** `experiments/t8_cost_dashboard.py`

**Критерии успеха:**
- Запрос данных из Langfuse API
- Отчёт: стоимость за оценку, за день, по модели
- Выявление дорогих оценок (длинные промпты, много токенов)

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t8_cost_dashboard.py — cost analysis from Langfuse data.

1. Query Langfuse for recent traces:
   langfuse = Langfuse()
   traces = langfuse.fetch_traces(limit=50)

2. For each trace, extract:
   - Total input tokens, output tokens
   - Cost (calculated from model pricing)
   - Latency
   - Trace name and metadata

3. Build report:
   - Per-assessment cost breakdown
   - Average cost per assessment
   - Most expensive assessments (top 5)
   - Token breakdown: input vs output
   - Model usage distribution

4. Print formatted table and summary

5. Bonus: calculate projected monthly cost at current usage rate

Run with: python -m experiments.t8_cost_dashboard
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t8_cost_dashboard.py:

1. DATA: Does it fetch real data from Langfuse?
2. METRICS: Are cost, tokens, and latency all reported?
3. AGGREGATION: Average, max, total costs calculated?
4. FORMATTING: Clear, readable output?
5. INSIGHTS: Does it highlight expensive or anomalous traces?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Назови 3 причины, почему LLM-приложения нуждаются в трейсинге больше, чем обычные API.
- [ ] Что такое Trace, Span, Generation в Langfuse? Как они связаны?
- [ ] Как callback handler подключается к LangChain chain?
- [ ] Зачем prompt management в Langfuse? Чем это лучше хардкода?
- [ ] Как считается стоимость LLM-вызова? Какие факторы на неё влияют?

---

## Частые ошибки

### 1. Langfuse как hard dependency

```python
# Плохо: приложение падает без Langfuse
handler = LangfuseCallbackHandler(...)  # crash если ключи не заданы

# Хорошо: graceful degradation
handler = None
if config.langfuse_enabled:
    try:
        handler = LangfuseCallbackHandler(...)
    except Exception:
        logger.warning("Langfuse unavailable, tracing disabled")
```

### 2. Не трейсить в development

```python
# Плохо: трейсинг только в production
if env == "production":
    callbacks = [langfuse_handler]

# Лучше: трейсинг везде, разные проекты в Langfuse
# Dev: langfuse project "assessment-dev"
# Prod: langfuse project "assessment-prod"
```

### 3. Не добавлять metadata к traces

```python
# Плохо: безымянный trace — невозможно найти
result = chain.ainvoke(data, config={"callbacks": [handler]})

# Хорошо: именованный trace с metadata
result = chain.ainvoke(data, config={
    "callbacks": [handler],
    "run_name": "essay_assessment",
    "metadata": {"rubric": "essay_default", "user_id": "teacher-1"},
})
```

---

## Что читать дальше

- [Langfuse Docs](https://langfuse.com/docs) — полная документация
- [Langfuse + LangChain](https://langfuse.com/docs/integrations/langchain/tracing) — интеграция
- [Langfuse Prompt Management](https://langfuse.com/docs/prompts/get-started) — управление промптами
- [LangSmith](https://docs.smith.langchain.com/) — альтернатива от LangChain (SaaS)

**Следующая тема:** [Тема 9: Evaluation](topic_09_evaluation.md) — как измерять качество LLM.

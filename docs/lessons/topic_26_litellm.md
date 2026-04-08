# Тема 26: LiteLLM — единый API для 100+ LLM-провайдеров

> **Пререквизиты:** [Тема 2](topic_02_langchain_lcel.md), [Тема 10](topic_10_production_patterns.md)
> **Зависимости:** `litellm`, `litellm[proxy]`

---

## Теория

### 1. Проблема: зоопарк LLM API

Каждый LLM-провайдер — свой SDK, свой формат запросов, свой формат ответов, своя система ошибок. При работе с несколькими провайдерами код превращается в хаос:

```python
if provider == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
elif provider == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(model="claude-sonnet-4-20250514", messages=messages, max_tokens=1000)
elif provider == "google":
    ...
```

Каждый новый провайдер — новый `elif`, новый SDK, новые edge cases. Fallback между провайдерами — ещё сложнее. Cost tracking — отдельная боль.

LiteLLM решает это: один интерфейс (OpenAI-совместимый) для 100+ провайдеров.

### 2. LiteLLM: что это

LiteLLM — Python-библиотека и proxy-сервер, предоставляющие единый OpenAI-совместимый интерфейс для всех LLM-провайдеров.

Два режима использования:

| Режим | Описание | Когда использовать |
|---|---|---|
| **Python SDK** | `litellm.completion()` — вызов из Python | Скрипты, notebooks, backend |
| **Proxy Server** | HTTP-сервер с OpenAI-совместимым API | Микросервисы, мультиязычные команды, централизованное управление |

Поддерживаемые провайдеры (выборка):

| Провайдер | Формат модели | Пример |
|---|---|---|
| OpenAI | `gpt-4o`, `o1-mini` | `litellm.completion(model="gpt-4o")` |
| Anthropic | `anthropic/claude-...` | `litellm.completion(model="anthropic/claude-sonnet-4-20250514")` |
| Google | `gemini/gemini-1.5-pro` | `litellm.completion(model="gemini/gemini-1.5-pro")` |
| AWS Bedrock | `bedrock/anthropic.claude-v2` | `litellm.completion(model="bedrock/...")` |
| Azure | `azure/gpt-4o` | `litellm.completion(model="azure/gpt-4o")` |
| Ollama | `ollama/llama3.1` | `litellm.completion(model="ollama/llama3.1")` |
| Groq | `groq/llama-3.1-70b` | `litellm.completion(model="groq/llama-3.1-70b")` |
| Mistral | `mistral/mistral-large` | `litellm.completion(model="mistral/mistral-large")` |
| Together AI | `together_ai/...` | `litellm.completion(model="together_ai/...")` |

### 3. Python SDK — litellm.completion()

Основная функция — `litellm.completion()`. Интерфейс идентичен OpenAI:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "Ты — оценщик эссе. Оцени от 1 до 10."},
        {"role": "user", "content": "AI is transforming society..."},
    ],
    temperature=0.0,
    max_tokens=1000,
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
```

Ответ — `ModelResponse`, совместимый с OpenAI. Один и тот же код работает для любого провайдера.

Async:

```python
response = await litellm.acompletion(
    model="openai/gpt-4o",
    messages=messages,
)
```

Streaming:

```python
response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### 4. Fallbacks и Routing

Killer-фича LiteLLM — встроенные fallbacks. Если основная модель недоступна, LiteLLM автоматически переключается на резервную:

```python
import litellm
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "assessment-model",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        },
        {
            "model_name": "assessment-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "assessment-model",
            "litellm_params": {
                "model": "groq/llama-3.1-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
            },
        },
    ],
    fallbacks=[
        {"assessment-model": ["assessment-model"]},
    ],
    routing_strategy="least-busy",
)

response = await router.acompletion(
    model="assessment-model",
    messages=messages,
)
```

Три модели под одним именем `assessment-model`. Если Claude недоступен — LiteLLM пробует GPT-4o, затем Llama через Groq.

Стратегии маршрутизации:

| Стратегия | Описание |
|---|---|
| `simple-shuffle` | Случайный выбор |
| `least-busy` | Модель с наименьшей очередью |
| `latency-based-routing` | Самая быстрая модель |
| `cost-based-routing` | Самая дешёвая модель |
| `usage-based-routing` | Лимиты по RPM/TPM |

### 5. Cost Tracking

LiteLLM знает цены всех моделей и автоматически считает стоимость:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
)

cost = litellm.completion_cost(completion_response=response)
print(f"Cost: ${cost:.6f}")
```

Budget control:

```python
import litellm

litellm.max_budget = 10.0
litellm.budget_duration = "daily"

try:
    response = litellm.completion(model="openai/gpt-4o", messages=messages)
except litellm.BudgetExceededError:
    print("Daily budget exceeded!")
```

### 6. LiteLLM Proxy — OpenAI-совместимый сервер

LiteLLM Proxy — HTTP-сервер, который выставляет OpenAI-совместимый API. Любое приложение, работающее с OpenAI, может подключиться к proxy без изменения кода.

Конфигурация `litellm_config.yaml`:

```yaml
model_list:
  - model_name: "assessment-fast"
    litellm_params:
      model: "anthropic/claude-haiku-3.5"
      api_key: "os.environ/ANTHROPIC_API_KEY"

  - model_name: "assessment-quality"
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"
      api_key: "os.environ/ANTHROPIC_API_KEY"

  - model_name: "assessment-quality"
    litellm_params:
      model: "openai/gpt-4o"
      api_key: "os.environ/OPENAI_API_KEY"

router_settings:
  routing_strategy: "latency-based-routing"
  num_retries: 3
  timeout: 120

general_settings:
  master_key: "sk-litellm-master-key"
  max_budget: 100
  budget_duration: "monthly"
```

Запуск:

```bash
litellm --config litellm_config.yaml --port 4000
```

Использование из любого OpenAI-клиента:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-master-key",
)

response = client.chat.completions.create(
    model="assessment-quality",
    messages=[{"role": "user", "content": "Оцени эссе..."}],
)
```

### 7. Виртуальные ключи и rate limits

Proxy поддерживает виртуальные API-ключи с индивидуальными лимитами:

```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-litellm-master-key" \
  -d '{"max_budget": 10, "budget_duration": "monthly", "models": ["assessment-fast"]}'
```

Каждый разработчик / сервис получает свой ключ с бюджетом. Централизованное управление расходами без раздачи реальных API-ключей провайдеров.

### 8. Интеграция с LangChain

LiteLLM работает с LangChain через `ChatLiteLLM` или через proxy как OpenAI endpoint:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-key",
    model="assessment-quality",
)

chain = prompt | llm | parser
result = chain.invoke({"essay": "AI is transforming..."})
```

Вся инфраструктура LiteLLM (fallbacks, routing, cost tracking) прозрачна для LangChain — chain не знает, что за proxy стоят разные модели.

---

## Справочник API

### litellm.completion

```python
import litellm

response = litellm.completion(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stream: bool = False,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    n: int = 1,
    stop: str | list[str] | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    metadata: dict | None = None,
) -> ModelResponse
```

### litellm.acompletion

```python
response = await litellm.acompletion(
    model: str,
    messages: list[dict],
    **kwargs,
) -> ModelResponse
```

### litellm.embedding

```python
response = litellm.embedding(
    model: str,
    input: list[str],
    api_key: str | None = None,
) -> EmbeddingResponse
```

### litellm.completion_cost

```python
cost = litellm.completion_cost(
    completion_response: ModelResponse | None = None,
    model: str | None = None,
    prompt: str = "",
    completion: str = "",
) -> float
```

### Router

```python
from litellm import Router

router = Router(
    model_list: list[dict],
    fallbacks: list[dict] | None = None,
    routing_strategy: str = "simple-shuffle",
    num_retries: int = 3,
    timeout: float = 120,
    retry_after: float = 0,
    enable_pre_call_checks: bool = True,
)

response = await router.acompletion(model: str, messages: list, **kwargs)
```

### Callbacks

```python
import litellm

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

litellm.callbacks = [my_custom_callback]
```

---

## Практика

### Пример 1: Один интерфейс для разных моделей

Один и тот же код для OpenAI, Anthropic, Groq.

```python
import litellm

models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-3.5",
    "groq/llama-3.1-70b-versatile",
]

messages = [
    {"role": "system", "content": "Оцени эссе от 1 до 10. Ответь одним числом."},
    {"role": "user", "content": "AI is transforming society in profound ways."},
]

for model in models:
    try:
        response = litellm.completion(model=model, messages=messages, temperature=0.0)
        cost = litellm.completion_cost(completion_response=response)
        print(
            f"{model:40s} | "
            f"Score: {response.choices[0].message.content.strip():5s} | "
            f"Tokens: {response.usage.total_tokens:5d} | "
            f"Cost: ${cost:.6f}"
        )
    except Exception as e:
        print(f"{model:40s} | Error: {e}")
```

### Пример 2: Router с fallbacks

Автоматическое переключение между провайдерами при ошибках.

```python
import os
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "assessor",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        },
        {
            "model_name": "assessor",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ],
    fallbacks=[{"assessor": ["assessor"]}],
    routing_strategy="latency-based-routing",
    num_retries=2,
    timeout=30,
)

messages = [
    {"role": "system", "content": "Оцени эссе от 1 до 10."},
    {"role": "user", "content": "The rapid advancement of AI poses fundamental questions..."},
]

import asyncio

async def assess():
    response = await router.acompletion(model="assessor", messages=messages)
    print(f"Model used: {response.model}")
    print(f"Response: {response.choices[0].message.content}")

asyncio.run(assess())
```

### Пример 3: Cost tracking и budget

Отслеживание расходов по моделям и установка бюджетов.

```python
import litellm

models_and_essays = [
    ("anthropic/claude-sonnet-4-20250514", "AI is transforming society..."),
    ("openai/gpt-4o", "Machine learning uses data to learn."),
    ("anthropic/claude-haiku-3.5", "The ethical implications of AI extend beyond accuracy."),
    ("openai/gpt-4o-mini", "AI robots will take over."),
]

total_cost = 0.0
cost_by_provider = {}

for model, essay in models_and_essays:
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": "Оцени эссе от 1 до 10. Дай фидбек."},
            {"role": "user", "content": essay},
        ],
        temperature=0.0,
    )

    cost = litellm.completion_cost(completion_response=response)
    total_cost += cost

    provider = model.split("/")[0]
    cost_by_provider[provider] = cost_by_provider.get(provider, 0) + cost

    print(f"{model:45s} | ${cost:.6f} | {response.usage.total_tokens} tokens")

print(f"\nTotal: ${total_cost:.6f}")
for provider, cost in sorted(cost_by_provider.items()):
    print(f"  {provider}: ${cost:.6f}")
```

### Пример 4: Proxy Server с конфигурацией

Настройка и запуск LiteLLM Proxy.

**Файл `litellm_config.yaml`:**

```yaml
model_list:
  - model_name: "essay-assessor"
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"
      api_key: "os.environ/ANTHROPIC_API_KEY"
      max_tokens: 2000
    model_info:
      description: "Primary assessment model"

  - model_name: "essay-assessor"
    litellm_params:
      model: "openai/gpt-4o"
      api_key: "os.environ/OPENAI_API_KEY"
    model_info:
      description: "Fallback assessment model"

  - model_name: "essay-assessor-fast"
    litellm_params:
      model: "anthropic/claude-haiku-3.5"
      api_key: "os.environ/ANTHROPIC_API_KEY"
    model_info:
      description: "Fast/cheap model for simple assessments"

router_settings:
  routing_strategy: "latency-based-routing"
  num_retries: 3
  timeout: 120
  retry_after: 5

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: "sk-litellm-master"
  max_budget: 50
  budget_duration: "monthly"
```

```bash
litellm --config litellm_config.yaml --port 4000 --detailed_debug
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="sk-litellm-master")

response = client.chat.completions.create(
    model="essay-assessor",
    messages=[
        {"role": "system", "content": "Оцени эссе от 1 до 10."},
        {"role": "user", "content": "AI is transforming society..."},
    ],
)

print(response.choices[0].message.content)
```

### Пример 5: Интеграция с LangChain через proxy

LangChain chain, работающий через LiteLLM proxy.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm_quality = ChatOpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-master",
    model="essay-assessor",
    temperature=0.0,
)

llm_fast = ChatOpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-master",
    model="essay-assessor-fast",
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — оценщик эссе. Оцени от 1 до 10. Дай развёрнутый фидбек."),
    ("human", "{essay}"),
])

chain_quality = prompt | llm_quality | StrOutputParser()
chain_fast = prompt | llm_fast | StrOutputParser()

essay = "The rapid advancement of AI poses fundamental questions..."

print("=== Quality model ===")
print(chain_quality.invoke({"essay": essay}))

print("\n=== Fast model ===")
print(chain_fast.invoke({"essay": essay}))
```

### Пример 6: Callbacks для observability

Подключение Langfuse для трекинга всех вызовов через LiteLLM.

```python
import litellm
import os

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

models = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o-mini",
]

for model in models:
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": "Оцени эссе от 1 до 10."},
            {"role": "user", "content": "AI is transforming society..."},
        ],
        metadata={
            "trace_name": "essay-assessment",
            "trace_metadata": {"model": model, "experiment": "model-comparison"},
        },
    )
    cost = litellm.completion_cost(completion_response=response)
    print(f"{model}: {response.choices[0].message.content[:50]}... (${cost:.6f})")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | Единый интерфейс для разных провайдеров |
| 2 | Router + fallbacks |
| 3 | Cost tracking по моделям и провайдерам |
| 4 | Proxy Server с конфигурацией |
| 5 | LangChain через LiteLLM proxy |
| 6 | Observability через callbacks (Langfuse) |

---

## Чеклист самопроверки

- [ ] Зачем нужен LiteLLM? Какую проблему он решает?
- [ ] В чём разница между Python SDK (`litellm.completion`) и Proxy Server?
- [ ] Как работают fallbacks в Router? Что происходит при недоступности основной модели?
- [ ] Какие стратегии маршрутизации поддерживает Router? Когда какую использовать?
- [ ] Как LiteLLM считает стоимость вызовов? Как установить бюджет?
- [ ] Что такое виртуальные ключи в LiteLLM Proxy? Зачем они нужны?
- [ ] Как подключить LangChain к LiteLLM Proxy без изменения кода?
- [ ] Как настроить Langfuse callbacks для трекинга через LiteLLM?
- [ ] Когда использовать LiteLLM vs прямой SDK провайдера?
- [ ] Как LiteLLM обрабатывает различия в API разных провайдеров (max_tokens, stop sequences)?

---

## Частые ошибки

### 1. Неправильный формат model name

```python
litellm.completion(model="claude-sonnet-4-20250514", messages=messages)
```

```python
litellm.completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
```

LiteLLM требует prefix провайдера: `anthropic/`, `openai/`, `gemini/`, `ollama/`. Без prefix LiteLLM не знает, какой API вызвать. Исключение: модели OpenAI работают без prefix (`gpt-4o`), но лучше указывать явно.

### 2. Забыли API-ключ

```python
litellm.completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
```

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

litellm.completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
```

LiteLLM читает ключи из environment variables. Формат: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`. Альтернативно — передать `api_key` параметром.

### 3. Router без fallbacks

```python
router = Router(model_list=[
    {"model_name": "main", "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514"}},
    {"model_name": "backup", "litellm_params": {"model": "openai/gpt-4o"}},
])

response = await router.acompletion(model="main", messages=messages)
```

```python
router = Router(
    model_list=[
        {"model_name": "assessor", "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514"}},
        {"model_name": "assessor", "litellm_params": {"model": "openai/gpt-4o"}},
    ],
    fallbacks=[{"assessor": ["assessor"]}],
)

response = await router.acompletion(model="assessor", messages=messages)
```

Разные `model_name` не создают fallback-группу. Для fallback нужны одинаковые `model_name` + явный `fallbacks`. Без этого при ошибке Claude не переключится на GPT-4o.

### 4. Sync в async FastAPI

```python
@app.post("/assess")
async def assess(essay: str):
    response = litellm.completion(model="openai/gpt-4o", messages=messages)
    return response
```

```python
@app.post("/assess")
async def assess(essay: str):
    response = await litellm.acompletion(model="openai/gpt-4o", messages=messages)
    return response
```

`litellm.completion()` блокирует event loop в async-контексте. Используй `litellm.acompletion()` в FastAPI, aiohttp и других async-фреймворках.

### 5. Proxy без master_key в production

```yaml
general_settings: {}
```

```yaml
general_settings:
  master_key: "sk-very-strong-random-key"
```

Без `master_key` proxy доступен всем без аутентификации. Любой может вызвать модели и потратить бюджет. В production всегда устанавливай `master_key`.

---

## Что читать дальше

- [LiteLLM Documentation](https://docs.litellm.ai/) — полная документация
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) — список поддерживаемых провайдеров
- [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start) — настройка proxy server
- [LiteLLM Router](https://docs.litellm.ai/docs/routing) — fallbacks и routing
- [LiteLLM Budget Manager](https://docs.litellm.ai/docs/budget_manager) — управление бюджетом
- [LiteLLM + LangChain](https://docs.litellm.ai/docs/langchain/langchain) — интеграция
- [LiteLLM Callbacks](https://docs.litellm.ai/docs/observability/callbacks) — Langfuse, Helicone, и др.
- [LiteLLM GitHub](https://github.com/BerriAI/litellm) — исходный код

**Предыдущая тема:** [Тема 25: DSPy](topic_25_dspy.md) — программирование LLM-пайплайнов.
**Следующая тема:** [Тема 27: CrewAI](topic_27_crewai.md) — фреймворк для мульти-агентных систем.

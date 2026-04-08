# Тема 23: Databricks для AI Engineering — MLflow, Model Serving, Vector Search

> **Пререквизиты:** [Тема 5](topic_05_rag.md), [Тема 8](topic_08_observability.md), [Тема 20](topic_20_deployment.md)
> **Зависимости:** `databricks-sdk`, `databricks-langchain`, `mlflow`, `langchain-community`

---

## Теория

### 1. Databricks как платформа для AI Engineering

Databricks — unified data + AI платформа, построенная поверх Apache Spark. Для AI-инженера Databricks решает несколько проблем одновременно: хранение данных (Delta Lake), эксперименты и трекинг (MLflow), деплой моделей (Model Serving), векторный поиск (Vector Search) и governance (Unity Catalog).

Почему это важно для LLM-приложений:

- **Данные рядом с моделями.** RAG-приложение требует доступа к документам, embeddings, метаданным. В Databricks всё это лежит в Delta Lake — не нужно копировать данные между системами.
- **MLflow встроен.** Каждый эксперимент, каждый прогон промпта, каждая метрика — автоматически трекается. Не нужно поднимать отдельный сервер.
- **Model Serving с GPU.** Развернуть LLM-эндпоинт (свою модель или Foundation Model) — один клик или API-вызов. Автоскейлинг, мониторинг, A/B тесты — из коробки.
- **Vector Search.** Managed vector database, интегрированная с Delta Lake. Данные обновляются автоматически при изменении исходной таблицы.
- **Unity Catalog.** Единая система прав доступа: кто видит какие данные, какие модели, какие endpoints. Критично для enterprise.

Архитектура LLM-приложения на Databricks:

```
┌──────────────────────────────────────────────────┐
│              Databricks Workspace                 │
│                                                   │
│  ┌────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ Delta Lake │  │   MLflow    │  │   Unity    │ │
│  │  (данные,  │  │ (трекинг,  │  │  Catalog   │ │
│  │  embeddings)│  │  модели)   │  │ (governance)│ │
│  └─────┬──────┘  └──────┬──────┘  └────────────┘ │
│        │                │                         │
│  ┌─────▼──────┐  ┌──────▼──────┐                  │
│  │  Vector    │  │   Model     │                  │
│  │  Search    │  │  Serving    │                  │
│  │  Index     │  │  Endpoint   │                  │
│  └─────┬──────┘  └──────┬──────┘                  │
│        │                │                         │
│  ┌─────▼────────────────▼──────┐                  │
│  │     LangChain / Agent       │                  │
│  │     (notebooks, jobs)       │                  │
│  └─────────────────────────────┘                  │
└──────────────────────────────────────────────────┘
```

### 2. Foundation Model APIs и Model Serving

Databricks предоставляет два способа работать с LLM:

**Foundation Model APIs** — доступ к предобученным моделям (DBRX, Llama, Mixtral, MPT) через managed endpoints. Не нужно заниматься инфраструктурой — Databricks сам управляет GPU, масштабированием, обновлениями.

**Custom Model Serving** — деплой своей модели (fine-tuned, собственная архитектура) через MLflow. Модель регистрируется в Unity Catalog, создаётся serving endpoint с автоскейлингом.

Типы endpoints:

| Тип | Описание | Когда использовать |
|---|---|---|
| Foundation Model API | Managed LLM (DBRX, Llama 3, Mixtral) | Быстрый старт, стандартные задачи |
| External Model | Прокси к OpenAI, Anthropic, Cohere | Единый интерфейс + governance для внешних LLM |
| Custom Model | Своя модель через MLflow | Fine-tuned модели, специализированные задачи |
| Feature Serving | Real-time feature lookup | Обогащение запросов фичами |

**External Models** — ключевая фича для enterprise. Вместо того чтобы каждый разработчик хранил свой API-ключ OpenAI, создаётся единый endpoint через Databricks. Преимущества:

- Централизованное управление ключами (ключ хранится в Databricks secret scope, не у разработчиков)
- Rate limiting и cost tracking на уровне платформы
- Аудит: кто вызвал какую модель, когда, с какими данными
- Governance через Unity Catalog

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoint = w.serving_endpoints.create(
    name="claude-sonnet-proxy",
    config={
        "served_entities": [{
            "name": "claude-sonnet",
            "external_model": {
                "name": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "anthropic_config": {
                    "anthropic_api_key": "{{secrets/ai-keys/anthropic}}"
                }
            }
        }],
    },
)
```

После создания endpoint доступен через OpenAI-совместимый API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<workspace>.databricks.com/serving-endpoints",
    api_key=dbutils.secrets.get("tokens", "pat"),
)

response = client.chat.completions.create(
    model="claude-sonnet-proxy",
    messages=[{"role": "user", "content": "Оцени это эссе..."}],
    max_tokens=1000,
)
```

### 3. MLflow для LLM — трекинг, evaluation, deployment

MLflow — open-source платформа для ML lifecycle. В контексте LLM MLflow предоставляет:

**Трекинг экспериментов.** Каждый прогон промпта — это run в MLflow. Логируются: промпт, параметры (temperature, model), метрики (quality score, latency, cost), артефакты (полный ответ LLM).

```python
import mlflow

mlflow.set_experiment("/Users/me/essay-assessment")

with mlflow.start_run(run_name="claude-sonnet-v2-prompt"):
    mlflow.log_param("model", "claude-sonnet-4-20250514")
    mlflow.log_param("temperature", 0.0)
    mlflow.log_param("prompt_version", "v2")
    mlflow.log_text(system_prompt, "prompts/system.txt")

    result = chain.invoke({"essay": essay, "rubric": rubric})

    mlflow.log_metric("score", result["score"])
    mlflow.log_metric("latency_ms", result["latency_ms"])
    mlflow.log_metric("cost_usd", result["cost_usd"])
    mlflow.log_text(result["feedback"], "outputs/feedback.txt")
```

**MLflow Evaluate для LLM.** Встроенные метрики для оценки качества LLM-ответов:

```python
import mlflow
import pandas as pd

eval_data = pd.DataFrame({
    "inputs": ["Оцени эссе: AI is good.", "Оцени эссе: The intersection of AI and labor..."],
    "ground_truth": ["2", "8"],
})

results = mlflow.evaluate(
    model="endpoints:/claude-sonnet-proxy",
    data=eval_data,
    targets="ground_truth",
    model_type="question-answering",
    extra_metrics=[
        mlflow.metrics.exact_match(),
        mlflow.metrics.latency(),
        mlflow.metrics.toxicity(),
        mlflow.metrics.answer_relevance(),
    ],
)

print(results.metrics)
print(results.tables["eval_results_table"])
```

**MLflow Models для LLM.** Логирование LangChain chain как MLflow-модели:

```python
import mlflow

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model=chain,
        artifact_path="assessment_chain",
        input_example={"essay": "text", "rubric": "criteria"},
    )

loaded = mlflow.langchain.load_model(model_info.model_uri)
result = loaded.invoke({"essay": "AI is transforming...", "rubric": "..."})
```

Залогированную модель можно зарегистрировать в Unity Catalog и развернуть как serving endpoint — весь LangChain chain будет работать как REST API.

### 4. Databricks Vector Search

Vector Search — managed vector database, интегрированная с Delta Lake. Ключевое отличие от standalone vector databases (Pinecone, Qdrant): данные синхронизируются автоматически.

Архитектура:

```
Delta Lake Table          Vector Search Index
┌──────────────┐          ┌──────────────────┐
│ id │ text    │  ──sync──▶│ id │ embedding  │
│ 1  │ "doc A" │          │ 1  │ [0.1, ...]  │
│ 2  │ "doc B" │          │ 2  │ [0.3, ...]  │
└──────────────┘          └──────────────────┘
     ↓ INSERT                    ↓ auto-update
┌──────────────┐          ┌──────────────────┐
│ 3  │ "doc C" │  ──sync──▶│ 3  │ [0.2, ...]  │
└──────────────┘          └──────────────────┘
```

Три типа индексов:

| Тип | Описание | Embedding |
|---|---|---|
| Delta Sync Index (computed) | Embeddings вычисляются автоматически | Databricks считает |
| Delta Sync Index (self-managed) | Ты считаешь embeddings, Databricks синхронизирует | Свои |
| Direct Vector Access Index | Ручное управление (CRUD API) | Свои |

**Delta Sync Index** — рекомендуемый вариант. Ты добавляешь документы в Delta-таблицу, Databricks автоматически вычисляет embeddings и обновляет индекс. При изменении или удалении строки в таблице индекс обновляется.

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

index = w.vector_search_indexes.create_index(
    name="main.ai_bootcamp.essay_index",
    endpoint_name="vs-endpoint",
    primary_key="id",
    index_type="DELTA_SYNC",
    delta_sync_index_spec={
        "source_table": "main.ai_bootcamp.essays",
        "embedding_source_columns": [
            {"name": "text", "model_endpoint_name": "databricks-bge-large-en"}
        ],
        "pipeline_type": "TRIGGERED",
    },
)
```

Поиск:

```python
results = w.vector_search_indexes.query_index(
    index_name="main.ai_bootcamp.essay_index",
    columns=["id", "text", "score", "subject"],
    query_text="impact of artificial intelligence on education",
    num_results=5,
    filters_json='{"subject": "computer_science"}',
)

for doc in results.result.data_array:
    print(f"Score: {doc[2]:.3f} | {doc[1][:100]}...")
```

### 5. LangChain + Databricks — интеграция

Библиотека `databricks-langchain` предоставляет нативную интеграцию:

| Компонент | LangChain класс | Что делает |
|---|---|---|
| Chat Model | `ChatDatabricks` | Вызов Foundation/External/Custom endpoints |
| Embeddings | `DatabricksEmbeddings` | Вычисление embeddings через serving endpoint |
| Vector Store | `DatabricksVectorSearch` | Поиск в Vector Search Index |
| MLflow Callback | `MlflowCallbackHandler` | Автоматический трекинг в MLflow |

Полная RAG-цепочка на Databricks:

```python
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatDatabricks(endpoint="claude-sonnet-proxy", temperature=0.0)
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

vector_store = DatabricksVectorSearch(
    index_name="main.ai_bootcamp.essay_index",
    embedding=embeddings,
    text_column="text",
    columns=["id", "text", "score", "subject"],
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — эксперт-оценщик эссе. Используй примеры ранее оцененных "
        "работ для калибровки. Контекст:\n{context}"
    )),
    ("human", "Оцени это эссе от 1 до 10:\n\n{essay}"),
])

chain = (
    {"context": retriever | format_docs, "essay": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("AI is transforming society in profound ways...")
```

### 6. Unity Catalog — governance для AI

Unity Catalog — единая система управления данными, моделями и AI-артефактами. Для AI Engineering важны:

**Модели в каталоге.** Зарегистрированная модель доступна по трёхуровневому имени: `catalog.schema.model_name`. Права доступа, версионирование, lineage — всё через Unity Catalog.

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=chain,
        artifact_path="chain",
        registered_model_name="main.ai_bootcamp.assessment_chain",
    )
```

**Функции в каталоге.** Unity Catalog Functions — Python UDF, зарегистрированные как tools для LLM:

```sql
CREATE FUNCTION main.ai_bootcamp.count_words(text STRING)
RETURNS INT
LANGUAGE PYTHON
AS $$
  return len(text.split())
$$;
```

Эту функцию можно использовать как LangChain tool через `UCFunctionToolkit`:

```python
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import UCFunctionToolkit

llm = ChatDatabricks(endpoint="claude-sonnet-proxy")

toolkit = UCFunctionToolkit(
    warehouse_id="abc123",
    function_names=["main.ai_bootcamp.count_words"],
)

tools = toolkit.get_tools()
llm_with_tools = llm.bind_tools(tools)
```

**AI Gateway.** Прокси-слой для LLM endpoints с rate limiting, cost tracking и аудитом. Каждый запрос логируется: кто вызвал, когда, сколько токенов, какая модель. Это критично для compliance и cost management в enterprise.

### 7. Databricks Workflows — оркестрация AI-пайплайнов

Workflows (Jobs) — scheduler для задач. Для AI-приложений типичные сценарии:

- **Ежедневный пересчёт embeddings.** Новые документы → Delta Lake → trigger Vector Search sync.
- **Scheduled evaluation.** Прогон golden dataset через модель, сохранение метрик в MLflow.
- **Data pipeline.** Ingestion → preprocessing → chunking → embeddings → Delta Lake.
- **Model retraining.** Fine-tuning на новых данных, регистрация новой версии, A/B тест.

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, CronSchedule

w = WorkspaceClient()

job = w.jobs.create(
    name="daily-embedding-refresh",
    tasks=[
        Task(
            task_key="ingest",
            notebook_task=NotebookTask(
                notebook_path="/Repos/ai-bootcamp/notebooks/01_ingest"
            ),
        ),
        Task(
            task_key="embed",
            notebook_task=NotebookTask(
                notebook_path="/Repos/ai-bootcamp/notebooks/02_embed"
            ),
            depends_on=[{"task_key": "ingest"}],
        ),
        Task(
            task_key="sync_index",
            notebook_task=NotebookTask(
                notebook_path="/Repos/ai-bootcamp/notebooks/03_sync_vs"
            ),
            depends_on=[{"task_key": "embed"}],
        ),
    ],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 6 * * ?",
        timezone_id="Europe/Moscow",
    ),
)
```

### 8. Mosaic AI Agent Framework

Databricks Mosaic AI Agent Framework — набор инструментов для разработки, тестирования и деплоя AI-агентов на платформе. Ключевые компоненты:

**Agent development.** Разработка агента в notebook с автоматическим трекингом в MLflow. Поддержка LangChain, LangGraph, произвольного Python-кода.

**Agent evaluation.** Встроенные LLM-judges для оценки качества ответов агента:

| Judge | Что оценивает |
|---|---|
| Groundedness | Ответ основан на retrieved контексте |
| Relevance | Ответ релевантен вопросу |
| Safety | Ответ безопасен (нет toxicity, bias) |
| Chunk relevance | Каждый chunk релевантен запросу |

```python
import mlflow

eval_results = mlflow.evaluate(
    data=eval_dataset,
    model=agent,
    model_type="databricks-agent",
    evaluator_config={
        "databricks-agent": {
            "metrics": ["groundedness", "relevance", "safety", "chunk_relevance"],
        }
    },
)
```

**Agent deployment.** Деплой агента как serving endpoint с Review App — встроенным UI для тестирования и сбора feedback от стейкхолдеров.

```python
import mlflow

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model=agent,
        artifact_path="agent",
        registered_model_name="main.ai_bootcamp.essay_agent",
    )

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

endpoint = w.serving_endpoints.create(
    name="essay-agent",
    config={
        "served_entities": [{
            "entity_name": "main.ai_bootcamp.essay_agent",
            "entity_version": "1",
            "scale_to_zero_enabled": True,
        }]
    },
)
```

Review App доступен по URL endpoint'а — стейкхолдеры могут тестировать агента через чат-интерфейс и оставлять feedback (thumbs up/down, комментарии). Feedback автоматически сохраняется в inference table для дальнейшей оптимизации.

---

## Справочник API

### ChatDatabricks

```python
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(
    endpoint: str,
    target_uri: str = "databricks",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    extra_params: dict | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `endpoint` | `str` | Имя serving endpoint |
| `temperature` | `float` | Температура генерации |
| `max_tokens` | `int \| None` | Максимум токенов в ответе |
| `extra_params` | `dict \| None` | Дополнительные параметры модели |

### DatabricksEmbeddings

```python
from databricks_langchain import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(
    endpoint: str,
    target_uri: str = "databricks",
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `endpoint` | `str` | Имя embedding serving endpoint |

### DatabricksVectorSearch

```python
from databricks_langchain import DatabricksVectorSearch

vs = DatabricksVectorSearch(
    index_name: str,
    endpoint: str | None = None,
    embedding: Embeddings | None = None,
    text_column: str | None = None,
    columns: list[str] | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `index_name` | `str` | Полное имя индекса (`catalog.schema.index`) |
| `embedding` | `Embeddings \| None` | Embedding model (для self-managed индексов) |
| `text_column` | `str \| None` | Колонка с текстом |
| `columns` | `list[str] \| None` | Какие колонки возвращать |

### UCFunctionToolkit

```python
from databricks_langchain.uc_ai import UCFunctionToolkit

toolkit = UCFunctionToolkit(
    warehouse_id: str,
    function_names: list[str],
)
tools = toolkit.get_tools()
```

| Параметр | Тип | Описание |
|---|---|---|
| `warehouse_id` | `str` | ID SQL warehouse для выполнения функций |
| `function_names` | `list[str]` | Список UC функций (`catalog.schema.function`) |

### MLflow LLM Evaluate

```python
import mlflow

results = mlflow.evaluate(
    model: str | callable,
    data: pd.DataFrame,
    targets: str | None = None,
    model_type: str | None = None,
    extra_metrics: list | None = None,
    evaluator_config: dict | None = None,
)
```

| Параметр | Тип | Описание |
|---|---|---|
| `model` | `str \| callable` | Модель или URI endpoint'а (`endpoints:/name`) |
| `data` | `pd.DataFrame` | Данные для evaluation |
| `targets` | `str` | Колонка с ground truth |
| `model_type` | `str` | Тип: `"question-answering"`, `"text"`, `"databricks-agent"` |
| `extra_metrics` | `list` | Дополнительные метрики |
| `evaluator_config` | `dict` | Конфигурация evaluator'ов |

### Databricks SDK — Vector Search

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.vector_search_indexes.create_index(
    name: str,
    endpoint_name: str,
    primary_key: str,
    index_type: str,
    delta_sync_index_spec: dict | None = None,
    direct_access_index_spec: dict | None = None,
)

w.vector_search_indexes.query_index(
    index_name: str,
    columns: list[str],
    query_text: str | None = None,
    query_vector: list[float] | None = None,
    filters_json: str | None = None,
    num_results: int = 10,
    score_threshold: float | None = None,
)
```

---

## Практика

### Пример 1: Foundation Model API через LangChain

Подключение к Databricks Foundation Model или External Model endpoint через LangChain.

```python
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.0)

messages = [
    SystemMessage(content="Ты — эксперт-оценщик студенческих эссе. Оцени от 1 до 10."),
    HumanMessage(content=(
        "Оцени это эссе:\n\n"
        "AI is transforming society in profound ways. Studies show that "
        "47% of jobs are at risk of automation. However, new roles are "
        "being created in AI development, ethics, and oversight."
    )),
]

response = llm.invoke(messages)
print(response.content)
print(f"Tokens: {response.usage_metadata}")
```

### Пример 2: RAG с Vector Search

Полный RAG-пайплайн: документы в Delta Lake → Vector Search → LangChain retriever → ответ.

```python
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatDatabricks(endpoint="claude-sonnet-proxy", temperature=0.0)
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

vs = DatabricksVectorSearch(
    index_name="main.ai_bootcamp.essay_index",
    embedding=embeddings,
    text_column="text",
    columns=["id", "text", "previous_score", "subject"],
)
retriever = vs.as_retriever(search_kwargs={"k": 3, "filters": {"subject": "cs"}})


def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(
            f"Эссе (оценка {doc.metadata['previous_score']}/10):\n{doc.page_content[:500]}"
        )
    return "\n\n---\n\n".join(formatted)


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — эксперт-оценщик эссе. Ниже примеры ранее оцененных работ "
        "для калибровки твоих оценок.\n\n{context}"
    )),
    ("human", "Оцени это эссе от 1 до 10. Объясни оценку.\n\n{essay}"),
])

chain = (
    {"context": retriever | format_docs, "essay": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("The rapid advancement of AI poses fundamental questions...")
print(result)
```

### Пример 3: MLflow трекинг LLM-экспериментов

Сравнение двух промптов через MLflow: логируем параметры, метрики и артефакты каждого прогона.

```python
import mlflow
import time

mlflow.set_experiment("/Users/me/essay-assessment-prompts")

prompts = {
    "v1_basic": "Оцени эссе от 1 до 10.",
    "v2_detailed": (
        "Ты — строгий академический рецензент. Оцени эссе по критериям: "
        "1) Тезис и аргументация (0-3), 2) Доказательная база (0-3), "
        "3) Структура и логика (0-2), 4) Язык и стиль (0-2). "
        "Итого от 0 до 10. Обоснуй каждый балл."
    ),
}

test_essays = [
    {"text": "AI is good. It helps people.", "expected": 2},
    {"text": "The intersection of AI and labor economics presents nuanced challenges...", "expected": 8},
    {"text": "Machine learning uses data.", "expected": 1},
]

for prompt_name, system_prompt in prompts.items():
    with mlflow.start_run(run_name=prompt_name):
        mlflow.log_param("prompt_version", prompt_name)
        mlflow.log_text(system_prompt, "prompts/system.txt")

        total_error = 0
        start = time.perf_counter()

        for essay in test_essays:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Оцени:\n\n{essay['text']}"},
            ])
            predicted = extract_score(response.content)
            total_error += abs(predicted - essay["expected"])

        elapsed = time.perf_counter() - start
        mae = total_error / len(test_essays)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("avg_latency_s", elapsed / len(test_essays))
        mlflow.log_metric("total_essays", len(test_essays))

        print(f"{prompt_name}: MAE={mae:.2f}, Latency={elapsed:.1f}s")
```

### Пример 4: UC Functions как LLM tools

Создание Python-функций в Unity Catalog и использование их как инструментов для LLM-агента.

```sql
CREATE OR REPLACE FUNCTION main.ai_bootcamp.count_words(text STRING)
RETURNS INT
LANGUAGE PYTHON
AS $$
  return len(text.split())
$$;

CREATE OR REPLACE FUNCTION main.ai_bootcamp.check_citations(text STRING)
RETURNS STRING
LANGUAGE PYTHON
AS $$
  import re
  citations = re.findall(r'\([\w\s]+,\s*\d{4}\)', text)
  if not citations:
      return "Нет цитирований. Рекомендуется добавить ссылки на источники."
  return f"Найдено {len(citations)} цитирований: {', '.join(citations)}"
$$;
```

```python
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import UCFunctionToolkit
from langgraph.prebuilt import create_react_agent

llm = ChatDatabricks(endpoint="claude-sonnet-proxy")

toolkit = UCFunctionToolkit(
    warehouse_id="abc123def456",
    function_names=[
        "main.ai_bootcamp.count_words",
        "main.ai_bootcamp.check_citations",
    ],
)

agent = create_react_agent(llm, toolkit.get_tools())

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Проанализируй это эссе: подсчитай слова, проверь цитирования, "
            "затем дай оценку.\n\n"
            "The rapid advancement of AI (Russell, 2019) poses fundamental questions "
            "about the nature of work. Brynjolfsson and McAfee (2014) argue for a "
            "'race against the machine'."
        ),
    }]
})

print(result["messages"][-1].content)
```

### Пример 5: MLflow Evaluate — оценка качества агента

Запуск evaluation с встроенными LLM-judges для RAG-системы.

```python
import mlflow
import pandas as pd

eval_data = pd.DataFrame({
    "request": [
        "Оцени эссе: AI is changing the world in many ways.",
        "Оцени эссе: The ethical implications of AI extend beyond accuracy metrics.",
        "Оцени эссе: Machine learning is a subset of AI.",
    ],
    "expected_response": [
        "Оценка 3-4/10: поверхностный анализ без конкретики",
        "Оценка 7-8/10: глубокий анализ этических аспектов",
        "Оценка 1-2/10: перечисление фактов без анализа",
    ],
    "expected_retrieved_context": [
        "Примеры эссе с оценками 3-4 из базы",
        "Примеры эссе с оценками 7-8 из базы",
        "Примеры эссе с оценками 1-2 из базы",
    ],
})

results = mlflow.evaluate(
    data=eval_data,
    model=agent,
    model_type="databricks-agent",
    evaluator_config={
        "databricks-agent": {
            "metrics": [
                "groundedness",
                "relevance",
                "safety",
                "chunk_relevance",
            ],
        }
    },
)

print("Общие метрики:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.3f}")

print("\nДетальные результаты:")
print(results.tables["eval_results_table"])
```

### Пример 6: Полный пайплайн — от данных до endpoint'а

Типичный end-to-end workflow на Databricks: загрузка данных → Vector Search → RAG chain → MLflow → serving.

```python
from databricks.sdk import WorkspaceClient
import mlflow

w = WorkspaceClient()

spark.sql("""
    CREATE TABLE IF NOT EXISTS main.ai_bootcamp.essays (
        id STRING,
        text STRING,
        subject STRING,
        score INT,
        feedback STRING,
        created_at TIMESTAMP
    ) USING DELTA
""")

essays_df = spark.createDataFrame([
    ("e1", "AI is transforming society...", "cs", 7, "Good analysis", "2026-01-15"),
    ("e2", "Machine learning uses data.", "cs", 2, "Too superficial", "2026-01-16"),
    ("e3", "The ethical implications of AI...", "ethics", 9, "Excellent depth", "2026-01-17"),
])
essays_df.write.mode("append").saveAsTable("main.ai_bootcamp.essays")

w.vector_search_indexes.sync_index(index_name="main.ai_bootcamp.essay_index")

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatDatabricks(endpoint="claude-sonnet-proxy", temperature=0.0)
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

vs = DatabricksVectorSearch(
    index_name="main.ai_bootcamp.essay_index",
    embedding=embeddings,
    text_column="text",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — оценщик эссе. Контекст:\n{context}"),
    ("human", "Оцени от 1 до 10:\n\n{essay}"),
])

chain = (
    {"context": vs.as_retriever(search_kwargs={"k": 3}) | format_docs, "essay": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

mlflow.set_experiment("/Users/me/essay-rag-pipeline")

with mlflow.start_run(run_name="rag-v1"):
    mlflow.langchain.log_model(
        lc_model=chain,
        artifact_path="rag_chain",
        registered_model_name="main.ai_bootcamp.essay_rag_chain",
    )

endpoint = w.serving_endpoints.create(
    name="essay-rag-endpoint",
    config={
        "served_entities": [{
            "entity_name": "main.ai_bootcamp.essay_rag_chain",
            "entity_version": "1",
            "scale_to_zero_enabled": True,
        }]
    },
)

print(f"Endpoint ready: {endpoint.name}")
print(f"URL: https://<workspace>.databricks.com/serving-endpoints/{endpoint.name}/invocations")
```

**Связь примеров с теорией:**

| Пример | Концепция |
|---|---|
| 1 | Foundation Model API, ChatDatabricks |
| 2 | Vector Search + LangChain RAG chain |
| 3 | MLflow трекинг, сравнение промптов |
| 4 | Unity Catalog Functions как LLM tools |
| 5 | MLflow Evaluate, LLM-judges |
| 6 | End-to-end: Delta Lake → Vector Search → RAG → MLflow → Serving |

---

## Чеклист самопроверки

- [ ] В чём преимущество Databricks Vector Search перед standalone vector databases (Pinecone, Qdrant)? Назови 2 причины.
- [ ] Что такое External Model endpoint? Зачем проксировать OpenAI/Anthropic через Databricks?
- [ ] Чем Delta Sync Index отличается от Direct Vector Access Index? Когда использовать каждый?
- [ ] Как залогировать LangChain chain в MLflow и развернуть его как serving endpoint?
- [ ] Что такое Unity Catalog Functions и как их использовать как LLM tools?
- [ ] Какие встроенные LLM-judges есть в Mosaic AI Agent Framework? Что каждый оценивает?
- [ ] Как настроить автоматическую синхронизацию Delta Lake → Vector Search?
- [ ] Зачем нужен AI Gateway? Какие проблемы он решает в enterprise?
- [ ] Опиши end-to-end workflow: данные → embeddings → RAG → evaluation → deployment.
- [ ] В чём разница между Foundation Model API и Custom Model Serving?

---

## Частые ошибки

### 1. Хардкод API-ключей вместо secret scope

```python
llm = ChatDatabricks(endpoint="my-endpoint", api_key="sk-abc123...")
```

```python
llm = ChatDatabricks(endpoint="my-endpoint")
```

В Databricks аутентификация происходит автоматически через workspace credentials. API-ключи внешних провайдеров (OpenAI, Anthropic) хранятся в secret scope endpoint'а, не в коде. Хардкод ключей — угроза безопасности и нарушение governance.

### 2. Computed embeddings + свой embedding model

```python
vs = DatabricksVectorSearch(
    index_name="main.default.my_computed_index",
    embedding=DatabricksEmbeddings(endpoint="my-custom-embeddings"),
)
```

```python
vs = DatabricksVectorSearch(
    index_name="main.default.my_computed_index",
    text_column="text",
)
```

Если индекс создан с `embedding_source_columns` (computed embeddings), Databricks сам вычисляет embeddings. Передача своей embedding model приведёт к несовпадению векторов при поиске. Свой embedding нужен только для self-managed индексов.

### 3. Забыли sync после добавления данных

```python
essays_df.write.mode("append").saveAsTable("main.ai_bootcamp.essays")
results = w.vector_search_indexes.query_index(...)
```

```python
essays_df.write.mode("append").saveAsTable("main.ai_bootcamp.essays")
w.vector_search_indexes.sync_index(index_name="main.ai_bootcamp.essay_index")
import time; time.sleep(30)
results = w.vector_search_indexes.query_index(...)
```

Delta Sync Index с `pipeline_type="TRIGGERED"` не обновляется автоматически. Нужен явный `sync_index()`. Даже с `pipeline_type="CONTINUOUS"` есть задержка в несколько минут. Для тестов добавляй `sleep` после sync.

### 4. MLflow evaluate без правильного model_type

```python
results = mlflow.evaluate(
    model=agent,
    data=eval_data,
    model_type="question-answering",
)
```

```python
results = mlflow.evaluate(
    model=agent,
    data=eval_data,
    model_type="databricks-agent",
    evaluator_config={
        "databricks-agent": {
            "metrics": ["groundedness", "relevance", "safety"],
        }
    },
)
```

Для RAG-агентов на Databricks используй `model_type="databricks-agent"` — он включает специализированные LLM-judges (groundedness, chunk_relevance). Стандартный `"question-answering"` не оценивает retrieval quality.

### 5. Serving endpoint без scale-to-zero

```python
endpoint = w.serving_endpoints.create(
    name="my-agent",
    config={
        "served_entities": [{
            "entity_name": "main.default.my_model",
            "entity_version": "1",
            "workload_size": "Medium",
        }]
    },
)
```

```python
endpoint = w.serving_endpoints.create(
    name="my-agent",
    config={
        "served_entities": [{
            "entity_name": "main.default.my_model",
            "entity_version": "1",
            "scale_to_zero_enabled": True,
        }]
    },
)
```

Без `scale_to_zero_enabled` endpoint работает 24/7 даже без трафика. Для dev и staging это может стоить сотни долларов в месяц. Включай scale-to-zero для всех non-production endpoint'ов. Компромисс: cold start ~30-60 секунд при первом запросе.

---

## Что читать дальше

- [Databricks Documentation — Generative AI](https://docs.databricks.com/en/generative-ai/index.html) — полная документация по GenAI на Databricks
- [Databricks LangChain Integration](https://docs.databricks.com/en/large-language-models/langchain.html) — интеграция с LangChain
- [MLflow LLM Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html) — evaluation для LLM
- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) — документация Vector Search
- [Mosaic AI Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/index.html) — фреймворк для AI-агентов
- [Unity Catalog AI Functions](https://docs.databricks.com/en/generative-ai/unity-catalog-ai-functions.html) — UC Functions как LLM tools
- [databricks-langchain PyPI](https://pypi.org/project/databricks-langchain/) — Python-пакет
- [Databricks SDK for Python](https://docs.databricks.com/en/dev-tools/sdk-python.html) — SDK документация
- [MLflow + Databricks](https://docs.databricks.com/en/mlflow/index.html) — MLflow на Databricks

**Предыдущая тема:** [Тема 22: Prompt Optimization](topic_22_prompt_optimization.md) — автоматическая оптимизация промптов.
**Следующая тема:** [Тема 24: PydanticAI](topic_24_pydantic_ai.md) — type-safe агенты на Pydantic.

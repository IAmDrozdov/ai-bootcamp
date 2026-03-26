# Тема 19: Vector Databases — сравнение и выбор

> **Пререквизиты:** [Тема 5: RAG](topic_05_rag.md)
> **Что добавляем в проект:** `app/api/v1/vector.py`, `app/services/vector_stores/chroma_store.py`, `app/services/vector_stores/pgvector_store.py`, `app/services/vector_stores/qdrant_store.py`, `app/schemas/vector.py`
> **Зависимости:** `langchain-chroma`, `langchain-postgres`, `langchain-qdrant`, `pgvector`, `qdrant-client`, `asyncpg`

---

## Теория

### 1. Зачем отдельная тема про vector DB

В теме 5 (RAG) мы подключили ChromaDB как единственное хранилище векторов. Это работало для прототипа: запустил `Chroma.from_documents()`, получил retriever, вставил в chain. Но в production-системе выбор vector database — одно из ключевых архитектурных решений, которое определяет latency, масштабируемость, стоимость и операционную сложность на годы вперёд.

Почему нельзя просто оставить Chroma:

- **Performance.** При 100 тысячах документов ChromaDB в embedded mode начинает заметно тормозить. При миллионах — становится непригодным.
- **Scaling.** ChromaDB не имеет встроенного шардирования и репликации. Один инстанс — один потолок.
- **Filtering.** В production-системе оценки нужно фильтровать по предмету, курсу, году, студенту. ChromaDB поддерживает базовую фильтрацию, но Qdrant или pgvector предлагают значительно более мощные возможности.
- **Операционная сложность.** Managed-сервисы (Pinecone) берут ops на себя. Self-hosted (Qdrant, pgvector) требуют мониторинга, бэкапов, обновлений.
- **Стоимость.** Разница между embedded Chroma ($0), pgvector (стоимость уже имеющегося Postgres), managed Pinecone ($70+/мес за serverless) — существенна.

В этой теме мы:
1. Разберём, как vector search работает на уровне алгоритмов
2. Сравним 5 vector databases: ChromaDB, pgvector, Pinecone, Qdrant, Weaviate
3. Реализуем унифицированный интерфейс для 3 из них (Chroma, pgvector, Qdrant)
4. Добавим benchmark-эндпоинт для сравнения latency
5. Реализуем hybrid search (keyword + semantic)

### 2. Как работает vector search — основы

#### Embedding: текст → вектор

Embedding-модель преобразует текст произвольной длины в вектор фиксированной размерности. Модель `text-embedding-3-small` (OpenAI) даёт вектор из 1536 float'ов, `all-MiniLM-L6-v2` (open-source) — 384 float'а.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = await embeddings.aembed_query("Эссе про влияние ИИ на образование")
len(vector)  # 1536
type(vector[0])  # float
```

Каждая координата вектора — это числовое значение, кодирующее определённый аспект семантики. Близкие по смыслу тексты получают близкие векторы в этом пространстве.

#### Similarity metrics: cosine, euclidean, dot product

Три основные метрики расстояния между векторами:

| Метрика | Формула | Диапазон | Когда использовать |
|---------|---------|----------|-------------------|
| **Cosine similarity** | cos(θ) = (A·B) / (‖A‖·‖B‖) | [-1, 1] | Нормализованные embeddings (OpenAI, Cohere). Стандарт по умолчанию. |
| **Euclidean (L2)** | √Σ(aᵢ - bᵢ)² | [0, ∞) | Когда абсолютная величина вектора важна. Редко лучше cosine. |
| **Dot product** | Σ(aᵢ · bᵢ) | (-∞, +∞) | Если векторы уже нормализованы — эквивалентно cosine, но быстрее (нет деления). |

Практическое правило: если используешь OpenAI или Cohere embeddings (нормализованные до единичной длины), cosine и dot product дают одинаковый ranking. Dot product чуть быстрее вычислительно. Для большинства задач — cosine.

#### Approximate Nearest Neighbor (ANN)

Точный поиск (brute-force) сравнивает запрос с каждым вектором в базе — O(N). При 10 миллионах документов и 1536-мерных векторах это ~60 ГБ данных и секунды на один запрос. В production — неприемлемо.

ANN-алгоритмы находят «приблизительно ближайших соседей» — не гарантированно лучших, но очень близких к лучшим, за время значительно меньше O(N).

**HNSW (Hierarchical Navigable Small World)**

Графовый алгоритм. Строит многоуровневый граф, где каждый уровень — подмножество точек предыдущего. Поиск начинается с верхнего уровня (немного точек, грубая навигация) и спускается вниз (больше точек, точная навигация).

Характеристики:
- Время поиска: O(log N)
- Потребление памяти: высокое — хранит граф связей
- Время построения индекса: медленное (часы для миллионов)
- Recall: 95-99.5% при правильных параметрах
- Параметры: `M` (число связей на точку), `ef_construction` (тщательность при построении), `ef_search` (тщательность при поиске)

HNSW — стандарт де-факто в большинстве vector databases. Используется в Qdrant, pgvector, Chroma, Weaviate.

**IVF (Inverted File Index)**

Кластерный алгоритм. Все векторы разбиваются на `nlist` кластеров через k-means. При поиске — определяется ближайший кластер(ы), ищем только внутри них.

Характеристики:
- Время поиска: O(N/nlist × nprobe)
- Потребление памяти: умеренное
- Время построения: быстрое (быстрее HNSW)
- Recall: зависит от nprobe — больше кластеров проверяем, выше recall, медленнее
- Параметры: `nlist` (число кластеров), `nprobe` (сколько проверять при поиске)

IVF подходит, когда нужен быстрый build индекса и компромисс recall/speed настраивается в runtime.

**Flat index**

Точный поиск, O(N). Гарантирует recall = 100%. Подходит только для коллекций до ~50k документов, где latency не критична.

#### Trade-off: recall vs latency vs memory

```
Recall (точность)
  ↑
  │  Flat ●                    (100% recall, высокая latency)
  │
  │         ● HNSW ef=500     (99.5% recall, умеренная latency, высокая memory)
  │
  │       ● HNSW ef=100       (97% recall, низкая latency, высокая memory)
  │
  │     ● IVF nprobe=10       (95% recall, низкая latency, умеренная memory)
  │
  │   ● IVF nprobe=1          (85% recall, минимальная latency, умеренная memory)
  │
  └──────────────────────────→ Latency
```

Для нашего проекта (десятки тысяч эссе) HNSW с параметрами по умолчанию — оптимальный выбор. При масштабировании до миллионов — стоит рассмотреть IVF с quantization.

### 3. ChromaDB — embedded vector store

#### Позиционирование

ChromaDB — самый простой способ начать работу с vector search. Встраивается в процесс приложения (embedded mode), не требует отдельного сервера, конфигурации или Docker. Идеален для прототипирования, разработки, тестирования и небольших production-систем (до ~100k документов).

#### Архитектура

Два режима работы:

1. **In-process (embedded)** — библиотека работает внутри Python-процесса. Данные в памяти или на диске.
2. **Client-server** — отдельный процесс Chroma Server, клиент подключается по HTTP.

```python
import chromadb

client = chromadb.Client()

client = chromadb.PersistentClient(path="./chroma_data")

client = chromadb.HttpClient(host="localhost", port=8000)
```

#### Коллекции и операции

```python
collection = client.create_collection(
    name="student_essays",
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=["Эссе текст 1", "Эссе текст 2"],
    metadatas=[{"subject": "math"}, {"subject": "physics"}],
    ids=["essay_1", "essay_2"],
)

results = collection.query(
    query_texts=["влияние ИИ на образование"],
    n_results=5,
    where={"subject": "math"},
    where_document={"$contains": "искусственный интеллект"},
)
```

#### LangChain-интеграция

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="student_essays",
    persist_directory="./chroma_data",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "filter": {"subject": "math"}},
)
```

#### Filtering

ChromaDB поддерживает два типа фильтров:

| Тип | Параметр | Пример | Описание |
|-----|----------|--------|----------|
| Metadata | `where` | `{"subject": "math"}` | Фильтр по метаданным документа |
| Document content | `where_document` | `{"$contains": "ИИ"}` | Фильтр по содержимому текста |

Операторы metadata-фильтров: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.

Логические операторы: `$and`, `$or`.

```python
results = collection.query(
    query_texts=["..."],
    where={
        "$and": [
            {"subject": {"$eq": "math"}},
            {"year": {"$gte": 2024}},
        ]
    },
)
```

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Zero-config — pip install и готово | Нет production-grade scaling |
| Быстрый старт, минимум кода | Нет репликации и шардирования |
| Хорош для dev/test/прототипов | Ограниченная фильтрация (нет SQL) |
| Persistent mode для небольших данных | Нет транзакций и ACID |
| Бесплатный и open-source | Производительность падает >100k docs |

### 4. pgvector — PostgreSQL extension

#### Позиционирование

pgvector — расширение для PostgreSQL, добавляющее тип данных `vector` и операторы поиска ближайших соседей. Главное преимущество: если приложение уже использует PostgreSQL (а большинство — да), не нужен отдельный vector database. Реляционные данные (пользователи, рубрики, оценки) и векторы живут в одной базе, с полной мощью SQL, JOIN'ов, транзакций и ACID.

#### Установка и настройка

```sql
CREATE EXTENSION vector;

CREATE TABLE essay_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    subject VARCHAR(100),
    year INTEGER,
    embedding vector(1536)
);
```

#### Типы индексов

| Тип индекса | Алгоритм | Время build | Время search | Recall | Когда использовать |
|-------------|----------|-------------|--------------|--------|-------------------|
| `ivfflat` | IVF | Быстрое | Умеренное | 90-97% | Часто обновляемые данные |
| `hnsw` | HNSW | Медленное | Быстрое | 95-99.5% | Read-heavy workloads |
| Без индекса | Flat | — | Медленное | 100% | <50k записей |

```sql
CREATE INDEX ON essay_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX ON essay_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

#### Операторы расстояния

| Оператор | Метрика | Класс операций |
|----------|---------|---------------|
| `<=>` | Cosine distance | `vector_cosine_ops` |
| `<->` | Euclidean (L2) distance | `vector_l2_ops` |
| `<#>` | Negative inner product | `vector_ip_ops` |

```sql
SELECT id, content, subject,
       1 - (embedding <=> '[0.12, -0.34, ...]') AS similarity
FROM essay_embeddings
WHERE subject = 'math' AND year >= 2024
ORDER BY embedding <=> '[0.12, -0.34, ...]'
LIMIT 5;
```

Обратите внимание: `<=>` возвращает distance (меньше = ближе), поэтому для similarity используем `1 - distance`.

#### LangChain-интеграция

```python
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+asyncpg://user:pass@localhost:5432/bootcamp"

vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    connection=CONNECTION_STRING,
    collection_name="student_essays",
    pre_delete_collection=False,
)

results = await vectorstore.asimilarity_search_with_score(
    query="влияние ИИ на образование",
    k=5,
    filter={"subject": "math"},
)
```

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Одна БД для всего — relational + vector | Масштабирование ограничено одним Postgres |
| ACID-транзакции, JOIN, полноценный SQL | Индексация медленнее, чем у специализированных DB |
| Мощная фильтрация — WHERE-клаузы SQL | Потребляет RAM Postgres-инстанса |
| Familiar — SQL знает каждый бэкенд-разработчик | Нет multi-tenancy из коробки |
| Бэкапы, мониторинг — уже настроены | Нет sparse vectors (нет нативного hybrid search) |

### 5. Pinecone — managed cloud

#### Позиционирование

Pinecone — полностью управляемый vector database. Нет серверов, нет операций, нет конфигурации инфраструктуры. Платишь — работает. Идеален для команд, которые не хотят (или не могут) поддерживать инфраструктуру vector database.

#### Serverless vs Pods

| Характеристика | Serverless | Pods |
|---------------|------------|------|
| Модель оплаты | Pay-per-query | Pay-per-node |
| Масштабирование | Автоматическое | Вручную (replicas, pods) |
| Стоимость при малой нагрузке | Низкая | Высокая (платишь за простой) |
| Стоимость при высокой нагрузке | Может быть высокой | Предсказуемая |
| Latency | Выше (cold start) | Ниже (dedicated) |

Для нашего проекта (AI Assessment System с умеренной нагрузкой) serverless — разумный выбор.

#### Namespaces

Namespaces — логическое разделение данных внутри одного index. Каждый namespace полностью изолирован: поиск по одному namespace не видит данные другого.

```python
index.upsert(vectors=[...], namespace="math_2024")
index.upsert(vectors=[...], namespace="physics_2024")

results = index.query(vector=[...], top_k=5, namespace="math_2024")
```

В нашей системе оценок: один namespace на предмет/курс — чистое разделение без перекрёстного загрязнения результатов.

#### Metadata filtering

```python
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "subject": {"$eq": "math"},
        "year": {"$gte": 2024},
        "score": {"$gt": 70},
    },
)
```

Поддерживаемые операторы: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`. Логические: `$and`, `$or`.

#### Sparse-dense vectors: hybrid search

Pinecone нативно поддерживает sparse vectors — это разреженные векторы, используемые для keyword search (BM25). Можно отправить и dense (semantic) и sparse (keyword) вектор в одном запросе:

```python
results = index.query(
    vector=dense_embedding,
    sparse_vector={"indices": [102, 3547, 8901], "values": [0.8, 1.2, 0.5]},
    top_k=5,
)
```

#### LangChain-интеграция

```python
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    index_name="student-essays",
    namespace="math_2024",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Zero-ops — нет инфраструктуры | Vendor lock-in — данные в чужом облаке |
| Auto-scaling | Стоимость — от $70/мес за serverless |
| Нативный hybrid search | Latency — network round-trip |
| Мощная metadata-фильтрация | Нет self-hosted варианта |
| Готовые SDK и интеграции | Ограниченный контроль над индексами |

### 6. Qdrant — open-source powerhouse

#### Позиционирование

Qdrant — open-source vector database с акцентом на production-ready features. Можно запускать self-hosted (Docker, Kubernetes) или использовать managed cloud. Мощный filtering, поддержка нескольких векторов в одной точке, quantization для экономии памяти.

#### Архитектура

Qdrant предоставляет gRPC и REST API. Данные организованы в коллекции. Каждая запись — point, содержащий:
- `id` — уникальный идентификатор
- `vector` — один или несколько именованных векторов
- `payload` — произвольные метаданные (JSON)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="student_essays",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

client.upsert(
    collection_name="student_essays",
    points=[
        PointStruct(
            id=1,
            vector=[0.12, -0.34, ...],
            payload={"subject": "math", "year": 2024, "score": 85},
        ),
    ],
)
```

#### Filtering: payload-based pre-filtering

Qdrant выполняет фильтрацию **до** vector search (pre-filtering), а не после (post-filtering). Это критическое отличие: post-filtering может вернуть меньше k результатов, если большинство ближайших соседей отфильтровано. Pre-filtering гарантирует ровно k результатов.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.query_points(
    collection_name="student_essays",
    query=[0.12, -0.34, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="subject", match=MatchValue(value="math")),
            FieldCondition(key="year", range=Range(gte=2024)),
        ],
        must_not=[
            FieldCondition(key="score", range=Range(lt=30)),
        ],
    ),
    limit=5,
)
```

Поддерживаемые условия: `must` (AND), `should` (OR), `must_not` (NOT). Типы: `MatchValue`, `MatchAny`, `MatchExcept`, `Range`, `GeoBoundingBox`, `ValuesCount`.

#### Quantization: сжатие векторов

Quantization уменьшает размер векторов в памяти, ускоряя поиск за счёт незначительной потери recall.

| Тип | Сжатие | Потеря recall | Описание |
|-----|--------|---------------|----------|
| Scalar | 4x | <1% | float32 → uint8. Лучший баланс. |
| Product (PQ) | 16-64x | 2-5% | Разбивает вектор на подвекторы. Для очень больших коллекций. |
| Binary | 32x | 3-10% | float32 → 1 bit. Максимальное сжатие. |

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.create_collection(
    collection_name="student_essays",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True),
    ),
)
```

#### Multi-vector: несколько векторов в одной точке

Qdrant поддерживает именованные векторы — можно хранить несколько embeddings для одного документа:

```python
client.create_collection(
    collection_name="multimodal_essays",
    vectors_config={
        "text": VectorParams(size=1536, distance=Distance.COSINE),
        "summary": VectorParams(size=384, distance=Distance.COSINE),
    },
)
```

Применение в нашем проекте: `text` — embedding полного эссе, `summary` — embedding краткого описания. Поиск по summary быстрее (384 vs 1536 dim), но менее точен.

#### LangChain-интеграция

```python
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    url="http://localhost:6333",
    collection_name="student_essays",
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)
```

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Мощный payload-based filtering (pre-filter) | Ещё один сервис в инфраструктуре |
| Multi-vector — несколько embeddings на точку | Требует отдельного мониторинга |
| Quantization для экономии RAM | Сложнее ChromaDB в настройке |
| Open-source + managed cloud | Нет встроенного SQL |
| Production-ready, battle-tested | Миграции схемы — вручную |
| gRPC для высокой производительности | |

### 7. Weaviate — модульная архитектура

#### Позиционирование

Weaviate — векторная база данных с модульной архитектурой. Ключевая идея: база данных не просто хранит и ищет векторы, а интегрирует AI-модули (vectorizer, ranker, generator) напрямую. Можно отправить текст — Weaviate сам создаст embedding через встроенный модуль.

#### GraphQL API

Weaviate использует GraphQL как основной API. Это мощно, но создаёт learning curve:

```graphql
{
  Get {
    StudentEssay(
      nearText: { concepts: ["влияние ИИ на образование"] }
      where: {
        operator: And
        operands: [
          { path: ["subject"], operator: Equal, valueText: "math" }
          { path: ["year"], operator: GreaterThanEqual, valueInt: 2024 }
        ]
      }
      limit: 5
    ) {
      content
      subject
      year
      _additional { distance certainty }
    }
  }
}
```

#### Modules

| Модуль | Назначение | Пример |
|--------|-----------|--------|
| `text2vec-openai` | Векторизация через OpenAI | Автоматический embedding при insert |
| `text2vec-cohere` | Векторизация через Cohere | Альтернативный провайдер |
| `generative-openai` | Генерация через OpenAI | RAG внутри самой Weaviate |
| `reranker-cohere` | Переранжирование результатов | Улучшение recall |
| `qna-openai` | Q&A через OpenAI | Встроенный question answering |

При включённом модуле `text2vec-openai` вставка текста автоматически создаёт embedding:

```python
client.data_object.create(
    data_object={"content": "Текст эссе...", "subject": "math"},
    class_name="StudentEssay",
)
```

#### Hybrid search

Weaviate имеет встроенный hybrid search — комбинация BM25 (keyword) и vector (semantic) поиска с настраиваемым балансом:

```python
result = collection.query.hybrid(
    query="влияние ИИ на образование",
    alpha=0.5,
    limit=5,
)
```

Параметр `alpha`:
- `alpha=0` — чистый keyword (BM25)
- `alpha=1` — чистый semantic (vector)
- `alpha=0.5` — равный баланс

#### Multi-tenancy

Weaviate поддерживает нативную multi-tenancy — изоляцию данных между клиентами на уровне хранилища:

```python
collection = client.collections.create(
    name="StudentEssay",
    multi_tenancy_config=Configure.multi_tenancy(enabled=True),
)

collection.tenants.create([Tenant(name="school_a"), Tenant(name="school_b")])
```

Каждый tenant имеет свой изолированный набор данных. В нашей системе: один tenant на школу/университет.

#### LangChain-интеграция

```python
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    client=weaviate_client,
    index_name="StudentEssay",
)
```

#### Плюсы и минусы

| Плюсы | Минусы |
|-------|--------|
| Модули — встроенная векторизация | Сложнее в настройке, чем Chroma/pgvector |
| Нативный hybrid search | GraphQL learning curve |
| Multi-tenancy из коробки | Больше ресурсов (RAM, CPU) |
| Мощная фильтрация | Модульная система может быть избыточной |
| Active community + managed cloud | Медленнее Qdrant на чистом vector search |

### 8. Сравнительная таблица и дерево решений

#### Полная сравнительная таблица

| Характеристика | ChromaDB | pgvector | Pinecone | Qdrant | Weaviate |
|---------------|----------|----------|----------|--------|----------|
| **Deployment** | Embedded / Client-server | PostgreSQL extension | Managed cloud | Self-host / Managed | Self-host / Managed |
| **Filtering** | Базовый (metadata, $and/$or) | Полный SQL (WHERE, JOIN) | Rich metadata | Payload pre-filtering | GraphQL + filters |
| **Scaling** | Single-node | Single Postgres | Auto-scaling | Sharding + replicas | Sharding + replicas |
| **Hybrid search** | Нет | Нет (нужен tsvector отдельно) | Sparse-dense vectors | Sparse + dense | BM25 + vector (нативно) |
| **Max vectors** | ~100k (comfortable) | ~10M (с HNSW) | Billions | ~100M (single node) | ~100M (single node) |
| **Latency (p99)** | <5ms (embedded) | 10-50ms | 20-100ms (network) | 5-20ms | 10-30ms |
| **Index types** | HNSW | IVF, HNSW | Proprietary | HNSW + quantization | HNSW |
| **Multi-vector** | Нет | Нет (вручную) | Нет | Да (named vectors) | Нет |
| **Multi-tenancy** | Коллекции | Schemas/tables | Namespaces | Коллекции | Нативная |
| **ACID** | Нет | Да (PostgreSQL) | Нет | Нет | Нет |
| **Стоимость** | Бесплатно | Стоимость Postgres | $70+/мес (serverless) | Бесплатно (self-host) | Бесплатно (self-host) |
| **LangChain** | langchain-chroma | langchain-postgres | langchain-pinecone | langchain-qdrant | langchain-weaviate |
| **Лучше всего для** | Dev/прототип | Если уже есть PG | Zero-ops, auto-scale | Production + self-host | Модули + hybrid |

#### Дерево решений

```
Какой vector database выбрать?
│
├── Прототип / dev / учебный проект?
│   └── → ChromaDB (zero-config, встраиваемый)
│
├── Уже используете PostgreSQL в проекте?
│   └── → pgvector (одна БД для всего, SQL, ACID)
│
├── Не хотите управлять инфраструктурой?
│   └── → Pinecone (managed, auto-scaling, pay-per-query)
│
├── Нужен мощный filtering + self-hosted?
│   └── → Qdrant (pre-filtering, multi-vector, quantization)
│
└── Нужны встроенные AI-модули + GraphQL?
    └── → Weaviate (text2vec, hybrid search, multi-tenancy)
```

Для нашего AI Assessment System:
- **Dev/test** → ChromaDB (уже настроен из темы 5)
- **Production с PostgreSQL** → pgvector (если проект уже использует PG для данных)
- **Production без PG** → Qdrant (мощный, self-hosted, бесплатный)

### 9. Hybrid Search — keyword + semantic

#### Проблема

Semantic search (vector) находит документы по смыслу, но может пропустить точные термины. Keyword search (BM25) находит точные совпадения, но не понимает синонимы.

| Запрос | Semantic search | Keyword search |
|--------|-----------------|----------------|
| «влияние ИИ на образование» | Найдёт «роль машинного обучения в школах» | Не найдёт (нет общих слов) |
| «статья 42 ФЗ-273» | Может найти что-то про законы | Найдёт точное совпадение |
| «HNSW алгоритм» | Найдёт про ANN и графовые индексы | Найдёт точно «HNSW» |

Hybrid search решает эту проблему, комбинируя оба подхода.

#### Reciprocal Rank Fusion (RRF)

RRF — алгоритм объединения двух рейтингов. Для каждого документа вычисляется итоговый score:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

где `k` — константа (обычно 60), `rank_i(d)` — позиция документа в i-м рейтинге.

Пример:

| Документ | Rank (semantic) | Rank (keyword) | RRF score |
|----------|----------------|----------------|-----------|
| doc_A | 1 | 5 | 1/61 + 1/65 = 0.0318 |
| doc_B | 3 | 1 | 1/63 + 1/61 = 0.0323 |
| doc_C | 2 | 10 | 1/62 + 1/70 = 0.0304 |

doc_B побеждает — он хорош в обоих рейтингах.

#### Реализация в Qdrant: sparse + dense vectors

Qdrant поддерживает sparse vectors (для BM25-like keyword matching) наряду с dense vectors:

```python
from qdrant_client.models import SparseVectorParams, SparseIndexParams

client.create_collection(
    collection_name="hybrid_essays",
    vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    },
)
```

#### Реализация в pgvector: tsvector + vector

В PostgreSQL можно комбинировать полнотекстовый поиск (`tsvector`) с vector search в одном запросе:

```sql
SELECT id, content,
       ts_rank(to_tsvector('russian', content), plainto_tsquery('russian', 'ИИ образование')) AS text_rank,
       1 - (embedding <=> query_embedding) AS vector_similarity,
       0.5 * ts_rank(...) + 0.5 * (1 - (embedding <=> query_embedding)) AS hybrid_score
FROM essay_embeddings
WHERE to_tsvector('russian', content) @@ plainto_tsquery('russian', 'ИИ образование')
   OR embedding <=> query_embedding < 0.5
ORDER BY hybrid_score DESC
LIMIT 5;
```

#### Реализация в LangChain: EnsembleRetriever

Наиболее универсальный подход — `EnsembleRetriever`, который комбинирует несколько retriever'ов:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

bm25_retriever = BM25Retriever.from_documents(docs, k=5)

vectorstore = Chroma.from_documents(docs, embedding=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

results = await hybrid_retriever.ainvoke("HNSW алгоритм в vector database")
```

`weights` определяет вес каждого retriever'а в RRF. `[0.4, 0.6]` — 40% keyword, 60% semantic. Настраивается под конкретный домен.

#### Когда hybrid search нужен

| Домен | Нужен hybrid? | Почему |
|-------|--------------|--------|
| Technical docs | Да | Точные термины (API names, error codes) + семантика |
| Legal | Да | Номера статей, законов + смысл нормы |
| Medical | Да | Латинские термины, коды МКБ + описание симптомов |
| Общие тексты (эссе) | Частично | Зависит от наличия терминологии |
| Chatbot FAQ | Нет | Semantic достаточно для перефразированных вопросов |

В нашей системе оценок hybrid search полезен, когда рубрика содержит специфические термины (названия теорем, формул, авторов), которые семантический поиск может пропустить.

---

## Справочник API

### `Chroma` (langchain_chroma)

```python
from langchain_chroma import Chroma
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `collection_name` | `str` | `"langchain"` | Название коллекции |
| `embedding_function` | `Embeddings` | — | Функция эмбеддингов |
| `persist_directory` | `str \| None` | `None` | Путь для persistent storage |
| `client` | `ClientAPI \| None` | `None` | Внешний Chroma client |
| `collection_metadata` | `dict \| None` | `None` | Метаданные коллекции (distance metric и т.д.) |
| `relevance_score_fn` | `Callable \| None` | `None` | Кастомная функция relevance score |

Методы:

| Метод | Описание |
|-------|----------|
| `from_documents(documents, embedding, ...)` | Создаёт vectorstore из списка Document |
| `from_texts(texts, embedding, metadatas, ...)` | Создаёт из текстов и метаданных |
| `add_documents(documents, ids)` | Добавляет документы в существующую коллекцию |
| `similarity_search(query, k, filter)` | Поиск по похожести, возвращает `list[Document]` |
| `similarity_search_with_score(query, k, filter)` | Поиск с score, `list[tuple[Document, float]]` |
| `as_retriever(search_type, search_kwargs)` | Создаёт Retriever для использования в chain |
| `delete(ids)` | Удаляет документы по ID |

### `PGVector` (langchain_postgres)

```python
from langchain_postgres import PGVector
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `connection` | `str \| Engine \| AsyncEngine` | — | Connection string или SQLAlchemy Engine |
| `embedding_function` | `Embeddings` | — | Функция эмбеддингов |
| `collection_name` | `str` | `"langchain"` | Название коллекции |
| `pre_delete_collection` | `bool` | `False` | Удалить коллекцию перед созданием |
| `use_jsonb` | `bool` | `True` | Использовать JSONB для метаданных |

Методы:

| Метод | Описание |
|-------|----------|
| `from_documents(documents, embedding, connection, ...)` | Создаёт vectorstore из Document |
| `from_texts(texts, embedding, metadatas, connection, ...)` | Создаёт из текстов |
| `add_documents(documents, ids)` | Добавляет документы |
| `asimilarity_search(query, k, filter)` | Async поиск, `list[Document]` |
| `asimilarity_search_with_score(query, k, filter)` | Async поиск с score |
| `as_retriever(search_type, search_kwargs)` | Создаёт Retriever |
| `adelete(ids)` | Async удаление по ID |

### `QdrantVectorStore` (langchain_qdrant)

```python
from langchain_qdrant import QdrantVectorStore
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `client` | `QdrantClient` | — | Qdrant client instance |
| `collection_name` | `str` | — | Название коллекции |
| `embedding` | `Embeddings` | — | Функция эмбеддингов |
| `content_payload_key` | `str` | `"page_content"` | Ключ для текста в payload |
| `metadata_payload_key` | `str` | `"metadata"` | Ключ для метаданных в payload |
| `distance` | `Distance` | `COSINE` | Метрика расстояния |
| `vector_name` | `str \| None` | `None` | Имя вектора (для multi-vector) |

Методы:

| Метод | Описание |
|-------|----------|
| `from_documents(documents, embedding, url, collection_name, ...)` | Создаёт vectorstore из Document |
| `from_texts(texts, embedding, metadatas, url, ...)` | Создаёт из текстов |
| `add_documents(documents, ids)` | Добавляет документы |
| `similarity_search(query, k, filter)` | Поиск, `list[Document]` |
| `similarity_search_with_score(query, k, filter)` | Поиск с score |
| `as_retriever(search_type, search_kwargs)` | Создаёт Retriever |
| `delete(ids)` | Удаление по ID |

### `PineconeVectorStore` (langchain_pinecone)

```python
from langchain_pinecone import PineconeVectorStore
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `index` | `Index \| None` | `None` | Pinecone Index instance |
| `embedding` | `Embeddings` | — | Функция эмбеддингов |
| `text_key` | `str` | `"text"` | Ключ для текста в метаданных |
| `namespace` | `str \| None` | `None` | Namespace для изоляции данных |
| `index_name` | `str \| None` | `None` | Имя Pinecone index |

Методы:

| Метод | Описание |
|-------|----------|
| `from_documents(documents, embedding, index_name, namespace, ...)` | Создаёт vectorstore |
| `add_documents(documents, ids, namespace)` | Добавляет документы |
| `similarity_search(query, k, filter, namespace)` | Поиск |
| `similarity_search_with_score(query, k, filter)` | Поиск с score |
| `as_retriever(search_kwargs)` | Создаёт Retriever |
| `delete(ids, namespace)` | Удаление |

### `EnsembleRetriever` (langchain.retrievers)

```python
from langchain.retrievers import EnsembleRetriever
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `retrievers` | `list[BaseRetriever]` | — | Список retriever'ов для объединения |
| `weights` | `list[float]` | — | Веса каждого retriever'а (сумма = 1.0) |
| `c` | `int` | `60` | Константа RRF (Reciprocal Rank Fusion) |
| `id_key` | `str \| None` | `None` | Ключ для дедупликации документов |

Методы:

| Метод | Описание |
|-------|----------|
| `invoke(query)` | Синхронный поиск, `list[Document]` |
| `ainvoke(query)` | Async поиск, `list[Document]` |
| `get_relevant_documents(query)` | Deprecated, используйте `invoke` |

### `BM25Retriever` (langchain_community.retrievers)

```python
from langchain_community.retrievers import BM25Retriever
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `k` | `int` | `4` | Количество результатов |
| `preprocess_func` | `Callable \| None` | `None` | Функция предобработки текста |

Методы:

| Метод | Описание |
|-------|----------|
| `from_documents(documents, k, preprocess_func)` | Создаёт из Document |
| `from_texts(texts, metadatas, k)` | Создаёт из текстов |
| `invoke(query)` | Синхронный поиск |
| `ainvoke(query)` | Async поиск |

### `Document` (langchain_core.documents)

```python
from langchain_core.documents import Document
```

| Поле | Тип | Описание |
|------|-----|----------|
| `page_content` | `str` | Текстовое содержимое документа |
| `metadata` | `dict` | Произвольные метаданные (subject, year, source и т.д.) |
| `id` | `str \| None` | Опциональный уникальный идентификатор |
| `type` | `Literal["Document"]` | Тип объекта (всегда "Document") |

---

## Практика: роутер `/api/v1/vector`

### Шаг 1. Схемы — `app/schemas/vector.py`

Создаём Pydantic-модели для всех эндпоинтов.

```python
from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)
    store: str = Field(pattern="^(chroma|pgvector|qdrant)$")
    collection_name: str = Field(default="student_essays")


class IndexResponse(BaseModel):
    store: str
    collection_name: str
    indexed_count: int
    ids: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    store: str = Field(pattern="^(chroma|pgvector|qdrant)$")
    collection_name: str = Field(default="student_essays")
    k: int = Field(default=5, ge=1, le=50)
    filter: dict | None = None


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    store: str
    query: str
    results: list[SearchResult]
    total: int


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_name: str = Field(default="student_essays")
    k: int = Field(default=5, ge=1, le=50)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)


class HybridSearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    semantic_weight: float
    keyword_weight: float
    total: int


class BenchmarkRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_name: str = Field(default="student_essays")
    stores: list[str] = Field(default=["chroma", "pgvector", "qdrant"])
    k: int = Field(default=5, ge=1, le=50)


class StoreLatency(BaseModel):
    store: str
    latency_ms: float
    results_count: int
    top_score: float | None = None
    error: str | None = None


class BenchmarkResponse(BaseModel):
    query: str
    benchmarks: list[StoreLatency]
    fastest_store: str
```

### Шаг 2. Vector store implementations

Создаём унифицированный интерфейс для трёх vector stores. Все три модуля реализуют одинаковые функции: `init_store`, `add_documents`, `search`, `delete`.

#### `app/services/vector_stores/chroma_store.py`

```python
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

_stores: dict[str, Chroma] = {}


def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )


def get_store(collection_name: str) -> Chroma:
    if collection_name not in _stores:
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=_get_embeddings(),
            persist_directory="./chroma_data",
        )
    return _stores[collection_name]


def add_documents(
    collection_name: str,
    documents: list[Document],
) -> list[str]:
    store = get_store(collection_name)
    ids = [str(uuid.uuid4()) for _ in documents]
    store.add_documents(documents, ids=ids)
    return ids


def search(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    store = get_store(collection_name)
    return store.similarity_search_with_score(
        query,
        k=k,
        filter=filter,
    )


def delete(collection_name: str, ids: list[str]) -> None:
    store = get_store(collection_name)
    store.delete(ids=ids)
```

#### `app/services/vector_stores/pgvector_store.py`

```python
import uuid

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import get_settings

_stores: dict[str, PGVector] = {}


def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )


def _get_connection_string() -> str:
    settings = get_settings()
    return (
        f"postgresql+asyncpg://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
    )


def get_store(collection_name: str) -> PGVector:
    if collection_name not in _stores:
        _stores[collection_name] = PGVector(
            embeddings=_get_embeddings(),
            collection_name=collection_name,
            connection=_get_connection_string(),
            use_jsonb=True,
        )
    return _stores[collection_name]


def add_documents(
    collection_name: str,
    documents: list[Document],
) -> list[str]:
    store = get_store(collection_name)
    ids = [str(uuid.uuid4()) for _ in documents]
    store.add_documents(documents, ids=ids)
    return ids


async def search(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    store = get_store(collection_name)
    return await store.asimilarity_search_with_score(
        query,
        k=k,
        filter=filter,
    )


async def delete(collection_name: str, ids: list[str]) -> None:
    store = get_store(collection_name)
    await store.adelete(ids=ids)
```

#### `app/services/vector_stores/qdrant_store.py`

```python
import uuid

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import get_settings

_stores: dict[str, QdrantVectorStore] = {}
_client: QdrantClient | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


def get_store(collection_name: str) -> QdrantVectorStore:
    if collection_name not in _stores:
        _stores[collection_name] = QdrantVectorStore(
            client=_get_client(),
            collection_name=collection_name,
            embedding=_get_embeddings(),
        )
    return _stores[collection_name]


def add_documents(
    collection_name: str,
    documents: list[Document],
) -> list[str]:
    store = get_store(collection_name)
    ids = [str(uuid.uuid4()) for _ in documents]
    store.add_documents(documents, ids=ids)
    return ids


def search(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    store = get_store(collection_name)
    return store.similarity_search_with_score(
        query,
        k=k,
    )


def delete(collection_name: str, ids: list[str]) -> None:
    store = get_store(collection_name)
    store.delete(ids=ids)
```

Обратите внимание на единообразие интерфейса: каждый модуль предоставляет `get_store`, `add_documents`, `search`, `delete`. Это позволяет роутеру выбирать backend по строковому параметру `store`.

### Шаг 3. Router — `app/api/v1/vector.py`

Роутер с 4 эндпоинтами, использующий store implementations из шага 2.

```python
import time

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from app.schemas.vector import (
    BenchmarkRequest,
    BenchmarkResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StoreLatency,
)
from app.services.vector_stores import chroma_store, pgvector_store, qdrant_store

router = APIRouter(prefix="/vector", tags=["lesson-19-vector-databases"])

STORE_MAP = {
    "chroma": chroma_store,
    "pgvector": pgvector_store,
    "qdrant": qdrant_store,
}


def _get_store_module(store_name: str):
    if store_name not in STORE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown store: {store_name}. Available: {list(STORE_MAP.keys())}",
        )
    return STORE_MAP[store_name]


@router.post("/index") 
async def index_documents(request: IndexRequest) -> IndexResponse:
    store_module = _get_store_module(request.store)

    documents = [
        Document(page_content=doc.content, metadata=doc.metadata)
        for doc in request.documents
    ]

    ids = store_module.add_documents(request.collection_name, documents)

    return IndexResponse(
        store=request.store,
        collection_name=request.collection_name,
        indexed_count=len(ids),
        ids=ids,
    )


@router.post("/search")
async def search_documents(request: SearchRequest) -> SearchResponse:
    store_module = _get_store_module(request.store)

    results = store_module.search(
        collection_name=request.collection_name,
        query=request.query,
        k=request.k,
        filter=request.filter,
    )

    if hasattr(results, "__await__"):
        results = await results

    search_results = [
        SearchResult(
            content=doc.page_content,
            metadata=doc.metadata,
            score=float(score),
        )
        for doc, score in results
    ]

    return SearchResponse(
        store=request.store,
        query=request.query,
        results=search_results,
        total=len(search_results),
    )


@router.post("/hybrid")
async def hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    chroma_vs = chroma_store.get_store(request.collection_name)

    existing_docs = chroma_vs.similarity_search("", k=1000)
    if not existing_docs:
        raise HTTPException(status_code=404, detail="Collection is empty")

    bm25_retriever = BM25Retriever.from_documents(existing_docs, k=request.k)
    vector_retriever = chroma_vs.as_retriever(search_kwargs={"k": request.k})

    keyword_weight = round(1.0 - request.semantic_weight, 2)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[keyword_weight, request.semantic_weight],
    )

    results = await ensemble.ainvoke(request.query)

    search_results = [
        SearchResult(
            content=doc.page_content,
            metadata=doc.metadata,
            score=1.0 / (i + 1),
        )
        for i, doc in enumerate(results[: request.k])
    ]

    return HybridSearchResponse(
        query=request.query,
        results=search_results,
        semantic_weight=request.semantic_weight,
        keyword_weight=keyword_weight,
        total=len(search_results),
    )


@router.post("/benchmark")
async def benchmark_stores(request: BenchmarkRequest) -> BenchmarkResponse:
    benchmarks: list[StoreLatency] = []
    min_latency = float("inf")
    fastest = ""

    for store_name in request.stores:
        try:
            store_module = _get_store_module(store_name)

            start = time.perf_counter()
            results = store_module.search(
                collection_name=request.collection_name,
                query=request.query,
                k=request.k,
            )
            if hasattr(results, "__await__"):
                results = await results
            elapsed_ms = (time.perf_counter() - start) * 1000

            top_score = float(results[0][1]) if results else None

            benchmarks.append(
                StoreLatency(
                    store=store_name,
                    latency_ms=round(elapsed_ms, 2),
                    results_count=len(results),
                    top_score=top_score,
                )
            )

            if elapsed_ms < min_latency:
                min_latency = elapsed_ms
                fastest = store_name

        except Exception as e:
            benchmarks.append(
                StoreLatency(
                    store=store_name,
                    latency_ms=-1,
                    results_count=0,
                    error=str(e),
                )
            )

    if not fastest:
        fastest = "none"

    return BenchmarkResponse(
        query=request.query,
        benchmarks=benchmarks,
        fastest_store=fastest,
    )
```

### Шаг 4. Docker setup для pgvector и Qdrant

Для запуска pgvector и Qdrant используем Docker Compose.

`docker-compose.vector.yml`:

```yaml
services:
  pgvector:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: bootcamp
      POSTGRES_PASSWORD: bootcamp
      POSTGRES_DB: bootcamp
    ports:
      - "5433:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bootcamp"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

volumes:
  pgvector_data:
  qdrant_data:
```

Запуск:

```bash
docker compose -f docker-compose.vector.yml up -d
```

Проверка pgvector:

```bash
docker exec -it $(docker ps -qf "ancestor=pgvector/pgvector:pg16") \
  psql -U bootcamp -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

Проверка Qdrant:

```bash
curl http://localhost:6333/healthz
```

Добавьте в `.env`:

```
PG_USER=bootcamp
PG_PASSWORD=bootcamp
PG_HOST=localhost
PG_PORT=5433
PG_DATABASE=bootcamp
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

И обновите `app/config.py`:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    pg_user: str = "bootcamp"
    pg_password: str = "bootcamp"
    pg_host: str = "localhost"
    pg_port: int = 5433
    pg_database: str = "bootcamp"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
```

### Шаг 5. Регистрация роутера и тестирование

Добавьте роутер в `app/api/router.py`:

```python
from app.api.v1 import assessment, rubrics, prompts, vector

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(vector.router)
```

#### Тестирование: индексация

```bash
curl -X POST http://localhost:8000/api/v1/vector/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Эссе про влияние искусственного интеллекта на современное образование. ИИ трансформирует методы преподавания.", "metadata": {"subject": "ai", "year": 2024}},
      {"content": "Математический анализ: пределы и производные. Предел функции в точке определяется через эпсилон-дельта определение.", "metadata": {"subject": "math", "year": 2024}},
      {"content": "Влияние машинного обучения на медицинскую диагностику. Нейросети анализируют снимки МРТ.", "metadata": {"subject": "ai", "year": 2025}},
      {"content": "Квантовая механика: принцип неопределённости Гейзенберга и волновая функция.", "metadata": {"subject": "physics", "year": 2024}},
      {"content": "Применение нейронных сетей в обработке естественного языка. Трансформеры произвели революцию в NLP.", "metadata": {"subject": "ai", "year": 2025}}
    ],
    "store": "chroma",
    "collection_name": "test_essays"
  }'
```

Ожидаемый ответ:

```json
{
  "store": "chroma",
  "collection_name": "test_essays",
  "indexed_count": 5,
  "ids": ["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"]
}
```

#### Тестирование: поиск

```bash
curl -X POST http://localhost:8000/api/v1/vector/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "как ИИ влияет на образование",
    "store": "chroma",
    "collection_name": "test_essays",
    "k": 3
  }'
```

Ожидаемый ответ:

```json
{
  "store": "chroma",
  "query": "как ИИ влияет на образование",
  "results": [
    {
      "content": "Эссе про влияние искусственного интеллекта на современное образование...",
      "metadata": {"subject": "ai", "year": 2024},
      "score": 0.92
    },
    {
      "content": "Влияние машинного обучения на медицинскую диагностику...",
      "metadata": {"subject": "ai", "year": 2025},
      "score": 0.78
    },
    {
      "content": "Применение нейронных сетей в обработке естественного языка...",
      "metadata": {"subject": "ai", "year": 2025},
      "score": 0.71
    }
  ],
  "total": 3
}
```

#### Тестирование: hybrid search

```bash
curl -X POST http://localhost:8000/api/v1/vector/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "HNSW алгоритм в vector database",
    "collection_name": "test_essays",
    "k": 3,
    "semantic_weight": 0.6
  }'
```

#### Тестирование: benchmark

```bash
curl -X POST http://localhost:8000/api/v1/vector/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "query": "влияние ИИ на образование",
    "collection_name": "test_essays",
    "stores": ["chroma"],
    "k": 3
  }'
```

Ожидаемый ответ:

```json
{
  "query": "влияние ИИ на образование",
  "benchmarks": [
    {
      "store": "chroma",
      "latency_ms": 12.45,
      "results_count": 3,
      "top_score": 0.92,
      "error": null
    }
  ],
  "fastest_store": "chroma"
}
```

Для полного бенчмарка с pgvector и Qdrant — запустите Docker Compose из шага 4, проиндексируйте данные в оба store, и используйте `"stores": ["chroma", "pgvector", "qdrant"]`.

### Связь с теорией

| Эндпоинт | Теоретический раздел | Что демонстрирует |
|----------|---------------------|-------------------|
| `POST /vector/index` | §3–6: ChromaDB, pgvector, Qdrant | Унифицированный интерфейс для разных vector DB |
| `POST /vector/search` | §2: similarity metrics, ANN | Vector search с выбором backend |
| `POST /vector/hybrid` | §9: Hybrid search, EnsembleRetriever | Комбинация BM25 + semantic |
| `POST /vector/benchmark` | §8: сравнительная таблица | Количественное сравнение latency |

Каждый эндпоинт — практическая демонстрация теоретических концепций:

- **Index** показывает, как один и тот же набор документов сохраняется в разных backend'ах с единым API.
- **Search** демонстрирует работу ANN-алгоритмов (HNSW в Chroma и Qdrant, HNSW/IVF в pgvector) через единый интерфейс.
- **Hybrid** реализует EnsembleRetriever из §9 — комбинация BM25Retriever (keyword) и VectorStoreRetriever (semantic) с настраиваемыми весами.
- **Benchmark** позволяет количественно проверить сравнительную таблицу из §8 на реальных данных вашего проекта.

---

## Чеклист самопроверки

- [ ] Объясни разницу между cosine similarity, euclidean distance и dot product. Когда какую метрику выбрать?
- [ ] Как работает HNSW? Почему он быстрее brute-force? Какие параметры влияют на recall и latency?
- [ ] В чём разница между IVF и HNSW? Когда предпочтительнее IVF?
- [ ] Почему pre-filtering (Qdrant) лучше post-filtering для vector search? Приведи пример, когда post-filtering вернёт меньше k результатов.
- [ ] Какой vector database ты выберешь для проекта, где уже используется PostgreSQL для основных данных? Почему?
- [ ] В чём trade-off между ChromaDB (embedded) и Qdrant (self-hosted) для production-системы с 500k документов?
- [ ] Как работает Reciprocal Rank Fusion (RRF)? Почему он лучше простого усреднения scores?
- [ ] Когда hybrid search (keyword + semantic) даёт значительное улучшение по сравнению с чистым semantic search? Приведи 3 примера доменов.
- [ ] Что такое quantization в контексте vector databases? Какой тип quantization даёт лучший баланс сжатия и recall?
- [ ] Пройди весь pipeline: проиндексируй 5 документов в Chroma, выполни search, затем hybrid search. Сравни результаты.

---

## Частые ошибки

### 1. Индексация без metadata

```python
store.add_documents([Document(page_content="Текст эссе")])
```

```python
store.add_documents([
    Document(
        page_content="Текст эссе",
        metadata={"subject": "math", "year": 2024, "student_id": "s123"},
    )
])
```

Без metadata невозможно фильтровать результаты. В production-системе оценок нужно всегда сохранять предмет, год, идентификатор студента — это позволяет ограничивать поиск релевантным контекстом.

### 2. Использование Chroma в production с большими данными

```python
vectorstore = Chroma(
    collection_name="all_essays",
    persist_directory="./chroma_data",
)
```

```python
vectorstore = QdrantVectorStore(
    client=QdrantClient(host="qdrant-server", port=6333),
    collection_name="all_essays",
    embedding=embeddings,
)
```

ChromaDB в embedded mode начинает деградировать при >100k документов. Для production с большими объёмами используйте Qdrant, pgvector или Pinecone. ChromaDB отлично подходит для dev и тестирования.

### 3. Забытый индекс в pgvector

```sql
SELECT * FROM essay_embeddings
ORDER BY embedding <=> '[...]'
LIMIT 5;
```

```sql
CREATE INDEX ON essay_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

SELECT * FROM essay_embeddings
ORDER BY embedding <=> '[...]'
LIMIT 5;
```

Без индекса pgvector выполняет sequential scan — O(N). При 100k записей запрос занимает секунды вместо миллисекунд. HNSW индекс ускоряет search на порядки.

### 4. Неправильные веса в EnsembleRetriever

```python
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.9, 0.1],
)
```

```python
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],
)
```

Вес 0.9 для BM25 почти полностью игнорирует semantic search. Для большинства задач semantic search важнее keyword. Начинайте с `[0.3, 0.7]` или `[0.4, 0.6]` и настраивайте по результатам A/B-тестирования.

### 5. Смешение метрик расстояния

```python
client.create_collection(
    collection_name="essays",
    vectors_config=VectorParams(size=1536, distance=Distance.EUCLID),
)

results = store.similarity_search_with_score(query, k=5)
for doc, score in results:
    if score > 0.8:  # <-- трактуем как cosine similarity
        print(doc.page_content)
```

```python
client.create_collection(
    collection_name="essays",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
```

Cosine similarity — в диапазоне [0, 1] (для нормализованных векторов). Euclidean distance — в диапазоне [0, ∞). Пороги и интерпретация score зависят от метрики. Если коллекция создана с euclidean, а код ожидает cosine score — пороги будут неверными.

### 6. Отсутствие обработки ошибок при недоступном vector store

```python
results = store_module.search(collection_name=name, query=query, k=k)
```

```python
try:
    results = store_module.search(collection_name=name, query=query, k=k)
except Exception as e:
    raise HTTPException(
        status_code=503,
        detail=f"Vector store '{store_name}' unavailable: {str(e)}",
    )
```

В production pgvector или Qdrant могут быть временно недоступны (network, restart, OOM). Всегда обрабатывайте ошибки подключения и возвращайте осмысленный HTTP-ответ, а не 500 с трейсбеком.

---

## Что читать дальше

- [LangChain Vector Stores](https://python.langchain.com/docs/concepts/vectorstores/) — концепция vector store в LangChain
- [Chroma Documentation](https://docs.trychroma.com/) — официальная документация ChromaDB
- [pgvector GitHub](https://github.com/pgvector/pgvector) — README с примерами SQL и индексов
- [Qdrant Documentation](https://qdrant.tech/documentation/) — полная документация Qdrant
- [Pinecone Documentation](https://docs.pinecone.io/) — гайды по Pinecone
- [Weaviate Documentation](https://weaviate.io/developers/weaviate) — полная документация Weaviate
- [ANN Benchmarks](https://ann-benchmarks.com/) — сравнение производительности ANN-алгоритмов
- [HNSW Paper](https://arxiv.org/abs/1603.09320) — оригинальная статья про HNSW
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — оригинальная статья про RRF
- [LangChain EnsembleRetriever](https://python.langchain.com/docs/how_to/ensemble_retriever/) — hybrid search в LangChain

**Следующая тема:** [Тема 20: Deployment](topic_20_deployment.md)

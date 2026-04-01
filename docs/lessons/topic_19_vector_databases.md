# Тема 19: Vector Databases — сравнение и выбор

> **Пререквизиты:** [Тема 5: RAG](topic_05_rag.md)
> **Зависимости:** `langchain-chroma`, `langchain-postgres`, `langchain-qdrant`, `pgvector`, `qdrant-client`, `psycopg2-binary`, `rank-bm25`

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
3. Попрактикуемся с 3 из них (Chroma, pgvector, Qdrant)
4. Сравним latency разных vector stores
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

## Практика

### Подготовка окружения

Для примеров с pgvector и Qdrant нужны Docker-контейнеры:

```bash
docker run -d --name pgvector-demo \
  -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=demo \
  -p 5433:5432 pgvector/pgvector:pg16

docker run -d --name qdrant-demo \
  -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Все примеры ниже — самодостаточные скрипты. Запускайте как ячейки Jupyter-ноутбука или как `.py`-файлы.

### Пример 1. ChromaDB — базовое использование

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(
        page_content="Эссе про влияние искусственного интеллекта на современное образование. ИИ трансформирует методы преподавания.",
        metadata={"subject": "ai", "year": 2024},
    ),
    Document(
        page_content="Математический анализ: пределы и производные. Предел функции в точке определяется через эпсилон-дельта определение.",
        metadata={"subject": "math", "year": 2024},
    ),
    Document(
        page_content="Влияние машинного обучения на медицинскую диагностику. Нейросети анализируют снимки МРТ.",
        metadata={"subject": "ai", "year": 2025},
    ),
    Document(
        page_content="Квантовая механика: принцип неопределённости Гейзенберга и волновая функция.",
        metadata={"subject": "physics", "year": 2024},
    ),
    Document(
        page_content="Применение нейронных сетей в обработке естественного языка. Трансформеры произвели революцию в NLP.",
        metadata={"subject": "ai", "year": 2025},
    ),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="essay_demo",
    persist_directory="./chroma_demo",
)

results = vectorstore.similarity_search_with_score("как ИИ влияет на образование", k=3)

for doc, score in results:
    print(f"[score={score:.4f}] {doc.page_content[:80]}...")
    print(f"  metadata: {doc.metadata}")
```

Ожидаемый вывод — три наиболее релевантных документа, отсортированных по cosine distance (меньше = ближе). Документы про ИИ получают наименьшее расстояние.

### Пример 2. ChromaDB — фильтрация по метаданным

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_filter_demo")

collection = client.get_or_create_collection(
    name="filtering_demo",
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=[
        "Эссе про влияние ИИ на образование",
        "Математический анализ: пределы",
        "Нейросети в медицинской диагностике",
        "Квантовая механика: принцип Гейзенберга",
        "Трансформеры в NLP",
    ],
    metadatas=[
        {"subject": "ai", "year": 2024},
        {"subject": "math", "year": 2024},
        {"subject": "ai", "year": 2025},
        {"subject": "physics", "year": 2024},
        {"subject": "ai", "year": 2025},
    ],
    ids=["e1", "e2", "e3", "e4", "e5"],
)

results = collection.query(
    query_texts=["искусственный интеллект"],
    n_results=5,
    where={"subject": "ai"},
)

print("Фильтр subject=ai:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc[:60]}... | {meta}")

results = collection.query(
    query_texts=["наука"],
    n_results=5,
    where={
        "$and": [
            {"year": {"$gte": 2025}},
            {"subject": {"$eq": "ai"}},
        ]
    },
)

print("\nФильтр year>=2025 AND subject=ai:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc[:60]}... | {meta}")
```

Фильтрация в ChromaDB — post-filtering: сначала находятся ближайшие соседи, затем отфильтровываются по метаданным. При агрессивной фильтрации может вернуться меньше `n_results` документов.

### Пример 3. pgvector — SQL-запросы с векторным поиском

```python
import psycopg2
from langchain_openai import OpenAIEmbeddings

conn = psycopg2.connect(
    host="localhost", port=5433, user="demo", password="demo", dbname="demo"
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    DROP TABLE IF EXISTS essay_embeddings;
    CREATE TABLE essay_embeddings (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        subject VARCHAR(100),
        year INTEGER,
        embedding vector(1536)
    )
""")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    ("Эссе про влияние ИИ на образование", "ai", 2024),
    ("Математический анализ: пределы и производные", "math", 2024),
    ("Нейросети в медицинской диагностике", "ai", 2025),
    ("Квантовая механика: принцип Гейзенберга", "physics", 2024),
    ("Трансформеры произвели революцию в NLP", "ai", 2025),
]

for text, subject, year in texts:
    vector = embeddings.embed_query(text)
    cur.execute(
        "INSERT INTO essay_embeddings (content, subject, year, embedding) VALUES (%s, %s, %s, %s)",
        (text, subject, year, str(vector)),
    )

cur.execute("""
    CREATE INDEX IF NOT EXISTS essay_hnsw_idx ON essay_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
""")

query_vector = embeddings.embed_query("как ИИ влияет на образование")

cur.execute("""
    SELECT content, subject, year,
           1 - (embedding <=> %s::vector) AS similarity
    FROM essay_embeddings
    WHERE subject = 'ai'
    ORDER BY embedding <=> %s::vector
    LIMIT 3
""", (str(query_vector), str(query_vector)))

print("pgvector — поиск с фильтром subject='ai':")
for content, subject, year, similarity in cur.fetchall():
    print(f"  [{similarity:.4f}] {content} | subject={subject}, year={year}")

cur.close()
conn.close()
```

Ключевое преимущество pgvector — полный SQL. `WHERE subject = 'ai'` выполняется совместно с vector search. Можно добавлять JOIN'ы, подзапросы, агрегации — всё, что умеет PostgreSQL.

### Пример 4. Qdrant — использование и pre-filtering

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="essay_demo",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

texts = [
    ("Эссе про влияние ИИ на образование", {"subject": "ai", "year": 2024}),
    ("Математический анализ: пределы", {"subject": "math", "year": 2024}),
    ("Нейросети в медицинской диагностике", {"subject": "ai", "year": 2025}),
    ("Квантовая механика: принцип Гейзенберга", {"subject": "physics", "year": 2024}),
    ("Трансформеры в NLP", {"subject": "ai", "year": 2025}),
]

points = []
for i, (text, meta) in enumerate(texts):
    vector = embeddings.embed_query(text)
    points.append(PointStruct(id=i + 1, vector=vector, payload={"content": text, **meta}))

client.upsert(collection_name="essay_demo", points=points)

query_vector = embeddings.embed_query("влияние ИИ на образование")

results = client.query_points(
    collection_name="essay_demo",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="subject", match=MatchValue(value="ai")),
            FieldCondition(key="year", range=Range(gte=2025)),
        ]
    ),
    limit=3,
)

print("Qdrant — pre-filtering (subject=ai, year>=2025):")
for point in results.points:
    print(f"  [score={point.score:.4f}] {point.payload['content']}")
    print(f"    subject={point.payload['subject']}, year={point.payload['year']}")
```

Qdrant выполняет pre-filtering — фильтрация происходит **до** vector search. Это гарантирует ровно `limit` результатов (если столько документов проходит фильтр), в отличие от post-filtering, который может вернуть меньше.

### Пример 5. Hybrid search — keyword + semantic

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

docs = [
    Document(
        page_content="HNSW (Hierarchical Navigable Small World) — графовый алгоритм ANN-поиска.",
        metadata={"topic": "algorithms"},
    ),
    Document(
        page_content="Влияние искусственного интеллекта на школьное образование.",
        metadata={"topic": "ai"},
    ),
    Document(
        page_content="Алгоритм HNSW строит многоуровневый граф для навигации по точкам.",
        metadata={"topic": "algorithms"},
    ),
    Document(
        page_content="Нейронные сети используют backpropagation для обучения.",
        metadata={"topic": "ai"},
    ),
    Document(
        page_content="IVF-индекс разбивает пространство на кластеры через k-means.",
        metadata={"topic": "algorithms"},
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="hybrid_demo",
)

bm25_retriever = BM25Retriever.from_documents(docs, k=3)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

query = "HNSW алгоритм в vector database"

print("=== Только semantic search ===")
for i, doc in enumerate(vectorstore.similarity_search(query, k=3), 1):
    print(f"  {i}. {doc.page_content[:80]}")

print("\n=== Только keyword search (BM25) ===")
for i, doc in enumerate(bm25_retriever.invoke(query), 1):
    print(f"  {i}. {doc.page_content[:80]}")

print("\n=== Hybrid search (BM25 0.4 + semantic 0.6) ===")
for i, doc in enumerate(ensemble.invoke(query), 1):
    print(f"  {i}. {doc.page_content[:80]}")
```

Сравните результаты трёх подходов. Для запроса «HNSW алгоритм» BM25 точно найдёт документы с этим термином, semantic search — документы близкие по смыслу. Hybrid search объединяет оба рейтинга через Reciprocal Rank Fusion.

### Пример 6. Бенчмарк latency — сравнение vector stores

Для этого примера нужны запущенные Docker-контейнеры pgvector и Qdrant (см. «Подготовка окружения»).

```python
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(
        page_content=f"Документ #{i}: текст для бенчмарка vector search по теме {topic}.",
        metadata={"subject": topic, "index": i},
    )
    for i, topic in enumerate(["ai", "math", "physics", "ai", "math"] * 20)
]

chroma_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="bench_chroma",
)

qdrant_client = QdrantClient(host="localhost", port=6333)
qdrant_client.recreate_collection(
    collection_name="bench_qdrant",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
qdrant_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="bench_qdrant",
    url="http://localhost:6333",
)

query = "влияние ИИ на образование"
n_runs = 5

stores = {"ChromaDB": chroma_store, "Qdrant": qdrant_store}

for name, store in stores.items():
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        store.similarity_search_with_score(query, k=5)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    avg = sum(latencies) / len(latencies)
    mn, mx = min(latencies), max(latencies)
    print(f"{name}: avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  (n={n_runs})")
```

На 100 документах разница между ChromaDB и Qdrant минимальна. Различия проявляются на сотнях тысяч документов — Qdrant с HNSW + quantization значительно быстрее.

### Связь с теорией

| Пример | Теоретический раздел | Что демонстрирует |
|--------|---------------------|-------------------|
| Пример 1 (ChromaDB) | §3: ChromaDB — embedded store | Базовый workflow: индексация → similarity search |
| Пример 2 (Фильтрация) | §3: Filtering | Metadata-фильтры `$and`, `$gte`, `$eq` |
| Пример 3 (pgvector) | §4: pgvector — PostgreSQL extension | SQL + vector search, HNSW-индекс, WHERE-фильтры |
| Пример 4 (Qdrant) | §6: Qdrant — pre-filtering | Payload-based filtering, must/must_not условия |
| Пример 5 (Hybrid) | §9: Hybrid search, EnsembleRetriever | BM25 + semantic через RRF с настраиваемыми весами |
| Пример 6 (Бенчмарк) | §8: Сравнительная таблица | Количественное сравнение latency на реальных данных |

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
results = client.query_points(collection_name=name, query=vector, limit=k)
```

```python
try:
    results = client.query_points(collection_name=name, query=vector, limit=k)
except Exception as e:
    print(f"Vector store недоступен: {e}")
    results = []
```

В production pgvector или Qdrant могут быть временно недоступны (network, restart, OOM). Всегда обрабатывайте ошибки подключения — иначе один упавший vector store сломает весь pipeline.

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

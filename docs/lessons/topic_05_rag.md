# Тема 5: RAG (Retrieval-Augmented Generation)

> **Пререквизиты:** темы 1–4 (prompt engineering, structured output, LCEL chains)
> **Что добавляем в проект:** `app/api/v1/rag.py`, `app/services/vector_store.py`, `app/schemas/rag.py`
> **Зависимости:** `langchain-chroma`, `langchain-openai`, `chromadb`, `sentence-transformers`, `pypdf` (группа `rag`)

---

## Теория

### 1. Зачем RAG

LLM обучена на публичных данных до определённой даты — cutoff date. После этой даты модель ничего не знает о мире. Кроме того, она не имеет доступа к:

- Внутренним документам организации (рубрики, учебные планы, прошлые оценки студентов)
- Приватным данным, которые никогда не публиковались в интернете
- Информации, появившейся после cutoff-даты обучения
- Данным, которые часто обновляются (расписание, актуальные политики, курсы)

Когда LLM не хватает знаний, она **галлюцинирует** — уверенно генерирует правдоподобный, но фактически неверный ответ. В production-системе, особенно в образовании, это критическая проблема: модель может придумать несуществующий критерий рубрики или присвоить баллы по шкале, которой нет.

**Три подхода к расширению знаний LLM:**

| Подход | Суть | Стоимость | Актуальность данных | Прозрачность |
|--------|------|-----------|---------------------|--------------|
| Prompt engineering | Вставить данные прямо в промпт | Бесплатно | Ручное обновление | Полная |
| Fine-tuning | Дообучить модель на своих данных | $100–$10 000+ | Требует переобучения | Низкая — данные «зашиты» в веса |
| **RAG** | Найти релевантные документы, подставить в промпт | $0.01–$0.10 за запрос | Мгновенное обновление | Полная — источники видны |

Prompt engineering работает, пока данных мало — 1–2 страницы текста. Но если у тебя 500 прошлых эссе, 20 рубрик и 50 учебных планов, всё это не поместится в context window даже самой большой модели. Fine-tuning «вплавляет» знания в веса модели, делая их непрозрачными и дорогими в обновлении: каждое изменение рубрики требует нового цикла обучения.

**RAG = Retrieval-Augmented Generation** — архитектурный паттерн, при котором система:

1. **Retrieval** — по запросу пользователя находит релевантные документы из базы знаний через vector search
2. **Augmented** — вставляет найденные документы в промпт как дополнительный контекст
3. **Generation** — LLM генерирует ответ, опираясь на подставленный контекст, а не только на свои «врождённые» знания

Архитектура RAG-пайплайна:

```
query → embed(query) → vector search → top-k documents → format_context → prompt + context → LLM → response
```

В нашем проекте RAG решает конкретную задачу: найти прошлые оценки похожих эссе и использовать их как калибровку. Если два эссе одинакового качества, они должны получить близкие оценки — RAG обеспечивает эту консистентность, предоставляя модели примеры прошлых решений.

### 2. Embeddings — как текст становится вектором

Embedding — числовое представление текста в многомерном пространстве. Каждый текст превращается в вектор фиксированной длины, причём семантически близкие тексты получают близкие векторы:

```
embed("столица Франции")    → [0.12, -0.34, 0.78, ...]  (1536 чисел)
embed("Париж — город")      → [0.11, -0.32, 0.76, ...]  ← косинусное расстояние мало
embed("программирование")    → [-0.45, 0.22, -0.11, ...] ← косинусное расстояние велико
```

Embedding-модель не сравнивает слова побуквенно — она «понимает» смысл. Фраза «влияние ИИ на образование» будет близка к «как машинное обучение меняет школы», хотя общих слов почти нет.

**Cosine similarity** — стандартная мера близости векторов:

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

Результат от −1 до 1: значение 1 означает идентичные направления (тексты об одном и том же), 0 — не связаны, −1 — противоположны по смыслу. На практике значения для текстовых embedding'ов редко опускаются ниже 0.

Почему cosine similarity, а не евклидово расстояние? Cosine игнорирует «длину» вектора и сравнивает только направление. Это важно, потому что embedding длинного текста может иметь бо́льшую норму, чем embedding короткого, но семантически они могут быть об одном и том же.

**Embedding-модели:**

| Модель | Размерность | Скорость | Качество | Стоимость |
|--------|------------|----------|----------|-----------|
| `text-embedding-3-small` (OpenAI) | 1536 | Быстро (API) | Высокое | $0.02 / 1M tokens |
| `text-embedding-3-large` (OpenAI) | 3072 | Быстро (API) | Очень высокое | $0.13 / 1M tokens |
| `all-MiniLM-L6-v2` (HuggingFace) | 384 | Быстро (локально) | Хорошее | Бесплатно |
| `BGE-base-en-v1.5` (HuggingFace) | 768 | Средне (локально) | Высокое | Бесплатно |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Средне (локально) | Хорошее (мультиязычное) | Бесплатно |

Важные ограничения embedding-моделей:
- **Token limit** — каждая модель имеет максимальную длину входного текста (8191 токен для OpenAI, 256–512 для многих HuggingFace-моделей). Текст сверх лимита обрезается, что приводит к потере информации — поэтому документы нарезают на чанки.
- **Batch processing** — при индексации тысяч документов embedding'и вычисляются батчами. OpenAI API принимает до 2048 текстов за один запрос. Локальные модели ограничены GPU-памятью.
- **Консистентность** — один и тот же текст всегда даёт один и тот же вектор (embedding-модели детерминированы). Но разные модели дают разные пространства — нельзя сравнивать вектор от OpenAI с вектором от MiniLM.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("impact of AI on education")
print(len(vector))
```

### 3. Vector Store — хранилище векторов

Vector store — специализированная база данных, оптимизированная для **поиска по сходству** (similarity search). Обычные базы данных ищут по точному совпадению (`WHERE name = 'John'`), а vector store находит k записей с наиболее близкими векторами к запросу.

**Как работает vector search:**

1. Запрос пользователя превращается в вектор через ту же embedding-модель, которая использовалась при индексации
2. Vector store сравнивает этот вектор со всеми хранимыми векторами
3. Возвращает top-k наиболее близких документов

Для малых коллекций (до ~100 000 документов) применяется **brute-force search** — сравнение с каждым вектором. Для больших — **приближённый поиск** (ANN — Approximate Nearest Neighbors) через индексы HNSW, IVF или другие. ANN жертвует точностью ради скорости: вместо гарантированного нахождения ближайшего соседа он находит «достаточно близкого» за доли секунды даже на миллионах записей.

**ChromaDB** — локальный vector store, идеальный для разработки и обучения:

- Работает in-memory (быстро, данные теряются при перезапуске) или с файловым хранилищем (persistent)
- Не требует отдельного сервера — всё в одном процессе
- Поддерживает metadata filtering — фильтрация результатов по метаданным до vector search
- По умолчанию использует L2 (евклидово) расстояние, но поддерживает cosine и inner product
- Организует данные в **коллекции** — аналог таблиц в SQL

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="assessments",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
)

vectorstore.add_texts(
    texts=["Essay about AI impact on education..."],
    metadatas=[{"source": "essay_1", "quality": "high", "score": 85}],
)

results = vectorstore.similarity_search("thesis about AI", k=3)
```

**Metadata filtering** позволяет сузить поиск до подмножества документов:

```python
results = vectorstore.similarity_search(
    "strong thesis",
    k=3,
    filter={"quality": "high"},
)
```

Это полезно, когда нужно искать только среди определённой категории документов — например, только среди эссе с высокой оценкой.

Для production-систем ChromaDB можно заменить на Pinecone, Weaviate, Qdrant или pgvector — интерфейс LangChain останется тем же, меняется только инициализация vector store.

### 4. Document Loaders — загрузка документов

LangChain предоставляет унифицированный интерфейс для загрузки документов из десятков источников. Каждый loader возвращает список объектов `Document` — стандартную обёртку над текстом с метаданными.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/syllabus.pdf")
docs = loader.load()
```

Каждый `Document` содержит два поля:
- `page_content: str` — текстовое содержимое
- `metadata: dict` — словарь метаданных (source, page, date и т.д.)

Метаданные критически важны для RAG: они позволяют отследить, из какого документа пришёл чанк (attribution), фильтровать результаты поиска и показывать пользователю источники.

**Основные loader'ы:**

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

pdf_docs = PyPDFLoader("data/report.pdf").load()

txt_docs = TextLoader("data/essay.txt", encoding="utf-8").load()

all_docs = DirectoryLoader(
    "data/essays/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
).load()
```

`DirectoryLoader` особенно полезен для batch-загрузки: он рекурсивно обходит директорию, применяя указанный loader к каждому файлу. Параметр `glob` позволяет фильтровать файлы по паттерну.

При загрузке всегда обогащай метаданные:

```python
from langchain_core.documents import Document

docs = []
for path in Path("data/essays").glob("*.txt"):
    text = path.read_text()
    docs.append(Document(
        page_content=text,
        metadata={
            "source": str(path),
            "type": "student_essay",
            "filename": path.name,
        },
    ))
```

### 5. Text Splitters — нарезка на чанки

Документ целиком обычно слишком велик для embedding и для контекстного окна промпта. Кроме того, embedding длинного текста «усредняет» его смысл — вектор 10-страничного документа будет слабо отличаться от вектора любого другого длинного текста. Нарезка на чанки (chunking) решает обе проблемы.

**RecursiveCharacterTextSplitter** — самый универсальный splitter в LangChain. Он пытается разделить текст по иерархии разделителей, сохраняя семантическую целостность:

1. Сначала пытается разделить по `\n\n` (границы абзацев)
2. Если чанк всё ещё слишком велик — по `\n` (переводы строк)
3. Затем по `. ` (границы предложений)
4. Затем по ` ` (границы слов)
5. В крайнем случае — посимвольно

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_documents(docs)
```

**Как выбрать chunk_size:**

| chunk_size | Плюсы | Минусы | Когда использовать |
|-----------|-------|--------|-------------------|
| 200 | Точный поиск, фокусированные фрагменты | Мало контекста, мысль может быть обрезана | FAQ, короткие ответы |
| 500 | Баланс точности и контекста | — | Большинство задач (рекомендуемый дефолт) |
| 1000 | Полный контекст абзацев | Менее точный поиск, больше «шума» | Длинные аналитические тексты |
| 2000+ | Целые секции документа | Embedding «усредняет» смысл | Суммаризация, не для поиска |

**Overlap** — перекрытие между соседними чанками. Если `chunk_size=500` и `chunk_overlap=50`, последние 50 символов чанка N будут первыми 50 символами чанка N+1:

```
Чанк 1: [====================XXXXX]
Чанк 2:                     [XXXXX====================]
                             ↑ overlap — 50 символов
```

Зачем overlap? Без него предложение на границе двух чанков будет разрезано пополам. Первая половина останется в одном чанке, вторая — в другом. Ни один из них не будет найден по запросу, связанному с полным предложением. Overlap гарантирует, что граничные предложения попадут хотя бы в один чанк целиком.

Хорошее правило: overlap = 10–20% от chunk_size.

**Другие splitter'ы** (менее распространённые):
- `CharacterTextSplitter` — разделяет только по одному символу (например, `\n\n`), без рекурсии
- `TokenTextSplitter` — считает размер в токенах, а не символах (точнее для LLM)
- `MarkdownTextSplitter` — учитывает структуру Markdown (заголовки, списки, блоки кода)
- `HTMLSectionSplitter` — разделяет HTML по тегам-секциям

### 6. Retriever и стратегии поиска

Retriever — абстракция LangChain, инкапсулирующая логику поиска документов. Любой vector store можно превратить в retriever одним вызовом:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
docs = retriever.invoke("thesis about AI impact")
```

Retriever реализует Runnable-интерфейс — его можно использовать в LCEL-цепочках с оператором `|`.

**Similarity search** — возвращает k документов с наименьшим расстоянием до запроса. Прост и эффективен, но имеет проблему: если топ-3 документа почти идентичны (например, три абзаца из одного эссе), ты получишь дублирующуюся информацию вместо разнообразного контекста.

**MMR (Maximal Marginal Relevance)** — балансирует **релевантность** и **разнообразие**. Алгоритм:
1. Сначала находит `fetch_k` кандидатов по similarity
2. Первый результат — самый релевантный (как обычный similarity search)
3. Каждый следующий результат — максимально релевантный запросу И максимально отличающийся от уже выбранных

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.7,
    },
)
```

Параметр `lambda_mult` контролирует баланс: 1.0 = только релевантность (эквивалент similarity search), 0.0 = только разнообразие. Значение 0.5–0.7 — хороший дефолт для большинства задач.

**Similarity score threshold** — третья стратегия, возвращающая только документы с relevance score выше порога:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 5},
)
```

Это полезно, когда лучше не вернуть ничего, чем вернуть нерелевантный документ. Если все документы в коллекции далеки от запроса, retriever вернёт пустой список вместо «лучших из плохих».

**Когда что использовать:**
- `similarity` — дефолт, подходит для большинства задач
- `mmr` — когда важно разнообразие контекста (например, разные аспекты темы)
- `similarity_score_threshold` — когда критично не подставлять нерелевантный контекст

### 7. RAG Chain — собираем пайплайн

RAG chain связывает все компоненты в единый пайплайн с помощью LCEL:

```
query → retriever → format_docs → prompt + context → LLM → response
```

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=retriever | format_docs)
    | prompt
    | llm
)
```

Разберём, что происходит при `rag_chain.invoke({"student_work": "...", "rubric": "..."})`:

1. `RunnablePassthrough.assign(context=...)` — пропускает весь входной dict без изменений, но добавляет ключ `context`. Значение `context` вычисляется через под-цепочку `retriever | format_docs`
2. `retriever` получает входной dict, извлекает из него запрос и возвращает список Document
3. `format_docs` принимает список Document и склеивает их `page_content` в одну строку
4. Результат — dict с ключами `student_work`, `rubric` и `context` — передаётся в `prompt`
5. `prompt` формирует сообщения для LLM, подставляя значения из dict
6. `llm` генерирует ответ, используя контекст из найденных документов

Для нашего проекта RAG chain находит прошлые оценки похожих эссе и вставляет их в промпт:

```python
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assessor.

## Rubric
{rubric}

## Previously assessed similar works (use as calibration):
{context}
"""),
    ("human", "Assess this student work:\n\n{student_work}"),
])
```

Модель видит реальные примеры оценок похожих работ и калибруется по ним — одинаковое качество получает одинаковые баллы.

**Отладка RAG chain:** если результаты неудовлетворительны, проверяй каждый этап отдельно:

```python
retrieved = retriever.invoke("query text")
for doc in retrieved:
    print(doc.page_content[:200], doc.metadata)
```

Частые проблемы:
- Retriever возвращает нерелевантные документы → проверь embedding-модель и chunk_size
- Контекст слишком длинный → уменьши k или chunk_size
- Модель игнорирует контекст → усиль инструкцию в промпте

---

## Справочник API

### `Chroma`

**Импорт:** `from langchain_chroma import Chroma`

Vector store на базе ChromaDB. Хранит документы с embedding-векторами и поддерживает similarity search с фильтрацией по метаданным.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `collection_name` | `str` | `"langchain"` | Имя коллекции в ChromaDB |
| `embedding_function` | `Embeddings \| None` | `None` | Embedding-модель для векторизации текста |
| `persist_directory` | `str \| None` | `None` | Путь для файлового хранилища; `None` = in-memory |
| `client` | `chromadb.ClientAPI \| None` | `None` | Готовый клиент ChromaDB (если нужна кастомная конфигурация) |
| `collection_metadata` | `dict \| None` | `None` | Метаданные коллекции (например, `{"hnsw:space": "cosine"}`) |
| `relevance_score_fn` | `Callable \| None` | `None` | Функция для преобразования расстояния в relevance score |
| `create_collection_if_not_exists` | `bool` | `True` | Создать коллекцию, если не существует |

**Основные методы:**

| Метод | Описание |
|-------|----------|
| `add_documents(documents, **kwargs)` | Добавить список `Document` в коллекцию; возвращает `list[str]` (ID) |
| `add_texts(texts, metadatas, **kwargs)` | Добавить список строк с опциональными метаданными |
| `similarity_search(query, k=4, filter=None)` | Поиск k ближайших документов; возвращает `list[Document]` |
| `similarity_search_with_score(query, k=4)` | То же + расстояние; возвращает `list[tuple[Document, float]]` |
| `max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5)` | MMR-поиск с балансом релевантности и разнообразия |
| `delete(ids)` | Удалить документы по ID |
| `get(ids=None, where=None, limit=None)` | Получить документы по фильтру |
| `as_retriever(**kwargs)` | Создать `VectorStoreRetriever` |
| `from_documents(documents, embedding, **kwargs)` | Classmethod: создать store и добавить документы за один вызов |
| `from_texts(texts, embedding, metadatas, **kwargs)` | Classmethod: аналогично, но из строк |

**Пример:**

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

store = Chroma.from_texts(
    texts=["Document one", "Document two"],
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    metadatas=[{"source": "a"}, {"source": "b"}],
    collection_name="demo",
)
results = store.similarity_search("query", k=2)
```

### `OpenAIEmbeddings`

**Импорт:** `from langchain_openai import OpenAIEmbeddings`

Embedding-модель через OpenAI API. Требует `OPENAI_API_KEY`.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | `str` | `"text-embedding-ada-002"` | Имя модели (`text-embedding-3-small`, `text-embedding-3-large`) |
| `dimensions` | `int \| None` | `None` | Укороченная размерность вектора (только для `3-*` моделей) |
| `api_key` | `str \| None` | `None` | API-ключ; если `None`, берётся из `OPENAI_API_KEY` |
| `organization` | `str \| None` | `None` | ID организации OpenAI |
| `base_url` | `str \| None` | `None` | Кастомный URL API (для прокси, Azure и т.д.) |
| `chunk_size` | `int` | `1000` | Размер батча при embed_documents |
| `max_retries` | `int` | `2` | Количество повторов при ошибках API |
| `timeout` | `float \| None` | `None` | Таймаут запроса в секундах |

**Методы:**

| Метод | Описание |
|-------|----------|
| `embed_query(text) -> list[float]` | Получить вектор для одного текста (запроса) |
| `embed_documents(texts) -> list[list[float]]` | Получить векторы для списка текстов (батч) |

**Пример:**

```python
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vector = emb.embed_query("What is machine learning?")
print(f"Dimensions: {len(vector)}")
```

### `HuggingFaceEmbeddings`

**Импорт:** `from langchain_huggingface import HuggingFaceEmbeddings`
(альтернативно: `from langchain_community.embeddings import HuggingFaceEmbeddings`)

Локальная embedding-модель из HuggingFace Hub. Бесплатна, работает offline, но требует скачивания модели (~90 MB для MiniLM).

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model_name` | `str` | `"sentence-transformers/all-mpnet-base-v2"` | Имя модели на HuggingFace Hub |
| `model_kwargs` | `dict` | `{}` | Аргументы для `SentenceTransformer()` (например, `{"device": "cuda"}`) |
| `encode_kwargs` | `dict` | `{}` | Аргументы для `.encode()` (например, `{"normalize_embeddings": True}`) |
| `cache_folder` | `str \| None` | `None` | Папка для кэша скачанных моделей |
| `multi_process` | `bool` | `False` | Использовать multiprocessing для encode |

**Методы:**

| Метод | Описание |
|-------|----------|
| `embed_query(text) -> list[float]` | Получить вектор для одного текста |
| `embed_documents(texts) -> list[list[float]]` | Получить векторы для списка текстов |

**Пример:**

```python
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vector = emb.embed_query("student essay about climate")
print(f"Dimensions: {len(vector)}")
```

### `RecursiveCharacterTextSplitter`

**Импорт:** `from langchain_text_splitters import RecursiveCharacterTextSplitter`

Разделяет текст на чанки, рекурсивно применяя иерархию разделителей для сохранения семантической целостности.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `chunk_size` | `int` | `4000` | Максимальный размер чанка (в единицах `length_function`) |
| `chunk_overlap` | `int` | `200` | Перекрытие между соседними чанками |
| `separators` | `list[str] \| None` | `["\n\n", "\n", " ", ""]` | Иерархия разделителей (от крупных к мелким) |
| `keep_separator` | `bool \| str` | `True` | Сохранять разделитель: `True`/`"start"`/`"end"`/`False` |
| `is_separator_regex` | `bool` | `False` | Интерпретировать separators как regex |
| `length_function` | `Callable[[str], int]` | `len` | Функция измерения длины (можно подставить `tiktoken` для подсчёта в токенах) |
| `add_start_index` | `bool` | `False` | Добавить позицию чанка в metadata (`start_index`) |
| `strip_whitespace` | `bool` | `True` | Убирать пробелы по краям чанков |

**Методы:**

| Метод | Описание |
|-------|----------|
| `split_text(text) -> list[str]` | Разделить строку на список строк |
| `split_documents(documents) -> list[Document]` | Разделить список Document; metadata копируется в каждый чанк |
| `create_documents(texts, metadatas) -> list[Document]` | Создать Document из строк и разделить |

**Пример:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)
chunks = splitter.split_text("Very long text here...")
```

### `PyPDFLoader`

**Импорт:** `from langchain_community.document_loaders import PyPDFLoader`

Загружает PDF-файл. Каждая страница становится отдельным `Document` с metadata `source` и `page`.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `file_path` | `str \| Path` | — (обязательный) | Путь к PDF-файлу |
| `password` | `str \| None` | `None` | Пароль для защищённого PDF |
| `extract_images` | `bool` | `False` | Извлекать текст из изображений (OCR) |

**Методы:**

| Метод | Описание |
|-------|----------|
| `load() -> list[Document]` | Загрузить все страницы |
| `load_and_split(text_splitter) -> list[Document]` | Загрузить и сразу разделить на чанки |
| `lazy_load() -> Iterator[Document]` | Ленивая загрузка (по одной странице) |

**Пример:**

```python
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("data/syllabus.pdf").load()
print(docs[0].metadata)
```

### `TextLoader`

**Импорт:** `from langchain_community.document_loaders import TextLoader`

Загружает текстовый файл как один `Document`.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `file_path` | `str \| Path` | — (обязательный) | Путь к текстовому файлу |
| `encoding` | `str \| None` | `None` | Кодировка файла (`"utf-8"`, `"latin-1"` и т.д.) |
| `autodetect_encoding` | `bool` | `False` | Автоматически определять кодировку |

**Методы:**

| Метод | Описание |
|-------|----------|
| `load() -> list[Document]` | Загрузить файл как один Document |
| `lazy_load() -> Iterator[Document]` | Ленивая загрузка |

**Пример:**

```python
from langchain_community.document_loaders import TextLoader

docs = TextLoader("data/essay.txt", encoding="utf-8").load()
```

### `DirectoryLoader`

**Импорт:** `from langchain_community.document_loaders import DirectoryLoader`

Рекурсивно загружает файлы из директории, применяя указанный loader к каждому файлу.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `path` | `str` | — (обязательный) | Путь к директории |
| `glob` | `str \| list[str]` | `"**/[!.]*"` | Glob-паттерн для фильтрации файлов |
| `loader_cls` | `type[BaseLoader]` | `UnstructuredFileLoader` | Класс loader'а для каждого файла |
| `loader_kwargs` | `dict \| None` | `None` | Аргументы для конструктора loader'а |
| `recursive` | `bool` | `False` | Рекурсивный обход поддиректорий |
| `show_progress` | `bool` | `False` | Показывать progress bar |
| `use_multithreading` | `bool` | `False` | Параллельная загрузка |
| `max_concurrency` | `int` | `4` | Максимум потоков при multithreading |
| `sample_size` | `int` | `0` | Загрузить только N файлов (0 = все) |
| `randomize_sample` | `bool` | `False` | Рандомизировать выборку |

**Пример:**

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

docs = DirectoryLoader(
    "data/essays/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
).load()
```

### `VectorStoreRetriever`

**Создание:** `vectorstore.as_retriever(**kwargs)`

Retriever-обёртка над vector store. Реализует Runnable-интерфейс — можно использовать в LCEL с `|`.

**Параметры `as_retriever()`:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `search_type` | `str` | `"similarity"` | Тип поиска: `"similarity"`, `"mmr"`, `"similarity_score_threshold"` |
| `search_kwargs` | `dict` | `{}` | Аргументы поиска (см. ниже) |

**search_kwargs по типу поиска:**

| Ключ | Для типа | Тип | По умолчанию | Описание |
|------|----------|-----|--------------|----------|
| `k` | все | `int` | `4` | Количество результатов |
| `filter` | все | `dict` | `None` | Фильтр по метаданным |
| `fetch_k` | `mmr` | `int` | `20` | Количество кандидатов для MMR |
| `lambda_mult` | `mmr` | `float` | `0.5` | Баланс релевантность/разнообразие (0–1) |
| `score_threshold` | `similarity_score_threshold` | `float` | — | Минимальный порог relevance score |

**Методы:**

| Метод | Описание |
|-------|----------|
| `invoke(query) -> list[Document]` | Синхронный поиск |
| `ainvoke(query) -> list[Document]` | Асинхронный поиск |
| `batch(queries) -> list[list[Document]]` | Батч-поиск |

**Пример:**

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7},
)
docs = retriever.invoke("well-structured essay with citations")
```

### `Document`

**Импорт:** `from langchain_core.documents import Document`

Базовый контейнер для текста с метаданными. Используется повсеместно в LangChain — от loader'ов до retriever'ов.

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `page_content` | `str` | — (обязательный) | Текстовое содержимое документа |
| `metadata` | `dict` | `{}` | Произвольные метаданные (source, page, date и т.д.) |
| `id` | `str \| None` | `None` | Уникальный идентификатор |
| `type` | `str` | `"Document"` | Тип документа (используется для сериализации) |

**Пример:**

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The impact of AI on modern education is profound...",
    metadata={"source": "essay_1.txt", "quality": "high", "score": 91},
)
print(doc.page_content[:50])
print(doc.metadata["source"])
```

---

## Практика: роутер `/api/v1/rag`

Мы создадим четыре эндпоинта, покрывающих полный RAG-пайплайн:

| Эндпоинт | Что делает |
|----------|------------|
| `POST /index` | Индексирует текстовые документы в ChromaDB |
| `POST /search` | Ищет похожие документы по запросу |
| `POST /assess` | Оценка с RAG-контекстом (прошлые похожие оценки) |
| `POST /compare` | Сравнение обычной и RAG-оценки side-by-side |

### Шаг 1. Схемы данных — `app/schemas/rag.py`

```python
from pydantic import BaseModel, Field

from app.schemas.assessment import AssessmentResponse


class TextDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[TextDocument]
    chunk_size: int = 500
    chunk_overlap: int = 50


class IndexResponse(BaseModel):
    chunk_count: int
    document_count: int
    collection_name: str


class SearchRequest(BaseModel):
    query: str
    k: int = 3


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class RagAssessRequest(BaseModel):
    student_work: str
    rubric_id: str | None = "essay_default"
    k: int = 3


class RagAssessResponse(BaseModel):
    assessment: AssessmentResponse
    retrieved_context: list[SearchResult]


class CompareRequest(BaseModel):
    student_work: str
    rubric_id: str | None = "essay_default"
    k: int = 3


class CompareResponse(BaseModel):
    normal: AssessmentResponse
    rag: AssessmentResponse
    retrieved_docs: list[SearchResult]
    normal_feedback_length: int
    rag_feedback_length: int
```

Схемы разделены по операциям. `SearchResult` используется повторно в `SearchResponse`, `RagAssessResponse` и `CompareResponse` — RORO-паттерн (Receive an Object, Return an Object).

### Шаг 2. Сервис vector store — `app/services/vector_store.py`

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        settings = get_settings()
        _vectorstore = Chroma(
            collection_name="assessments",
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key,
            ),
        )
    return _vectorstore


def index_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = []
    for i, text in enumerate(texts):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        documents.append(Document(page_content=text, metadata=meta))
    chunks = splitter.split_documents(documents)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return {
        "chunk_count": len(chunks),
        "document_count": len(texts),
    }


def search_similar(query: str, k: int = 3) -> list[tuple[Document, float]]:
    vs = get_vectorstore()
    raw = vs.similarity_search_with_score(query, k=k)
    return [(doc, 1.0 / (1.0 + dist)) for doc, dist in raw]


def format_retrieved_docs(results: list[tuple[Document, float]]) -> str:
    parts = []
    for doc, score in results:
        parts.append(f"[Relevance: {score:.3f}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

Ключевые решения:
- **Singleton-паттерн** для vector store через `_vectorstore` — аналог `@lru_cache` в `dependencies.py`. Коллекция создаётся один раз и переиспользуется.
- **`search_similar`** преобразует расстояние ChromaDB (L2, чем меньше — тем ближе) в relevance score (0–1, чем больше — тем релевантнее) по формуле `1 / (1 + distance)`.
- **`format_retrieved_docs`** формирует строку для вставки в промпт — каждый документ с его relevance score.

### Шаг 3. Роутер — `app/api/v1/rag.py`

```python
from fastapi import APIRouter, HTTPException
from langchain_core.prompts import ChatPromptTemplate

from app.dependencies import ChainDep, LLMDep, RubricStoreDep
from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse
from app.schemas.rag import (
    CompareRequest,
    CompareResponse,
    IndexRequest,
    IndexResponse,
    RagAssessRequest,
    RagAssessResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.vector_store import (
    format_retrieved_docs,
    index_documents,
    search_similar,
)

router = APIRouter(prefix="/rag", tags=["lesson-5-rag"])

RAG_SYSTEM_PROMPT = ASSESSMENT_SYSTEM_PROMPT + """

## Previously Assessed Similar Works
Use these as calibration — similar quality should receive similar scores.

{context}
"""


@router.post("/index")
async def index_docs(request: IndexRequest) -> IndexResponse:
    texts = [doc.content for doc in request.documents]
    metadatas = [doc.metadata for doc in request.documents]
    result = index_documents(
        texts=texts,
        metadatas=metadatas,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return IndexResponse(
        chunk_count=result["chunk_count"],
        document_count=result["document_count"],
        collection_name="assessments",
    )


@router.post("/search")
async def search_docs(request: SearchRequest) -> SearchResponse:
    results = search_similar(query=request.query, k=request.k)
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
            )
            for doc, score in results
        ],
    )


@router.post("/assess")
async def rag_assess(
    request: RagAssessRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> RagAssessResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    results = search_similar(query=request.student_work, k=request.k)
    context = format_retrieved_docs(results)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Please assess the following student work:\n\n{student_work}"),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )

    structured_llm = llm.with_structured_output(AssessmentResponse)
    chain = prompt | structured_llm
    assessment = await chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "context": context,
    })

    retrieved = [
        SearchResult(content=doc.page_content, metadata=doc.metadata, score=float(score))
        for doc, score in results
    ]
    return RagAssessResponse(assessment=assessment, retrieved_context=retrieved)


@router.post("/compare")
async def compare_assessment(
    request: CompareRequest,
    chain: ChainDep,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> CompareResponse:
    rubric = _resolve_rubric(request.rubric_id, rubrics)
    rubric_text = _format_rubric(rubric)

    normal_result = await chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
    })

    results = search_similar(query=request.student_work, k=request.k)
    context = format_retrieved_docs(results)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Please assess the following student work:\n\n{student_work}"),
    ]).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )

    structured_llm = llm.with_structured_output(AssessmentResponse)
    rag_chain = rag_prompt | structured_llm
    rag_result = await rag_chain.ainvoke({
        "student_work": request.student_work,
        "rubric": rubric_text,
        "context": context,
    })

    retrieved = [
        SearchResult(content=doc.page_content, metadata=doc.metadata, score=float(score))
        for doc, score in results
    ]

    def feedback_len(resp: AssessmentResponse) -> int:
        return sum(len(c.feedback) for c in resp.criterion_scores)

    return CompareResponse(
        normal=normal_result,
        rag=rag_result,
        retrieved_docs=retrieved,
        normal_feedback_length=feedback_len(normal_result),
        rag_feedback_length=feedback_len(rag_result),
    )


def _resolve_rubric(rubric_id, rubrics):
    rid = rubric_id or "essay_default"
    if rid not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rid}' not found")
    return rubrics[rid]


def _format_rubric(rubric):
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)
```

Обрати внимание на паттерны:
- **`RAG_SYSTEM_PROMPT`** расширяет базовый `ASSESSMENT_SYSTEM_PROMPT`, добавляя секцию `{context}` для найденных документов. Это сохраняет все инструкции оригинального промпта.
- **`/compare`** использует `ChainDep` (обычная цепочка из `dependencies.py`) для normal-оценки и строит RAG-цепочку вручную для RAG-оценки — так видно разницу.
- **`_resolve_rubric`** и **`_format_rubric`** дублируют логику из `app/api/v1/assessment.py`. В реальном проекте стоит вынести в общий utility-модуль.

### Шаг 4. Регистрация в `app/api/router.py`

Добавь импорт и include нового роутера:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rag, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(rag.router)
```

### Шаг 5. Тестирование

Установи RAG-зависимости и запусти сервер:

```bash
uv pip install -e ".[rag]"
uvicorn app.main:app --reload
```

**Индексирование документов:**

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "The impact of artificial intelligence on modern education is profound and multifaceted. AI-powered tools are transforming how students learn, enabling personalized learning paths and immediate feedback. Research by Smith (2023) demonstrates that AI tutoring systems improve test scores by 15-20% compared to traditional methods. However, concerns about academic integrity and over-reliance on technology remain valid.",
        "metadata": {"source": "essay_ai_education", "quality": "high", "score": 88}
      },
      {
        "content": "Social media is bad for kids. Everyone knows this. Kids spend too much time on their phones and dont study. The government should ban social media for anyone under 18. This is my opinion and I think its right.",
        "metadata": {"source": "essay_social_media", "quality": "low", "score": 35}
      },
      {
        "content": "Climate change presents one of the most significant challenges of the 21st century. According to the IPCC (2023), global temperatures have risen by 1.1 degrees Celsius since pre-industrial times. This essay examines three key mitigation strategies: carbon pricing, renewable energy investment, and reforestation programs.",
        "metadata": {"source": "essay_climate", "quality": "medium", "score": 72}
      }
    ],
    "chunk_size": 500,
    "chunk_overlap": 50
  }'
```

**Поиск похожих документов:**

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence impact on learning", "k": 2}' | python -m json.tool
```

**RAG-оценка:**

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Artificial intelligence is revolutionizing education across the globe. Machine learning algorithms now power adaptive learning platforms that adjust difficulty based on student performance (Johnson, 2024). This essay argues that AI integration in classrooms, when properly implemented, leads to measurably better learning outcomes.",
    "rubric_id": "essay_default",
    "k": 3
  }' | python -m json.tool
```

**Сравнение normal vs RAG:**

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/compare \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "AI is changing how we learn. Some people think its good, others think its bad. I think AI will help students learn better because computers are smart.",
    "rubric_id": "essay_default",
    "k": 3
  }' | python -m json.tool
```

### Связь с теорией

Каждый эндпоинт соответствует этапу RAG-пайплайна из теории:

| Эндпоинт | Теоретический концепт | Что демонстрирует |
|----------|----------------------|-------------------|
| `POST /index` | Text Splitters + Vector Store (разделы 3, 5) | Chunking документов и индексация embedding'ов в ChromaDB |
| `POST /search` | Retriever + Similarity Search (раздел 6) | Поиск по семантическому сходству с relevance score |
| `POST /assess` | RAG Chain (раздел 7) | Полный пайплайн: retrieve → format → prompt → LLM |
| `POST /compare` | RAG vs no-RAG | Измеримое влияние RAG на качество оценки |

Сервис `vector_store.py` инкапсулирует работу с Chroma и embedding-моделью (разделы 2, 3), а функция `format_retrieved_docs` реализует этап форматирования контекста из раздела 7.

---

## Чеклист самопроверки

- [ ] Объясни RAG-пайплайн: query → embed → search → format → prompt → LLM. Зачем каждый шаг?
- [ ] Что такое embedding и cosine similarity? Почему семантически похожие тексты получают близкие векторы?
- [ ] Как chunk_size влияет на качество поиска? Почему 500 — разумный дефолт?
- [ ] В чём разница между similarity search и MMR? В каких случаях MMR даёт лучший результат?
- [ ] Зачем overlap при chunking? Что теряется без него?
- [ ] RAG vs fine-tuning: назови три преимущества каждого подхода
- [ ] Что произойдёт, если использовать разные embedding-модели при индексации и поиске?
- [ ] Как metadata filtering помогает в vector search?
- [ ] Запусти `/compare` — в чём разница между normal и RAG-оценками?

---

## Частые ошибки

### 1. Слишком большие чанки

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
```

Embedding «усредняет» смысл: чем длиннее текст, тем менее специфичен его вектор. Для поиска оптимальны чанки 300–800 символов.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

### 2. Документы без метаданных

```python
docs = [Document(page_content=text)]
```

Без метаданных невозможно отследить источник, фильтровать результаты или показать attribution пользователю.

```python
docs = [Document(
    page_content=text,
    metadata={"source": "essay_1.txt", "quality": "high"},
)]
```

### 3. Разные embedding-модели для индексации и поиска

```python
Chroma.from_texts(texts, embedding=OpenAIEmbeddings())

store = Chroma(embedding_function=HuggingFaceEmbeddings())
store.similarity_search("query")
```

Векторные пространства разных моделей несовместимы. OpenAI выдаёт 1536-мерные векторы, MiniLM — 384-мерные. Даже модели одной размерности дадут разные пространства. Всегда используй одну и ту же модель.

### 4. RAG без проверки релевантности

```python
context = retriever.invoke(query)
prompt = f"Context: {format(context)}\nQuestion: {query}"
```

Если в коллекции нет релевантных документов, retriever вернёт «лучшие из плохих» — нерелевантный контекст может ухудшить ответ. Используй score threshold или проверяй score вручную:

```python
results = vectorstore.similarity_search_with_score(query, k=5)
relevant = [(doc, score) for doc, score in results if score < 1.0]
```

### 5. Одна embedding-модель для разных языков

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

`all-MiniLM-L6-v2` обучена на английском тексте. Русский текст будет индексироваться, но качество поиска значительно ниже. Для мультиязычных задач используй мультиязычную модель:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
```

---

## Что читать дальше

- [LangChain RAG Conceptual Guide](https://python.langchain.com/docs/concepts/rag/) — архитектура RAG
- [LangChain Retrievers](https://python.langchain.com/docs/concepts/retrievers/) — типы retriever'ов
- [ChromaDB Documentation](https://docs.trychroma.com/) — API и конфигурация vector store
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) — обзор подходов к нарезке
- Paper: *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* (Lewis et al., 2020) — оригинальная статья RAG
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — гайд по embedding-моделям

**Следующая тема:** [Тема 6: LangGraph + Agents](topic_06_langgraph_agents.md) — многошаговые графы и агенты.

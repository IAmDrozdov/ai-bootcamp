# Тема 5: RAG (Retrieval Augmented Generation)

> **Пререквизиты:** [Тема 1-4](topic_01_prompt_engineering.md) (основы LangChain)
> **Где в проекте:** `app/services/` (новые), `app/chains/` (RAG chain), `data/`
> **Зависимости:** `langchain-chroma`, `chromadb`, `sentence-transformers`, `pypdf` (группа `rag`)

---

## Теория

### 1. Зачем RAG

LLM обучена на публичных данных до определённой даты. Она **не знает**:
- Твои внутренние документы (рубрики, учебные планы, прошлые оценки)
- Данные после cutoff-даты обучения
- Конфиденциальную информацию

**Два подхода** к добавлению знаний:
1. **Fine-tuning** — дообучение модели на твоих данных. Дорого, сложно, результат "зашит" в веса.
2. **RAG** — находим релевантные документы и вставляем в промпт как контекст. Дёшево, гибко, обновляется мгновенно.

RAG = **Retrieval** (поиск документов) + **Augmented Generation** (генерация с дополнительным контекстом).

### 2. Embeddings — как текст превращается в вектор

Embedding — это числовое представление текста в многомерном пространстве. Семантически похожие тексты получают **близкие** векторы:

```
embed("capital of France")  →  [0.12, -0.34, 0.78, ...]  (1536 чисел)
embed("Paris is a city")    →  [0.11, -0.32, 0.76, ...]  ← близко!
embed("Python programming") →  [-0.45, 0.22, -0.11, ...]  ← далеко
```

Близость измеряется **cosine similarity**: cos(A, B) = 1 означает идентичные, 0 — не связаны, -1 — противоположные.

Embedding-модели:

| Модель | Размерность | Скорость | Качество | Стоимость |
|--------|------------|----------|----------|-----------|
| OpenAI text-embedding-3-small | 1536 | Быстро (API) | Высокое | $0.02/M tokens |
| OpenAI text-embedding-3-large | 3072 | Быстро (API) | Очень высокое | $0.13/M tokens |
| all-MiniLM-L6-v2 | 384 | Быстро (локально) | Хорошее | Бесплатно |
| BGE-base-en | 768 | Средне (локально) | Высокое | Бесплатно |

### 3. Vector Store — хранилище векторов

Vector store — это база данных, оптимизированная для **поиска по сходству** (similarity search). Ты сохраняешь документы с их embedding-векторами, а потом ищешь: "найди 5 документов, наиболее похожих на этот запрос".

ChromaDB — локальный vector store, идеальный для обучения:
- Работает in-memory или с файловым хранилищем
- Не требует отдельного сервера
- Поддерживает metadata filtering

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

results = vectorstore.similarity_search("thesis about AI", k=3)
```

### 4. Document Loaders — загрузка документов

LangChain предоставляет loader'ы для десятков форматов:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

loader = PyPDFLoader("data/syllabus.pdf")
docs = loader.load()
# docs = [Document(page_content="...", metadata={"source": "data/syllabus.pdf", "page": 0})]
```

Каждый `Document` содержит:
- `page_content` — текст
- `metadata` — словарь с метаданными (source, page, date, etc.)

### 5. Text Splitters — нарезка на чанки

Документ целиком не влезет в промпт и будет слишком "размытым" для embedding. Нужна **нарезка на чанки** (chunking):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_documents(docs)
```

**RecursiveCharacterTextSplitter** — самый универсальный. Пытается разделить по `\n\n` (параграфы), затем `\n` (строки), затем `. ` (предложения), затем ` ` (слова).

#### Chunk size vs quality

| chunk_size | Плюсы | Минусы |
|-----------|-------|--------|
| 200 | Точный поиск, релевантные фрагменты | Мало контекста, может обрезать мысль |
| 500 | Баланс точности и контекста | Оптимально для большинства задач |
| 1000 | Полный контекст абзацев | Менее точный поиск, больше "шума" |
| 2000+ | Целые секции | Embedding "усредняет" смысл, поиск деградирует |

**Overlap** — перекрытие между чанками. Если chunk_size=500 и overlap=50, последние 50 символов чанка N будут первыми 50 символами чанка N+1. Это предотвращает потерю смысла на границах.

### 6. Retriever и стратегии поиска

Retriever — абстракция LangChain для поиска документов:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # или "mmr"
    search_kwargs={"k": 3},
)
docs = retriever.invoke("thesis about AI impact")
```

**Similarity search** — возвращает k самых похожих документов. Проблема: если топ-3 документа почти одинаковы, ты получишь дублирующуюся информацию.

**MMR (Maximal Marginal Relevance)** — балансирует **релевантность** и **разнообразие**. Первый документ — самый релевантный, второй — релевантный, но максимально отличающийся от первого:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7},
)
```

`lambda_mult`: 1.0 = только релевантность (как similarity), 0.0 = только разнообразие.

### 7. RAG Chain — собираем всё вместе

```
query → retriever → format_docs → prompt → llm → parser
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

Что происходит:
1. Retriever находит 3 самых релевантных документа
2. `format_docs` склеивает их в одну строку
3. Контекст вставляется в промпт через `{context}`
4. LLM генерирует ответ, используя найденный контекст

---

## Практические задания

### Задание 1: Загрузка документов

**Цель:** создать набор документов и загрузить через Document Loaders.

**Файлы:** `data/sample_essays/`, `app/services/document_loader.py`

**Критерии успеха:**
- 5+ примеров эссе в `data/sample_essays/`
- Document loader корректно загружает все файлы
- Каждый Document имеет metadata (source, type)

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a document loading system for the RAG pipeline.

1. Create data/sample_essays/ directory with 5 text files:
   - strong_ai_essay.txt — well-written essay on AI (200+ words, citations, clear thesis)
   - weak_opinion_piece.txt — poor essay, no citations, vague arguments
   - medium_climate_essay.txt — decent essay on climate change
   - strong_education_essay.txt — strong essay on education
   - weak_tech_essay.txt — weak essay on technology

   Each file should be a realistic student essay with varying quality levels.

2. Create app/services/document_loader.py:
   - Function load_sample_essays() -> list[Document]:
     Uses DirectoryLoader to load all .txt files from data/sample_essays/
     Adds metadata: "type": "sample_essay", "quality": inferred from filename
   - Function load_rubrics() -> list[Document]:
     Loads rubric JSON files from data/rubrics/ as documents
     Adds metadata: "type": "rubric"
   - Function load_all_documents() -> list[Document]:
     Combines all document sources

3. Create experiments/t5_load_docs.py:
   - Load all documents
   - Print: count, sources, metadata for each
   - Verify all documents loaded correctly

Install RAG dependencies: uv pip install -e ".[rag]"
Run with: python -m experiments.t5_load_docs
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review the document loading implementation:

1. DOCUMENTS: Are 5+ sample essays created with varying quality levels?
2. LOADER: Does it use LangChain Document Loaders (not manual file reading)?
3. METADATA: Does each Document have meaningful metadata (source, type, quality)?
4. STRUCTURE: Is the loader a reusable service, not a script?
5. ERROR HANDLING: What happens if a file is missing or malformed?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Эксперимент с чанками

**Цель:** понять, как chunk_size влияет на качество поиска.

**Файлы:** `experiments/t5_chunking.py`

**Критерии успеха:**
- Документы индексируются с chunk_size 200, 500, 1000
- Один и тот же запрос выполняется на всех трёх индексах
- Сравнение: количество чанков, качество найденных результатов

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t5_chunking.py comparing different chunk sizes.

1. Load sample essays using the document loader from task 1

2. For each chunk_size in [200, 500, 1000]:
   - Split documents with RecursiveCharacterTextSplitter(chunk_size=X, chunk_overlap=50)
   - Create in-memory Chroma vectorstore with the chunks
   - Use sentence-transformers all-MiniLM-L6-v2 as embedding model (free, local)

3. Run 3 test queries against each index:
   - "essay with strong thesis and clear argument"
   - "work that uses peer-reviewed citations"
   - "well-structured essay with good transitions"

4. For each query × chunk_size, print:
   - Number of total chunks in the index
   - Top 3 results: first 100 chars + similarity score
   - Source document and chunk position

5. Print comparison table: chunk_size | total_chunks | avg_relevance_score

Use HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") for local embeddings.
Run with: python -m experiments.t5_chunking
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t5_chunking.py:

1. SPLITTING: Are 3 different chunk sizes tested with the same overlap?
2. EMBEDDINGS: Are local embeddings used (sentence-transformers)?
3. QUERIES: Are test queries meaningful and varied?
4. COMPARISON: Is there a clear table comparing chunk sizes?
5. ANALYSIS: Can you see how chunk size affects result quality?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: RAG chain

**Цель:** построить полный RAG pipeline, который находит похожие прошлые оценки и использует их как контекст.

**Файлы:** `app/chains/rag_assessment_chain.py`, `experiments/t5_rag_chain.py`

**Критерии успеха:**
- RAG chain находит 3 похожих прошлых эссе
- Вставляет их в промпт как дополнительный контекст
- Оценка учитывает найденные примеры

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a RAG-powered assessment chain.

1. Create app/chains/rag_assessment_chain.py:
   - Function build_rag_assessment_chain(llm, retriever) -> Runnable
   - Chain structure:
     RunnablePassthrough.assign(
         similar_essays=lambda x: retriever.invoke(x["student_work"])
     )
     | RunnablePassthrough.assign(
         context=lambda x: format_similar_essays(x["similar_essays"])
     )
     | rag_prompt
     | llm.with_structured_output(AssessmentResponse)
   
   - rag_prompt should extend the assessment prompt with:
     "## Previously assessed similar essays:\n{context}\n\n
     Use these as calibration — similar quality should get similar scores."

2. Create experiments/t5_rag_chain.py:
   - Load and index sample essays
   - Build RAG chain with retriever
   - Evaluate a new essay with RAG context
   - Print: which similar essays were retrieved, final assessment
   - Compare: run same essay with and without RAG, show score differences

Run with: python -m experiments.t5_rag_chain
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/chains/rag_assessment_chain.py:

1. RAG PATTERN: Is it retriever | format | prompt | llm (standard RAG chain)?
2. CONTEXT INJECTION: Are retrieved documents formatted and inserted into the prompt?
3. PROMPT: Does it instruct the model to use similar essays for calibration?
4. COMPARISON: Is there a with-RAG vs without-RAG comparison?
5. RETRIEVER: Is the retriever configurable (not hardcoded)?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: A/B сравнение RAG vs no-RAG

**Цель:** измерить, улучшает ли RAG качество оценок.

**Файлы:** `experiments/t5_rag_ab_test.py`

**Критерии успеха:**
- 5 эссе оцениваются с RAG и без
- Сравнение: детализация фидбека, consistency оценок, время ответа

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t5_rag_ab_test.py — A/B comparison of RAG vs no-RAG assessment.

1. Prepare 5 test essays of varying quality (can reuse from sample_essays)

2. For each essay, run:
   a. WITHOUT RAG: standard assessment chain
   b. WITH RAG: RAG assessment chain from task 3

3. Compare for each essay:
   - Overall scores (RAG vs no-RAG)
   - Feedback length per criterion
   - Whether RAG feedback references the similar essays
   - Execution time (RAG is slower due to retrieval)

4. Aggregate metrics:
   - Mean score difference (RAG - no-RAG)
   - Mean feedback length difference
   - Time overhead of RAG

5. Print detailed comparison table and summary

Use temperature=0 for deterministic results.
Run with: python -m experiments.t5_rag_ab_test
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t5_rag_ab_test.py:

1. FAIR COMPARISON: Same temperature, same model, same rubric for both?
2. SAMPLE SIZE: 5 essays with different quality levels?
3. METRICS: Score comparison, feedback length, timing all measured?
4. CLARITY: Can you see whether RAG improves or not?
5. CONTEXT: Does the RAG version actually retrieve relevant documents?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 5: Embeddings — облако vs локально

**Цель:** сравнить OpenAI embeddings и sentence-transformers по качеству, скорости, стоимости.

**Файлы:** `experiments/t5_embeddings_comparison.py`

**Критерии успеха:**
- Один и тот же набор документов индексируется обоими embedding-моделями
- Сравнение: качество поиска, скорость индексации, скорость поиска

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t5_embeddings_comparison.py comparing cloud vs local embeddings.

1. Two embedding models:
   a. OpenAIEmbeddings(model="text-embedding-3-small") — cloud
   b. HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") — local

2. For each model:
   - Measure indexing time: embed all sample essay chunks
   - Measure query time: 5 test queries, average time per query
   - Measure search quality: for each query, print top-3 results

3. Compare:
   - Indexing speed (seconds)
   - Query speed (ms per query)
   - Relevance of top results (manual inspection)
   - Cost (OpenAI has cost, HuggingFace is free)
   - Vector dimensions (1536 vs 384)

4. Print comparison table with all metrics

Make sure OPENAI_API_KEY is set in .env for OpenAI embeddings.
Run with: python -m experiments.t5_embeddings_comparison
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t5_embeddings_comparison.py:

1. TWO MODELS: Are both OpenAI and HuggingFace embeddings tested?
2. FAIR COMPARISON: Same documents, same queries, same chunk settings?
3. TIMING: Are indexing and query times measured separately?
4. QUALITY: Can you compare search results between models?
5. COST: Is cost difference noted (OpenAI = paid, HuggingFace = free)?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Объясни RAG pipeline: query → retriever → format → prompt → LLM. Зачем каждый шаг?
- [ ] Что такое embedding и cosine similarity? Почему семантически похожие тексты имеют близкие векторы?
- [ ] Как chunk_size влияет на качество поиска? Почему 500 — хороший дефолт?
- [ ] В чём разница между similarity search и MMR? Когда MMR лучше?
- [ ] Зачем overlap при chunking? Что теряется без него?
- [ ] RAG vs fine-tuning: назови 3 преимущества RAG.

---

## Частые ошибки

### 1. Слишком большие чанки

```python
# Плохо: целый документ — один чанк
splitter = RecursiveCharacterTextSplitter(chunk_size=10000)

# Хорошо: 300-800 символов для поиска
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

### 2. Забыть metadata

```python
# Плохо: документы без метаданных — невозможно отследить источник
docs = [Document(page_content=text)]

# Хорошо: metadata для фильтрации и attribution
docs = [Document(page_content=text, metadata={"source": "essay_1.txt", "quality": "high"})]
```

### 3. Один embedding-модель для разных языков

```python
# Плохо: all-MiniLM-L6-v2 обучена на английском, русский текст работает хуже
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Для мультиязычных задач: multilingual модель
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

### 4. RAG без проверки релевантности

```python
# Плохо: слепо вставляем в промпт всё, что нашлось
context = retriever.invoke(query)
prompt = f"Context: {format(context)}\nQuestion: {query}"

# Лучше: проверить similarity score, отфильтровать нерелевантные
results = vectorstore.similarity_search_with_score(query, k=5)
relevant = [(doc, score) for doc, score in results if score > 0.7]
```

---

## Что читать дальше

- [LangChain RAG Conceptual Guide](https://python.langchain.com/docs/concepts/rag/) — как устроен RAG
- [LangChain Retrievers](https://python.langchain.com/docs/concepts/retrievers/) — типы retriever'ов
- [ChromaDB Docs](https://docs.trychroma.com/) — vector store
- Paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) — обзор подходов

**Следующая тема:** [Тема 6: LangGraph + Agents](topic_06_langgraph_agents.md) — как строить многошаговые агенты.

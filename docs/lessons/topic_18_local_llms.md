# Тема 18: Ollama и локальные LLM — запуск моделей без облака

> **Пререквизиты:** [Тема 2: LangChain Core + LCEL](topic_02_langchain_lcel.md)
> **Что добавляем в проект:** `app/api/v1/local.py`, `app/services/local_llm.py`, `app/schemas/local.py`
> **Зависимости:** `langchain-ollama`, `langchain-community`, `ollama`

---

## Теория

### 1. Зачем локальные модели

Каждый вызов облачной LLM — это деньги. Claude Sonnet стоит $3/$15 за миллион input/output токенов, GPT-4o — $2.50/$10. При 10 000 оценок студенческих работ в месяц (каждая ~2000 токенов input + 500 output) расход составит $30-150 только на inference. Локальная модель стоит $0 за inference — платишь только за электричество.

Но стоимость — не единственная причина запускать модели локально.

**Приватность данных.** Когда студенческие работы содержат персональные данные, отправка в облако создаёт юридические риски. GDPR в Европе, FERPA в США, 152-ФЗ в России — все регулируют передачу персональных данных третьим лицам. Локальная модель обрабатывает всё на твоей машине. Данные не покидают периметр.

**Скорость итераций.** Облачные API имеют rate limits: 60 RPM для Claude, 500 RPM для GPT-4o-mini. При разработке промптов ты вызываешь модель сотни раз в час. Локальная модель не имеет rate limits — запускай 1000 запросов в минуту, если железо позволяет. Нет и сетевой задержки: 0ms latency на подключение vs 100-300ms для облака.

**Офлайн-работа.** В поезде, самолёте, за городом — локальная модель работает без интернета. Для bootcamp-проекта это удобно: можно разрабатывать и тестировать где угодно.

**Fine-tuning.** Облачные модели — чёрный ящик. Локальную модель можно дообучить на специфичных данных. Например, на 10 000 пар «студенческая работа → оценка», чтобы модель лучше понимала критерии конкретного курса.

**Ограничения.** Локальные модели уступают по качеству: Llama 3.1 8B слабее Claude Sonnet, особенно в сложных reasoning-задачах. Нужен GPU (или мощный CPU) для приемлемой скорости. Модель 70B требует 40+ GB RAM — это не каждый ноутбук.

**Когда использовать локальные модели:**

| Сценарий | Локальная модель | Облачная модель |
|----------|-----------------|-----------------|
| Разработка и тестирование | Быстро, бесплатно, нет rate limits | Дорого, медленные итерации |
| Preprocessing (классификация, фильтрация) | Достаточно качества, $0 | Избыточно для простых задач |
| Embeddings | Бесплатно, без лимитов | $0.02-0.13/M tokens |
| Финальная оценка сложных работ | Может не хватить качества | Лучший выбор |
| Fallback при недоступности API | Система продолжает работать | API может упасть |
| Конфиденциальные данные | Данные не покидают машину | Риски compliance |

Для нашего assessment проекта идеальна **гибридная стратегия**: локальная модель для разработки, тестирования и простых задач; облачная — для production-оценки сложных работ.

### 2. Ollama — запуск LLM одной командой

Ollama — runtime для запуска LLM на локальной машине. macOS, Linux, Windows. Одна команда — и модель работает.

**Установка:**

```bash
curl -fsSL https://ollama.com/install.sh | sh

brew install ollama
```

На macOS рекомендуется `brew install ollama` — он добавит Ollama как фоновый сервис. На Linux — curl-скрипт установит systemd-сервис автоматически.

**Архитектура Ollama:**

```
┌─────────────────────────────────────────────────┐
│  Твоё приложение (FastAPI)                      │
│  → ChatOllama(model="llama3.1")                 │
│  → HTTP POST http://localhost:11434/api/chat    │
└───────────────────┬─────────────────────────────┘
                    │ HTTP (localhost)
┌───────────────────▼─────────────────────────────┐
│  Ollama Server (порт 11434)                     │
│  ├── API handler (REST)                         │
│  ├── Model loader (GGUF → memory)               │
│  ├── Inference engine (llama.cpp)               │
│  └── GPU scheduler (CUDA / Metal)               │
├─────────────────────────────────────────────────┤
│  ~/.ollama/models/                              │
│  ├── manifests/  (метаданные моделей)           │
│  └── blobs/     (веса моделей, GGUF)           │
└─────────────────────────────────────────────────┘
```

Ollama запускает HTTP-сервер на `localhost:11434`. Все вызовы — обычные HTTP-запросы. Модели хранятся в `~/.ollama/models/` и загружаются в RAM/VRAM при первом использовании. Ollama автоматически определяет GPU и использует его: NVIDIA через CUDA, Apple Silicon через Metal.

**Основные команды:**

| Команда | Описание | Пример |
|---------|----------|--------|
| `ollama pull` | Скачать модель | `ollama pull llama3.1:8b` |
| `ollama run` | Скачать (если нет) и запустить интерактивный чат | `ollama run llama3.1` |
| `ollama list` | Показать скачанные модели | `ollama list` |
| `ollama rm` | Удалить модель | `ollama rm llama3.1:8b` |
| `ollama show` | Показать информацию о модели | `ollama show llama3.1` |
| `ollama ps` | Показать запущенные модели | `ollama ps` |
| `ollama serve` | Запустить сервер (если не запущен) | `ollama serve` |
| `ollama create` | Создать кастомную модель из Modelfile | `ollama create my-model -f Modelfile` |

**Modelfile — кастомизация модели:**

Modelfile позволяет создать свою «версию» модели с заданным system prompt, параметрами и шаблоном. Это не fine-tuning — это преднастроенная конфигурация.

```dockerfile
FROM llama3.1:8b

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
PARAMETER top_p 0.9

SYSTEM """
You are an expert teacher evaluating student work.
Always provide constructive feedback with specific examples.
Score each criterion from 0 to the maximum points.
Respond in Russian.
"""
```

Создание и использование:

```bash
ollama create assessment-model -f Modelfile

ollama run assessment-model
```

Теперь `assessment-model` доступна как обычная модель: `ChatOllama(model="assessment-model")`.

**Требования к памяти:**

| Размер модели | RAM/VRAM (fp16) | RAM/VRAM (q4_K_M) | Скорость на M1 | Скорость на RTX 3090 |
|---------------|----------------|-------------------|-----------------|---------------------|
| 1-3B (Phi-3, Gemma 2B) | 2-6 GB | 1-2 GB | ~50 tok/s | ~80 tok/s |
| 7-8B (Llama 3.1 8B) | 14-16 GB | 4-5 GB | ~25 tok/s | ~60 tok/s |
| 13B (Llama 2 13B) | 26 GB | 7-8 GB | ~15 tok/s | ~40 tok/s |
| 34B (CodeLlama 34B) | 68 GB | 18-20 GB | ~5 tok/s | ~25 tok/s |
| 70B (Llama 3.1 70B) | 140 GB | 35-40 GB | Не влезет | ~15 tok/s |

Правило: для q4_K_M квантизации нужно примерно **0.5 GB RAM на 1B параметров**. Для fp16 — примерно 2 GB на 1B.

### 3. Модели — обзор и выбор

Open-source модели догоняют проприетарные. В 2024-2025 году появилось множество сильных моделей, доступных бесплатно. Вот ключевые для нашего проекта:

**Llama 3.1 / 3.2 (Meta)**

Флагман open-source. Доступен в размерах 1B, 3B, 8B, 70B, 405B. Llama 3.1 8B — лучший баланс качества и скорости для локальной работы. Поддерживает tool calling, JSON mode, контекст 128K токенов. Llama 3.2 добавляет мультимодальность (vision) в 11B и 90B вариантах.

```bash
ollama pull llama3.1:8b
ollama pull llama3.2:3b
```

**Mistral / Mixtral (Mistral AI)**

Mistral 7B — один из первых сильных маленьких моделей. Mixtral 8x7B использует Mixture of Experts (MoE) архитектуру: 8 экспертов по 7B, из которых активируются 2 на каждый токен. Фактическое использование памяти как у 13B, но качество ближе к 30-40B. Сильный в reasoning и следовании инструкциям.

```bash
ollama pull mistral
ollama pull mixtral
```

**Gemma 2 (Google)**

Компактная модель от Google. Доступна в 2B и 9B вариантах. 9B Gemma 2 показывает результаты на уровне моделей 2-3x большего размера. Хороший выбор для ограниченных ресурсов.

```bash
ollama pull gemma2:9b
ollama pull gemma2:2b
```

**Qwen 2.5 (Alibaba)**

Сильная multilingual модель. Особенно хороша для задач с кодом (Qwen 2.5 Coder) и для неанглийских языков — включая русский. Доступна от 0.5B до 72B.

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

**Phi-3 / Phi-3.5 (Microsoft)**

Маленькая но умная. 3.8B параметров, но обучена на синтетических данных высокого качества. Удивительно хороша для своего размера, особенно в reasoning. Идеальна для быстрого прототипирования.

```bash
ollama pull phi3
```

**CodeLlama / DeepSeek Coder**

Специализированные модели для работы с кодом. DeepSeek Coder V2 особенно сильна в генерации и ревью кода.

```bash
ollama pull codellama:7b
ollama pull deepseek-coder-v2:16b
```

**Таблица выбора для нашего проекта:**

| Задача | Рекомендованная модель | Размер | Мин. RAM |
|--------|----------------------|--------|----------|
| Быстрая оценка (dev/test) | Llama 3.1 8B (q4_K_M) | 4.7 GB | 8 GB |
| Полная оценка (production) | Llama 3.1 70B (q4_K_M) | 40 GB | 48 GB |
| Классификация типа работы | Phi-3 (3.8B) | 2.3 GB | 4 GB |
| Embeddings | nomic-embed-text | 274 MB | 1 GB |
| Оценка кода | Qwen 2.5 Coder 7B | 4.7 GB | 8 GB |
| Multilingual оценка | Qwen 2.5 7B | 4.7 GB | 8 GB |
| Минимальные ресурсы | Gemma 2 2B | 1.6 GB | 4 GB |

Для assessment проекта рекомендуем: **Llama 3.1 8B** для быстрых задач в development, **Llama 3.1 70B** (если хватает RAM) для полной оценки, или **облачная Claude** для production.

### 4. Квантизация — компромисс качество/скорость/память

Модель Llama 3.1 8B в полной точности (fp16) занимает 16 GB. Это больше, чем RAM большинства ноутбуков может выделить для одной задачи. Квантизация решает эту проблему — уменьшает размер модели, снижая точность весов.

**Как работает квантизация:** каждый вес модели — число с плавающей запятой (float16, 16 бит). Квантизация превращает его в число с меньшей точностью: 8 бит, 4 бита, иногда 2 бита. Это как сжатие изображения из PNG в JPEG — теряешь немного качества, но файл в разы меньше.

**GGUF формат.** Стандарт для квантизированных моделей, разработанный в llama.cpp. Ollama использует именно GGUF. Формат хранит веса, метаданные, токенизатор — всё в одном файле.

**Уровни квантизации:**

| Квант | Биты/вес | Размер 8B модели | Качество (% от fp16) | Скорость |
|-------|---------|-------------------|---------------------|----------|
| `f16` | 16 | 16.0 GB | 100% (baseline) | 1.0x |
| `q8_0` | 8 | 8.0 GB | ~99.5% | 1.2x |
| `q6_K` | 6.6 | 6.6 GB | ~99% | 1.4x |
| `q5_K` | 5.5 | 5.5 GB | ~98.5% | 1.5x |
| `q4_K_M` | 4.8 | 4.8 GB | ~97.5% | 1.7x |
| `q4_0` | 4.0 | 4.0 GB | ~96% | 1.8x |
| `q3_K` | 3.5 | 3.5 GB | ~93% | 2.0x |
| `q2_K` | 2.5 | 2.5 GB | ~85% | 2.2x |

**Правило большого пальца: `q4_K_M` — лучший баланс.** Потеря качества минимальна (~2.5%), а размер уменьшается в 3.3 раза. Для большинства задач разница с fp16 незаметна.

Что значит «потеря качества»? Измеряется **perplexity** — мерой того, насколько хорошо модель предсказывает следующий токен. Чем ниже perplexity, тем лучше. На практике:

- `q4_K_M` vs `f16`: perplexity растёт на ~0.1-0.3 — в большинстве задач незаметно
- `q3_K` vs `f16`: perplexity растёт на ~0.5-1.0 — заметно на сложных reasoning
- `q2_K` vs `f16`: perplexity растёт на ~2-4 — существенная деградация

**Как выбрать квантизацию в Ollama:**

```bash
ollama pull llama3.1:8b-instruct-q4_K_M

ollama pull llama3.1:8b

ollama pull llama3.1:8b-instruct-q8_0
```

Формат тега: `model:size-variant-quantization`. Без указания квантизации Ollama выберет q4_0 или q4_K_M по умолчанию.

**Рекомендации по выбору:**

| Ситуация | Квантизация | Почему |
|----------|------------|--------|
| Мало RAM (8 GB) | q4_K_M или q4_0 | Минимальный размер при приемлемом качестве |
| Достаточно RAM (16+ GB) | q6_K или q8_0 | Лучшее качество без компромиссов |
| Максимальное качество | f16 | Только если хватает VRAM |
| Embedding модели | По умолчанию | Квантизация меньше влияет на embeddings |
| Быстрый прототип | q3_K | Максимальная скорость, качество второстепенно |

### 5. Интеграция с LangChain — ChatOllama и OllamaEmbeddings

LangChain предоставляет обёртки для Ollama, которые реализуют стандартный Runnable-интерфейс. Это означает, что локальная модель подключается к LCEL-цепочкам точно так же, как облачная.

**ChatOllama — основной класс:**

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,
    num_ctx=4096,
    num_predict=1024,
    base_url="http://localhost:11434",
)

response = await llm.ainvoke("Оцени эссе студента...")
```

Параметры `ChatOllama`:

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | `str` | Имя модели (как в `ollama list`) |
| `temperature` | `float` | Случайность генерации (0.0-1.0) |
| `base_url` | `str` | URL Ollama сервера (default: `http://localhost:11434`) |
| `num_ctx` | `int` | Размер контекстного окна в токенах |
| `num_predict` | `int` | Максимум токенов в ответе (-1 = без лимита) |
| `format` | `str` | Формат ответа (`"json"` для JSON mode) |
| `keep_alive` | `str` | Время жизни модели в памяти (`"5m"`, `"24h"`, `"-1"` = вечно) |
| `top_p` | `float` | Nucleus sampling (0.0-1.0) |
| `top_k` | `int` | Top-k sampling |
| `repeat_penalty` | `float` | Штраф за повторения (1.0 = нет штрафа) |
| `stop` | `list[str]` | Stop-последовательности |

**В LCEL-цепочках ChatOllama работает как любая ChatModel:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher evaluating student work. Respond in Russian."),
    ("human", "Оцени работу:\n\n{student_work}"),
])

llm = ChatOllama(model="llama3.1:8b", temperature=0.3)
parser = StrOutputParser()

chain = prompt | llm | parser

result = await chain.ainvoke({"student_work": "Эссе студента..."})
```

Замена модели — одна строка. Поменять `ChatOllama(model="llama3.1:8b")` на `ChatAnthropic(model="claude-sonnet-4-20250514")` — и вся цепочка работает с облачной моделью. Это сила абстракции LCEL.

**Structured output с Ollama:**

```python
from pydantic import BaseModel, Field

class QuickScore(BaseModel):
    score: int = Field(description="Score from 0 to 100")
    feedback: str = Field(description="Brief feedback in 2-3 sentences")

llm = ChatOllama(model="llama3.1:8b", temperature=0)
structured_llm = llm.with_structured_output(QuickScore)

result = await structured_llm.ainvoke("Score this essay: ...")
print(result.score, result.feedback)
```

Не все модели одинаково хорошо работают с `with_structured_output()`. Llama 3.1 поддерживает tool calling, что делает structured output надёжным. Для моделей без tool calling LangChain использует JSON mode — менее надёжно, но работает.

**Streaming:**

```python
async for chunk in llm.astream("Оцени эссе..."):
    print(chunk.content, end="", flush=True)
```

Streaming работает через Ollama API и возвращает токены по мере генерации — идентично облачным моделям.

**OllamaEmbeddings:**

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = await embeddings.aembed_query("Текст для vectorization")
print(len(vector))

vectors = await embeddings.aembed_documents(["Текст 1", "Текст 2", "Текст 3"])
```

### 6. vLLM — production inference server

Ollama оптимизирован для простоты: одна команда — модель работает. Но для production с множеством одновременных пользователей нужен более мощный сервер. vLLM — это high-performance inference engine, разработанный в UC Berkeley.

**Ключевые технологии vLLM:**

- **PagedAttention** — управление KV-cache по аналогии с виртуальной памятью ОС. Уменьшает потребление GPU-памяти на 60-80%.
- **Continuous batching** — новые запросы добавляются в batch без ожидания завершения текущих. Throughput выше в 2-4x по сравнению с наивным batching.
- **Tensor parallelism** — распределение модели по нескольким GPU.

**Установка и запуск:**

```bash
pip install vllm

vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

**Интеграция с LangChain.** vLLM предоставляет OpenAI-совместимый API, поэтому подключается через `ChatOpenAI`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0.3,
)

result = await llm.ainvoke("Оцени работу студента...")
```

Это работает потому, что vLLM эмулирует OpenAI API — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`. LangChain не знает, что общается с локальной моделью.

**Ollama vs vLLM:**

| Характеристика | Ollama | vLLM |
|---------------|--------|------|
| Назначение | Dev, single user | Production, multi-user |
| Установка | Одна команда | pip + конфигурация |
| Формат моделей | GGUF (квантизированные) | HuggingFace (fp16/bf16) |
| Batching | Нет | Continuous batching |
| GPU utilization | Средняя | Высокая (PagedAttention) |
| Throughput (8B, 1 GPU) | ~30 tok/s | ~100-200 tok/s |
| Concurrent users | 1-5 | 50-100+ |
| Квантизация | GGUF (q4, q8 и т.д.) | AWQ, GPTQ, FP8 |
| CPU inference | Да | Только GPU |
| API | Своё API | OpenAI-совместимое |

**Правило:** используй Ollama для разработки и тестирования (простота), vLLM для production (throughput и concurrency).

### 7. Локальные embeddings

Embeddings — векторные представления текста — нужны для поиска похожих работ, кластеризации, RAG. OpenAI embeddings (`text-embedding-3-small`) стоят $0.02/M tokens. Локальные — бесплатны.

**nomic-embed-text через Ollama:**

Самый простой способ получить локальные embeddings. 137M параметров, 768 dimensions, качество сравнимо с OpenAI ada-002.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = await embeddings.aembed_query("Студент продемонстрировал глубокое понимание темы")
print(f"Dimensions: {len(vector)}")
```

```bash
ollama pull nomic-embed-text
```

**sentence-transformers — Python-библиотека:**

Более гибкий подход. Модели скачиваются из HuggingFace и запускаются через PyTorch.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector = embeddings.embed_query("Текст для vectorization")
```

**Обзор моделей для embeddings:**

| Модель | Dimensions | Размер | Качество (MTEB) | Язык |
|--------|-----------|--------|-----------------|------|
| `nomic-embed-text` | 768 | 274 MB | Хорошее | EN, multi |
| `all-MiniLM-L6-v2` | 384 | 80 MB | Среднее | EN |
| `bge-large-en-v1.5` | 1024 | 1.3 GB | Высокое | EN |
| `multilingual-e5-large` | 1024 | 2.2 GB | Высокое | 100+ языков |
| `bge-m3` | 1024 | 2.2 GB | Высокое | 100+ языков |
| OpenAI `text-embedding-3-small` | 1536 | — | Высокое | Multi |
| OpenAI `text-embedding-3-large` | 3072 | — | Очень высокое | Multi |

**Для нашего проекта с русскоязычными работами** рекомендуем `multilingual-e5-large` или `bge-m3` — они хорошо работают с русским текстом. Для быстрого прототипирования `nomic-embed-text` через Ollama — проще всего.

**Сравнение с OpenAI embeddings:**

| Параметр | Локальные (nomic) | OpenAI (3-small) |
|----------|-------------------|------------------|
| Стоимость | $0 | $0.02/M tokens |
| Скорость (batch 100 текстов) | ~2-5 сек | ~1-3 сек + latency |
| Качество (MTEB avg) | 0.62 | 0.65 |
| Приватность | Полная | Данные на серверах OpenAI |
| Rate limits | Нет | 3000 RPM |
| Офлайн | Да | Нет |
| Setup | Ollama pull или pip install | API key |

Разница в качестве (0.62 vs 0.65) обычно незаметна для retrieval задач. Для нашего проекта локальные embeddings — отличный выбор для dev/test.

### 8. Гибридная стратегия — облако + локально

В реальном проекте не нужно выбирать «или/или». Лучшая стратегия — использовать обе: локальные модели для одних задач, облачные для других.

**Стратегия по окружению:**

| Окружение | Модель | Почему |
|-----------|--------|--------|
| Development | Локальная (Ollama) | Бесплатно, быстро, нет rate limits |
| Unit tests | Локальная (Ollama) | Тесты бесплатны и быстры |
| Integration tests | Облачная (Claude) | Проверяем с реальной production-моделью |
| Staging | Облачная (Claude) | Полная эмуляция production |
| Production | Облачная (Claude) + локальная fallback | Качество + отказоустойчивость |

**Fallback-цепочка в LangChain:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

primary = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)
fallback = ChatOllama(model="llama3.1:8b", temperature=0.3)

resilient_llm = primary.with_fallbacks([fallback])

result = await resilient_llm.ainvoke("Оцени эссе студента...")
```

Если Claude API недоступен (timeout, 500 error, rate limit), LangChain автоматически переключится на локальную Llama. Пользователь получит ответ, хотя, возможно, чуть менее качественный.

**Cost optimization по типу задачи:**

```python
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
cloud_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

async def smart_assess(student_work: str, task_type: str):
    if task_type in ("classification", "preprocessing", "spell_check"):
        return await local_llm.ainvoke(student_work)
    return await cloud_llm.ainvoke(student_work)
```

Простые задачи (классификация типа работы, проверка длины, предварительная фильтрация) отлично решаются локальной моделью. Финальная оценка с развёрнутым feedback — облачной.

**Fallback с structured output:**

```python
from pydantic import BaseModel, Field

class AssessmentResult(BaseModel):
    score: int = Field(description="Score 0-100")
    feedback: str = Field(description="Detailed feedback")

primary = ChatAnthropic(model="claude-sonnet-4-20250514").with_structured_output(AssessmentResult)
fallback = ChatOllama(model="llama3.1:8b").with_structured_output(AssessmentResult)

chain = primary.with_fallbacks([fallback])
```

Важно: `.with_structured_output()` нужно вызвать **на обоих моделях**. Иначе fallback вернёт `AIMessage` вместо `AssessmentResult`.

**Выбор модели через переменную окружения:**

```python
import os
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0.3,
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=0.3,
        )
    raise ValueError(f"Unknown provider: {provider}")
```

Это позволяет переключать модель без изменения кода: `LLM_PROVIDER=ollama python main.py` для разработки, `LLM_PROVIDER=anthropic python main.py` для production.

---

## Справочник API

### ChatOllama

```python
from langchain_ollama import ChatOllama
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `model` | `str` | обязательный | Имя модели в Ollama (`"llama3.1:8b"`, `"mistral"`) |
| `temperature` | `float` | `0.8` | Случайность генерации. 0.0 = детерминированный, 1.0 = максимум |
| `base_url` | `str` | `"http://localhost:11434"` | URL Ollama сервера |
| `num_ctx` | `int` | `2048` | Размер контекстного окна (токены). Больше = больше RAM |
| `num_predict` | `int` | `-1` | Макс. токенов в ответе. -1 = без лимита |
| `format` | `str \| None` | `None` | `"json"` для JSON mode |
| `keep_alive` | `str` | `"5m"` | Время жизни модели в RAM. `"-1"` = вечно |
| `top_p` | `float` | `0.9` | Nucleus sampling — отсечение по кумулятивной вероятности |
| `top_k` | `int` | `40` | Top-K sampling — сколько топ-токенов рассматривать |
| `repeat_penalty` | `float` | `1.1` | Штраф за повтор токенов. 1.0 = без штрафа |
| `stop` | `list[str] \| None` | `None` | Стоп-последовательности (генерация прекращается) |

Методы (Runnable interface):

| Метод | Описание |
|-------|----------|
| `invoke(input)` | Синхронный вызов |
| `ainvoke(input)` | Асинхронный вызов |
| `stream(input)` | Синхронный стриминг |
| `astream(input)` | Асинхронный стриминг |
| `batch(inputs)` | Параллельный вызов |
| `with_structured_output(schema)` | Принуждение к Pydantic-схеме |
| `with_fallbacks(fallbacks)` | Добавление fallback-моделей |
| `bind(**kwargs)` | Привязка параметров (stop, tools и т.д.) |

### OllamaEmbeddings

```python
from langchain_ollama import OllamaEmbeddings
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `model` | `str` | обязательный | Имя embedding-модели (`"nomic-embed-text"`) |
| `base_url` | `str` | `"http://localhost:11434"` | URL Ollama сервера |

Методы:

| Метод | Описание |
|-------|----------|
| `embed_query(text)` | Получить вектор для одного текста |
| `aembed_query(text)` | Асинхронный вариант |
| `embed_documents(texts)` | Получить векторы для списка текстов |
| `aembed_documents(texts)` | Асинхронный вариант |

### HuggingFaceEmbeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `model_name` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model ID |
| `model_kwargs` | `dict` | `{}` | Аргументы для model.load (`{"device": "cpu"}`) |
| `encode_kwargs` | `dict` | `{}` | Аргументы для model.encode (`{"normalize_embeddings": True}`) |
| `cache_folder` | `str \| None` | `None` | Папка для кэша моделей |

Методы: те же, что у `OllamaEmbeddings`.

### ollama Python client

```python
import ollama
```

| Функция | Описание | Пример |
|---------|----------|--------|
| `ollama.chat(model, messages)` | Chat completion | `ollama.chat("llama3.1", messages=[{"role": "user", "content": "Hi"}])` |
| `ollama.generate(model, prompt)` | Text generation | `ollama.generate("llama3.1", prompt="Hello")` |
| `ollama.embeddings(model, prompt)` | Embeddings | `ollama.embeddings("nomic-embed-text", prompt="text")` |
| `ollama.list()` | Список моделей | `ollama.list()` |
| `ollama.pull(model)` | Скачать модель | `ollama.pull("llama3.1:8b")` |
| `ollama.show(model)` | Информация о модели | `ollama.show("llama3.1")` |
| `ollama.ps()` | Запущенные модели | `ollama.ps()` |

### ChatOpenAI с custom base_url (для vLLM)

```python
from langchain_openai import ChatOpenAI
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `base_url` | `str` | `"https://api.openai.com/v1"` | URL vLLM сервера (`"http://localhost:8000/v1"`) |
| `api_key` | `str` | обязательный | Любая строка для vLLM (`"not-needed"`) |
| `model` | `str` | обязательный | ID модели на vLLM сервере |
| `temperature` | `float` | `0.7` | Случайность генерации |

### .with_fallbacks()

```python
primary_llm.with_fallbacks(
    fallbacks=[fallback_llm_1, fallback_llm_2],
    exceptions_to_handle=(Exception,),
)
```

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `fallbacks` | `list[Runnable]` | обязательный | Список fallback-моделей (по порядку) |
| `exceptions_to_handle` | `tuple[type]` | `(Exception,)` | Какие исключения ловить |

Порядок исполнения: primary → fallback_1 → fallback_2 → ... → Exception если все упали.

### Modelfile синтаксис

| Директива | Описание | Пример |
|-----------|----------|--------|
| `FROM` | Базовая модель | `FROM llama3.1:8b` |
| `PARAMETER` | Параметр модели | `PARAMETER temperature 0.3` |
| `SYSTEM` | System prompt | `SYSTEM "You are a teacher"` |
| `TEMPLATE` | Go template для форматирования промпта | `TEMPLATE "{{ .System }}\n{{ .Prompt }}"` |
| `ADAPTER` | LoRA adapter путь | `ADAPTER ./lora-weights.gguf` |
| `LICENSE` | Лицензия модели | `LICENSE "MIT"` |

---

## Практика: роутер `/api/v1/local`

Добавим в наш assessment-проект поддержку локальных моделей через Ollama. Четыре эндпоинта: список моделей, оценка локальной моделью, сравнение локальной и облачной, локальные embeddings.

### Шаг 1. Схемы — `app/schemas/local.py`

```python
from pydantic import BaseModel, Field


class OllamaModelInfo(BaseModel):
    name: str
    size: str
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class ModelsResponse(BaseModel):
    available: bool
    models: list[OllamaModelInfo] = []
    error: str | None = None


class LocalAssessRequest(BaseModel):
    student_work: str = Field(min_length=10, max_length=50000)
    rubric: str = Field(default="Evaluate the quality of the student work on a scale of 0-100. Provide detailed feedback.")
    model_name: str = Field(default="llama3.1:8b")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    num_ctx: int = Field(default=4096, ge=512, le=131072)


class LocalAssessResponse(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall score")
    feedback: str = Field(description="Detailed assessment feedback")
    model_used: str
    generation_time_ms: int


class CompareRequest(BaseModel):
    student_work: str = Field(min_length=10, max_length=50000)
    rubric: str = Field(default="Evaluate the quality of the student work on a scale of 0-100. Provide detailed feedback.")
    local_model: str = Field(default="llama3.1:8b")
    cloud_model: str = Field(default="claude-sonnet-4-20250514")


class SingleModelResult(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str
    model: str
    generation_time_ms: int


class CompareResponse(BaseModel):
    local_result: SingleModelResult
    cloud_result: SingleModelResult
    score_difference: int
    faster_model: str


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    model_name: str = Field(default="nomic-embed-text")


class EmbeddingResult(BaseModel):
    text: str
    vector_dim: int
    vector_preview: list[float] = Field(description="First 5 dimensions of the vector")


class EmbedResponse(BaseModel):
    results: list[EmbeddingResult]
    model_used: str
    total_texts: int
    dimension: int
```

Схемы следуют паттерну проекта: `Request` для входящих данных с валидацией через `Field`, `Response` для ответов с типизированными полями.

### Шаг 2. Сервис — `app/services/local_llm.py`

Сервис инкапсулирует работу с Ollama. Четыре функции: проверка доступности, получение модели, получение embedding-модели, список доступных моделей.

```python
import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings


OLLAMA_BASE_URL = "http://localhost:11434"


async def check_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def get_ollama_model(
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    num_ctx: int = 4096,
    **kwargs,
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=OLLAMA_BASE_URL,
        **kwargs,
    )


def get_ollama_embeddings(model_name: str = "nomic-embed-text") -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
    )


async def list_available_models() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return []
```

Ключевые решения:

- `check_ollama_available()` — асинхронная проверка через HTTP (не через CLI). Timeout 3 секунды — если Ollama не отвечает за 3 секунды, считаем недоступной.
- `get_ollama_model()` — фабрика для `ChatOllama`. Принимает `**kwargs` для гибкости (можно передать `format="json"`, `keep_alive` и т.д.).
- `get_ollama_embeddings()` — фабрика для `OllamaEmbeddings`.
- `list_available_models()` — вызывает Ollama REST API (`/api/tags`) напрямую, без LangChain. Возвращает список словарей с метаданными моделей.

### Шаг 3. Router — `app/api/v1/local.py`

```python
import time

from fastapi import APIRouter, HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.schemas.local import (
    CompareRequest,
    CompareResponse,
    EmbedRequest,
    EmbedResponse,
    EmbeddingResult,
    LocalAssessRequest,
    LocalAssessResponse,
    ModelsResponse,
    OllamaModelInfo,
    SingleModelResult,
)
from app.services.local_llm import (
    check_ollama_available,
    get_ollama_embeddings,
    get_ollama_model,
    list_available_models,
)

router = APIRouter(prefix="/local", tags=["local-llm"])

ASSESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert teacher evaluating student work.\n"
        "Rubric: {rubric}\n\n"
        "You MUST respond with a JSON object containing exactly two fields:\n"
        '- "score": integer from 0 to 100\n'
        '- "feedback": string with detailed assessment feedback',
    ),
    ("human", "Student work:\n\n{student_work}"),
])


class _AssessmentOutput(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str


@router.get("/models")
async def get_models() -> ModelsResponse:
    is_available = await check_ollama_available()
    if not is_available:
        return ModelsResponse(
            available=False,
            models=[],
            error="Ollama server is not running. Start it with: ollama serve",
        )

    raw_models = await list_available_models()
    models = [
        OllamaModelInfo(
            name=m.get("name", "unknown"),
            size=_format_size(m.get("size", 0)),
            family=m.get("details", {}).get("family"),
            parameter_size=m.get("details", {}).get("parameter_size"),
            quantization_level=m.get("details", {}).get("quantization_level"),
        )
        for m in raw_models
    ]
    return ModelsResponse(available=True, models=models)


@router.post("/assess")
async def assess_with_local(request: LocalAssessRequest) -> LocalAssessResponse:
    if not await check_ollama_available():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running. Start it with: ollama serve",
        )

    llm = get_ollama_model(
        model_name=request.model_name,
        temperature=request.temperature,
        num_ctx=request.num_ctx,
    )
    structured_llm = llm.with_structured_output(_AssessmentOutput)
    chain = ASSESS_PROMPT | structured_llm

    start = time.perf_counter()
    try:
        result = await chain.ainvoke({
            "student_work": request.student_work,
            "rubric": request.rubric,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama inference failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return LocalAssessResponse(
        score=result.score,
        feedback=result.feedback,
        model_used=request.model_name,
        generation_time_ms=elapsed_ms,
    )


@router.post("/compare")
async def compare_local_vs_cloud(request: CompareRequest) -> CompareResponse:
    if not await check_ollama_available():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running. Start it with: ollama serve",
        )

    local_llm = get_ollama_model(model_name=request.local_model, temperature=0.3)
    cloud_llm = ChatAnthropic(model=request.cloud_model, temperature=0.3)

    local_chain = ASSESS_PROMPT | local_llm.with_structured_output(_AssessmentOutput)
    cloud_chain = ASSESS_PROMPT | cloud_llm.with_structured_output(_AssessmentOutput)

    invoke_input = {
        "student_work": request.student_work,
        "rubric": request.rubric,
    }

    start_local = time.perf_counter()
    try:
        local_result = await local_chain.ainvoke(invoke_input)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Local model failed: {e}")
    local_ms = int((time.perf_counter() - start_local) * 1000)

    start_cloud = time.perf_counter()
    try:
        cloud_result = await cloud_chain.ainvoke(invoke_input)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cloud model failed: {e}")
    cloud_ms = int((time.perf_counter() - start_cloud) * 1000)

    return CompareResponse(
        local_result=SingleModelResult(
            score=local_result.score,
            feedback=local_result.feedback,
            model=request.local_model,
            generation_time_ms=local_ms,
        ),
        cloud_result=SingleModelResult(
            score=cloud_result.score,
            feedback=cloud_result.feedback,
            model=request.cloud_model,
            generation_time_ms=cloud_ms,
        ),
        score_difference=abs(local_result.score - cloud_result.score),
        faster_model=(
            request.local_model if local_ms < cloud_ms else request.cloud_model
        ),
    )


@router.post("/embed")
async def embed_texts(request: EmbedRequest) -> EmbedResponse:
    if not await check_ollama_available():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running. Start it with: ollama serve",
        )

    embeddings_model = get_ollama_embeddings(request.model_name)

    try:
        vectors = await embeddings_model.aembed_documents(request.texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    results = [
        EmbeddingResult(
            text=text[:100],
            vector_dim=len(vec),
            vector_preview=vec[:5],
        )
        for text, vec in zip(request.texts, vectors)
    ]

    return EmbedResponse(
        results=results,
        model_used=request.model_name,
        total_texts=len(request.texts),
        dimension=len(vectors[0]) if vectors else 0,
    )


def _format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "unknown"
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f} MB"
```

Разберём каждый эндпоинт:

**`GET /local/models`** — возвращает список моделей, скачанных в Ollama. Сначала проверяет доступность Ollama-сервера. Если сервер не запущен, возвращает `available: false` с подсказкой. Не бросает 503 — это информационный эндпоинт.

**`POST /local/assess`** — оценка студенческой работы локальной моделью. Использует `with_structured_output()` для получения типизированного ответа. Измеряет время генерации через `time.perf_counter()`. Если Ollama недоступна — 503. Если inference упал — 502.

**`POST /local/compare`** — сравнение оценки одной и той же работы локальной и облачной моделью. Запускает обе модели последовательно (не параллельно — чтобы корректно измерить время каждой). Возвращает результаты обеих, разницу в баллах и какая модель быстрее.

**`POST /local/embed`** — генерация embeddings локальной моделью. Принимает список текстов, возвращает для каждого размерность вектора и превью первых 5 значений. Полные векторы не возвращаем — они слишком большие для JSON-ответа.

### Шаг 4. Регистрация и тестирование

**Регистрация роутера** в `app/api/router.py`:

```python
from app.api.v1.local import router as local_router

api_router.include_router(local_router)
```

**Подготовка Ollama:**

```bash
ollama serve

ollama pull llama3.1:8b
ollama pull nomic-embed-text

ollama list
```

**Установка зависимостей:**

```bash
pip install langchain-ollama ollama
```

**Тестирование через curl:**

1. Проверка доступных моделей:

```bash
curl -s http://localhost:8000/api/v1/local/models | python -m json.tool
```

Ожидаемый ответ:

```json
{
    "available": true,
    "models": [
        {
            "name": "llama3.1:8b",
            "size": "4.7 GB",
            "family": "llama",
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0"
        },
        {
            "name": "nomic-embed-text:latest",
            "size": "274.0 MB",
            "family": "nomic-bert",
            "parameter_size": "137M",
            "quantization_level": "F16"
        }
    ],
    "error": null
}
```

2. Оценка локальной моделью:

```bash
curl -s -X POST http://localhost:8000/api/v1/local/assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The French Revolution began in 1789 with the storming of the Bastille. It was a period of radical political and societal change in France. The revolution led to the end of the monarchy and the rise of Napoleon Bonaparte.",
    "model_name": "llama3.1:8b",
    "temperature": 0.3
  }' | python -m json.tool
```

Ожидаемый ответ:

```json
{
    "score": 45,
    "feedback": "The student demonstrates basic knowledge of the French Revolution...",
    "model_used": "llama3.1:8b",
    "generation_time_ms": 3200
}
```

3. Сравнение локальной и облачной модели:

```bash
curl -s -X POST http://localhost:8000/api/v1/local/compare \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "The French Revolution began in 1789 with the storming of the Bastille. It was a period of radical political and societal change in France.",
    "local_model": "llama3.1:8b",
    "cloud_model": "claude-sonnet-4-20250514"
  }' | python -m json.tool
```

4. Генерация embeddings:

```bash
curl -s -X POST http://localhost:8000/api/v1/local/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Excellent essay with deep analysis",
      "Poor work, lacks understanding"
    ],
    "model_name": "nomic-embed-text"
  }' | python -m json.tool
```

Ожидаемый ответ:

```json
{
    "results": [
        {
            "text": "Excellent essay with deep analysis",
            "vector_dim": 768,
            "vector_preview": [0.0234, -0.0567, 0.1234, -0.0891, 0.0456]
        },
        {
            "text": "Poor work, lacks understanding",
            "vector_dim": 768,
            "vector_preview": [-0.0123, 0.0789, -0.0345, 0.0678, -0.0234]
        }
    ],
    "model_used": "nomic-embed-text",
    "total_texts": 2,
    "dimension": 768
}
```

**Тестирование fallback-поведения:**

Остановите Ollama и проверьте, что эндпоинты корректно возвращают ошибки:

```bash
ollama stop

curl -s http://localhost:8000/api/v1/local/models | python -m json.tool
```

Ожидаемый ответ:

```json
{
    "available": false,
    "models": [],
    "error": "Ollama server is not running. Start it with: ollama serve"
}
```

---

## Чеклист самопроверки

После выполнения всех шагов убедись, что можешь ответить «да» на каждый пункт:

- [ ] **Ollama установлена и работает.** `ollama --version` выводит версию, `curl http://localhost:11434` возвращает `Ollama is running`.

- [ ] **Модель скачана.** `ollama list` показывает хотя бы `llama3.1:8b` и `nomic-embed-text`.

- [ ] **GET `/api/v1/local/models`** возвращает список моделей с метаданными (name, size, family).

- [ ] **POST `/api/v1/local/assess`** принимает студенческую работу и возвращает `score` (0-100) и `feedback` от локальной модели.

- [ ] **POST `/api/v1/local/compare`** возвращает оценки от обеих моделей, `score_difference` и `faster_model`.

- [ ] **POST `/api/v1/local/embed`** возвращает размерность вектора и preview для каждого текста.

- [ ] **Обработка ошибок.** При остановленном Ollama: `/models` возвращает `available: false`, остальные эндпоинты — 503 с понятным сообщением.

- [ ] **ChatOllama в LCEL.** Умеешь построить цепочку `prompt | ChatOllama(...) | parser` и вызвать через `ainvoke()`.

- [ ] **Structured output с Ollama.** `ChatOllama(...).with_structured_output(MySchema)` возвращает Pydantic-объект.

- [ ] **Fallback-цепочка.** Можешь написать `primary.with_fallbacks([fallback])`, где primary — облачная модель, fallback — локальная.

---

## Частые ошибки

### 1. Ollama не запущена

```python
httpx.ConnectError: All connection attempts failed
```

Ollama — это сервер. Он должен работать в фоне, прежде чем отправлять запросы. На macOS если установлено через brew, Ollama запускается автоматически. Если нет:

```bash
ollama serve
```

Проверка:

```bash
curl http://localhost:11434
```

Если ответ `Ollama is running` — всё в порядке.

### 2. Модель не скачана

```python
ollama._types.ResponseError: model "llama3.1:8b" not found, try pulling it first
```

`ChatOllama(model="llama3.1:8b")` предполагает, что модель уже есть. Ollama не скачивает модели автоматически при API-вызове (в отличие от `ollama run`).

```bash
ollama pull llama3.1:8b
```

Всегда проверяй `ollama list` перед использованием модели в коде.

### 3. Нехватка памяти

```
llama_model_load: not enough memory
```

Модель не помещается в RAM/VRAM. Решения:

- Использовать модель меньшего размера: `llama3.1:8b` вместо `llama3.1:70b`
- Использовать более агрессивную квантизацию: `q3_K` вместо `q4_K_M`
- Уменьшить `num_ctx`: `ChatOllama(model="llama3.1:8b", num_ctx=2048)`
- Закрыть другие приложения, потребляющие память

```python
llm = ChatOllama(model="llama3.1:8b", num_ctx=2048)
```

Уменьшение `num_ctx` с 4096 до 2048 снижает потребление RAM на ~500MB для 8B модели.

### 4. Structured output не работает с некоторыми моделями

```python
OutputParserException: Failed to parse output
```

Не все модели поддерживают tool calling. Без tool calling `with_structured_output()` использует JSON mode, который менее надёжен. Решения:

```python
llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
structured_llm = llm.with_structured_output(_AssessmentOutput)
```

Явное указание `format="json"` повышает надёжность. Также помогает `temperature=0` — убирает случайность в формате ответа.

Если и это не помогает — используй `PydanticOutputParser` из Темы 3 вместо `with_structured_output()`.

### 5. Fallback без structured output на fallback-модели

```python
primary = ChatAnthropic(model="claude-sonnet-4-20250514").with_structured_output(Schema)
fallback = ChatOllama(model="llama3.1:8b")
chain = primary.with_fallbacks([fallback])
```

Ошибка: если primary падает, fallback вернёт `AIMessage` вместо `Schema`. Код, ожидающий `result.score`, упадёт с `AttributeError`.

```python
primary = ChatAnthropic(model="claude-sonnet-4-20250514").with_structured_output(Schema)
fallback = ChatOllama(model="llama3.1:8b").with_structured_output(Schema)
chain = primary.with_fallbacks([fallback])
```

Всегда применяй `.with_structured_output()` к обеим моделям в fallback-цепочке.

### 6. keep_alive не настроен — модель выгружается

```
time to first token: 15000ms
```

По умолчанию Ollama выгружает модель из памяти через 5 минут неактивности. При следующем запросе модель загружается заново — это 5-15 секунд. Для dev-сервера лучше держать модель в памяти:

```python
llm = ChatOllama(model="llama3.1:8b", keep_alive="-1")
```

`keep_alive="-1"` — модель остаётся в памяти навсегда (до остановки Ollama). Для production лучше `"30m"` или `"1h"`.

---

## Что читать дальше

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) — полное описание REST API
- [Ollama Model Library](https://ollama.com/library) — каталог всех доступных моделей
- [LangChain ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/) — интеграция с LangChain
- [LangChain OllamaEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/ollama/) — embeddings через Ollama
- [vLLM Documentation](https://docs.vllm.ai/) — production inference server
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — формат квантизированных моделей
- [Ollama Modelfile Reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) — кастомизация моделей
- [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — сравнение embedding-моделей

**Следующая тема:** [Тема 19: Vector Databases](topic_19_vector_databases.md)

# Тема 12: Multimodal AI — работа с изображениями

> **Пререквизиты:** [Тема 1-3](topic_01_prompt_engineering.md), рекомендуется [Тема 6 (LangGraph)](topic_06_langgraph_agents.md)
> **Что добавляем в проект:** `app/api/v1/multimodal.py`, `app/services/vision.py`, `app/schemas/multimodal.py`
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `langchain-openai`, `pillow`, `httpx`

---

## Теория

### 1. Зачем мультимодальность

До 2023 года LLM были исключительно текстовыми: на вход — текст, на выход — текст. Это фундаментальное ограничение: реальный мир не состоит только из текста. Студенческие работы — это рукописные тетради, диаграммы, скриншоты кода, фотографии проектов.

Мультимодальные модели принимают на вход **несколько типов данных** (modalities) в одном запросе:

| Модальность | Примеры | Поддержка |
|---|---|---|
| Текст | Промпты, инструкции, контекст | Все модели |
| Изображения | Фото, скриншоты, диаграммы, сканы | Claude 3+, GPT-4V/4o, Gemini |
| Аудио | Речь, звуковые записи | GPT-4o, Gemini |
| Видео | Записи экрана, видеоответы | Gemini |

Для нашего проекта (AI Assessment System) мультимодальность открывает новые сценарии:

- **Анализ рукописных работ** — фото тетради → оценка почерка, содержания, аккуратности
- **Оценка диаграмм** — скриншот UML/ER/блок-схемы → проверка корректности
- **Проверка скриншотов кода** — когда студент отправляет фото экрана вместо текста
- **Сравнение с образцом** — работа студента + эталонный пример → сравнительная оценка
- **Анализ визуальных проектов** — дизайн-макеты, графики, инфографика

Какие модели поддерживают vision:

| Модель | Vision | Max изображений | Max размер | Особенности |
|---|---|---|---|---|
| Claude 3 Haiku | Да | 20 | 20 MB | Быстрая, дешёвая |
| Claude 3.5 Sonnet | Да | 20 | 20 MB | Лучший баланс цена/качество |
| Claude 4 Sonnet | Да | 20 | 20 MB | Топовое качество анализа |
| Claude 4 Opus | Да | 20 | 20 MB | Максимальные возможности |
| GPT-4V | Да | Без лимита | 20 MB | Первая массовая vision LLM |
| GPT-4o | Да | Без лимита | 20 MB | Быстрее GPT-4V, дешевле |
| Gemini 1.5 Pro | Да | Без лимита | 20 MB | + видео и аудио |

Важные ограничения vision-моделей:

- **Не OCR-движок** — модель не даёт pixel-perfect распознавание текста. Для точного OCR лучше Tesseract + text LLM.
- **Не геометрический калькулятор** — модель плохо измеряет углы, расстояния, пропорции.
- **Не детектор объектов** — модель не даёт bounding boxes, не считает объекты точно.
- **Не стерео-зрение** — нет понимания глубины, 3D-расположения.

Vision LLM — это **семантический анализатор**: она понимает *смысл* изображения, но не его точные пиксельные характеристики.

### 2. Формат мультимодальных сообщений в LangChain

В обычном текстовом запросе `HumanMessage.content` — это строка. В мультимодальном запросе `content` становится **списком блоков**, каждый из которых описывает один элемент (текст или изображение):

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Опиши что на изображении"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        },
    ]
)
```

Типы блоков:

| Тип блока | Формат | Описание |
|---|---|---|
| `text` | `{"type": "text", "text": "..."}` | Текстовый блок — промпт, инструкции |
| `image_url` | `{"type": "image_url", "image_url": {"url": "..."}}` | Изображение по URL или base64 |

Нативный формат Anthropic API отличается от OpenAI:

```python
anthropic_format = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "<base64-encoded-data>",
    },
}

openai_format = {
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,<base64-encoded-data>",
    },
}
```

LangChain предоставляет **унифицированную абстракцию**: формат `image_url` с data URI (`data:image/png;base64,...`) работает одинаково для всех провайдеров. LangChain автоматически конвертирует его в нативный формат конкретного API при отправке.

Это означает, что один и тот же код работает и с `ChatAnthropic`, и с `ChatOpenAI`:

```python
from langchain_core.messages import HumanMessage

image_message = HumanMessage(
    content=[
        {"type": "text", "text": "Оцени эту работу"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_data}",
            },
        },
    ]
)

result_anthropic = await anthropic_llm.ainvoke([image_message])
result_openai = await openai_llm.ainvoke([image_message])
```

### 3. Base64 vs URL — два способа передать изображение

Существуют два способа передать изображение в LLM: по URL и через Base64-кодирование.

**Способ 1: URL**

```python
content_block = {
    "type": "image_url",
    "image_url": {"url": "https://example.com/student-work.jpg"},
}
```

Модель (или API-провайдер) сама загружает изображение по URL. Плюсы: простота, нет нагрузки на сериализацию. Минусы: URL должен быть публично доступен, провайдер может кешировать или не загрузить изображение, нет контроля над тем, что именно модель получила.

**Способ 2: Base64**

```python
import base64

with open("student-work.jpg", "rb") as f:
    image_bytes = f.read()

base64_data = base64.b64encode(image_bytes).decode("utf-8")

content_block = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{base64_data}",
    },
}
```

Изображение кодируется в Base64 и передаётся inline в запросе. Плюсы: полный контроль над данными, работает с приватными изображениями, детерминированность. Минусы: увеличивает размер запроса на ~33% (Base64 overhead), требует кодирования.

Сравнение подходов:

| Критерий | URL | Base64 |
|---|---|---|
| Простота | Высокая | Средняя |
| Контроль данных | Нет | Полный |
| Приватные изображения | Нет | Да |
| Размер запроса | Минимальный | +33% |
| Надёжность | Зависит от URL | Максимальная |
| Подходит для | Публичные изображения, прототипы | Загруженные файлы, production |

Для нашего проекта мы используем Base64: студенты загружают файлы через API, мы контролируем данные полностью.

Поддерживаемые форматы: **JPEG**, **PNG**, **GIF**, **WebP**. JPEG оптимален для фотографий (компрессия, малый размер). PNG — для скриншотов и диаграмм (без потерь, чёткие линии). GIF и WebP поддерживаются, но реже используются.

Ограничения по размеру:

| Провайдер | Max размер файла | Max количество изображений |
|---|---|---|
| Anthropic | 20 MB | 20 на запрос |
| OpenAI | 20 MB | Без жёсткого лимита |
| Google (Gemini) | 20 MB | Без жёсткого лимита |

На практике отправка изображений >5 MB увеличивает latency. Рекомендуется уменьшать изображения до 1-2 MB перед отправкой.

### 4. Vision + Structured Output

Одна из мощнейших комбинаций — vision + `with_structured_output()`. Модель анализирует изображение и возвращает **типизированный результат**, который можно напрямую использовать в коде.

Пример: извлечь из скана рукописной работы структурированную оценку:

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

class HandwritingAnalysis(BaseModel):
    recognized_text: str = Field(description="Распознанный текст с изображения")
    legibility_score: int = Field(ge=1, le=10, description="Разборчивость почерка 1-10")
    content_summary: str = Field(description="Краткое содержание работы")
    language: str = Field(description="Язык текста")

structured_llm = llm.with_structured_output(HandwritingAnalysis)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Проанализируй рукописную работу на изображении. "
                    "Распознай текст, оцени разборчивость, определи язык.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
        },
    ]
)

result = await structured_llm.ainvoke([message])
print(result.recognized_text)
print(result.legibility_score)
```

Механика: `with_structured_output()` работает с vision точно так же, как с текстом — через tool calling API. JSON Schema из Pydantic-модели передаётся как tool definition, и constrained decoding гарантирует корректный формат ответа. Мультимодальный контент в `HumanMessage` обрабатывается на уровне модели, а structured output — на уровне API.

Промпт-инжиниринг для vision отличается от текстового:

| Аспект | Текстовый промпт | Vision промпт |
|---|---|---|
| Что описывать | Задачу и формат | Задачу, формат и *что именно искать* на изображении |
| Уровень детализации | Средний | Высокий — модель не знает, на что обращать внимание |
| Примеры | Few-shot текстовые | Описательные, с конкретными визуальными признаками |
| Ограничения | Стандартные | + предупреждение о неточности OCR |

Эффективный vision-промпт:

```
Перед тобой фотография рукописной работы студента.
1. Внимательно прочитай весь текст на изображении
2. Обрати внимание на зачёркивания и исправления
3. Если текст нечитаем — укажи [неразборчиво] вместо угадывания
4. Оцени аккуратность: ровность строк, размер букв, отступы
```

Ограничения vision + structured output:

- Сложные таблицы с мелким текстом — ошибки в 20-40% ячеек
- Рукописный почерк — зависит от разборчивости, ошибки от 5% до 50%
- Мелкий текст (<12px на экране) — часто нераспознаваем
- Математические формулы — часто с ошибками в индексах и степенях

### 5. Multi-image input

Одно сообщение может содержать **несколько изображений**. Это открывает сценарии, невозможные с одним изображением:

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "Сравни работу студента с эталоном"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{student_work_b64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{reference_b64}"},
        },
    ]
)
```

Порядок изображений **имеет значение**. Модель обрабатывает контент слева направо (в порядке блоков в списке `content`). Если промпт говорит «сравни первое изображение со вторым», нужно убедиться, что порядок соответствует описанию.

Сценарии multi-image в нашем проекте:

| Сценарий | Изображения | Что получаем |
|---|---|---|
| Многостраничная работа | 3-5 страниц тетради | Единый анализ всех страниц |
| Сравнение с образцом | Работа + эталон | Сравнительная оценка |
| До/после | Черновик + чистовик | Анализ прогресса |
| Диаграмма + описание | Схема + скриншот кода | Проверка соответствия |

Пример: 3 страницы работы → единый анализ:

```python
pages = [page1_b64, page2_b64, page3_b64]

content_blocks = [
    {
        "type": "text",
        "text": "Перед тобой 3 страницы рукописной работы студента. "
                "Проанализируй все страницы как единое произведение. "
                "Страница 1 — введение, страница 2 — основная часть, "
                "страница 3 — заключение.",
    },
]

for i, page_b64 in enumerate(pages, 1):
    content_blocks.append({
        "type": "text",
        "text": f"--- Страница {i} ---",
    })
    content_blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{page_b64}"},
    })

message = HumanMessage(content=content_blocks)
```

Token cost для изображений зависит от размера и уровня detail:

| Размер изображения | detail: low | detail: high |
|---|---|---|
| 512×512 | 85 токенов | ~765 токенов |
| 1024×1024 | 85 токенов | ~1105 токенов |
| 2048×2048 | 85 токенов | ~1745 токенов |
| 4096×4096 | 85 токенов | ~1745 токенов (resize до 2048) |

При 5 изображениях с `detail: high` — это 5000-8000 дополнительных input-токенов. При стоимости $3/M input tokens (Claude Sonnet) — ~$0.015-0.024 за один запрос только на изображения.

### 6. Detail level и оптимизация

Параметр `detail` управляет тем, сколько токенов модель «тратит» на анализ изображения:

```python
content_block = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/png;base64,{base64_data}",
        "detail": "low",
    },
}
```

| Значение | Токенов | Скорость | Когда использовать |
|---|---|---|---|
| `auto` | Зависит от размера | — | По умолчанию, модель решает сама |
| `low` | 85 | Быстро | Общее понимание, классификация, крупный текст |
| `high` | 765-1745+ | Медленно | Мелкий текст, детали диаграмм, числа |

Механика `low`: изображение уменьшается до 512×512 и обрабатывается как единый тайл. Модель получает «общий смысл» без мелких деталей.

Механика `high`: изображение разбивается на тайлы 512×512, каждый обрабатывается отдельно, плюс уменьшенная копия целого изображения для контекста. Больше тайлов = больше токенов = больше деталей.

Стратегия двухпроходного анализа:

```python
low_result = await analyze(image, detail="low", prompt="Что на изображении? Тип контента?")

if low_result.content_type == "handwriting":
    high_result = await analyze(image, detail="high", prompt="Распознай весь текст подробно")
elif low_result.content_type == "diagram":
    high_result = await analyze(image, detail="high", prompt="Опиши все элементы диаграммы")
else:
    high_result = low_result
```

Первый проход с `low` стоит 85 токенов и классифицирует контент. Второй проход с `high` стоит 765-1745 токенов, но выполняется только когда нужна детализация. Экономия: если 60% изображений не требуют `high`, средняя стоимость падает на ~40%.

Preprocessing изображений перед отправкой:

```python
from PIL import Image
import io
import base64

def preprocess_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    image = Image.open(io.BytesIO(image_bytes))

    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
```

Уменьшение изображения до 1024px — разумный компромисс: достаточно для анализа текста и диаграмм, но в 4-16 раз меньше оригинала по размеру файла.

### 7. Мультимодальные chains и agents

Vision можно встроить в LCEL chain точно так же, как текстовый LLM:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def build_vision_chain(llm, response_model):
    structured = llm.with_structured_output(response_model)

    async def process(inputs: dict):
        message = HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                {
                    "type": "image_url",
                    "image_url": {"url": inputs["image_url"]},
                },
            ]
        )
        return await structured.ainvoke([message])

    return process
```

Для более сложных сценариев — vision как tool в агенте:

```python
from langchain_core.tools import tool

@tool
def analyze_image(image_base64: str, analysis_prompt: str) -> str:
    """Анализирует изображение и возвращает текстовое описание."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": analysis_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            },
        ]
    )
    result = vision_llm.invoke([message])
    return result.content
```

Комбинированный сценарий для нашего проекта:

```
Вход: текст эссе + фотографии диаграмм

1. Агент анализирует текст (обычный текстовый LLM)
2. Агент видит, что есть приложенные изображения
3. Агент вызывает vision tool для анализа каждой диаграммы
4. Агент комбинирует текстовую и визуальную оценку
5. Результат: единая структурированная оценка
```

Архитектурно это реализуется через LangGraph-граф, где vision-анализ — отдельный узел. Но для нашего роутера достаточно прямого вызова vision LLM без полноценного агента.

### 8. Ограничения и подводные камни

Vision-модели — мощный инструмент, но с серьёзными ограничениями. Понимание этих ограничений критично для production-систем.

**Hallucinations (галлюцинации)**

Модель может «увидеть» на изображении то, чего нет. Примеры:
- Добавить слова в распознанный рукописный текст
- Описать несуществующие элементы диаграммы
- Неправильно прочитать числа (7 → 1, 6 → 8)

Стратегия: всегда просить модель указывать уровень уверенности. Добавлять в промпт: «Если не уверен — напиши [неуверен]».

**Мелкий текст**

Текст размером менее ~12px (на оригинальном изображении, до resize) часто нераспознаваем. При `detail: low` порог ещё выше — ~20px. Для скриншотов с мелким кодом `detail: high` обязателен.

**Числа и подсчёт**

Модель не умеет точно считать объекты. «Сколько кружков на рисунке?» — ответ может отличаться на ±2-3 от правильного. Для точного подсчёта используй computer vision (OpenCV) + text LLM.

**Пространственные отношения**

«Что находится слева от красного квадрата?» — модель может ответить неправильно. Пространственные отношения («выше», «ниже», «левее», «внутри») работают приблизительно, не точно.

**Генерация текста на изображениях**

Это не относится к vision (анализу), но важно помнить: когда модель генерирует изображения (DALL-E, Midjourney), текст на них часто содержит ошибки. Vision-модели анализируют, но не генерируют изображения.

**Когда лучше использовать OCR + text LLM:**

| Задача | Vision LLM | OCR + text LLM |
|---|---|---|
| Распознание рукописного текста | Приемлемо (80-95%) | Лучше (Tesseract + LLM) |
| Табличные данные | Плохо (60-80%) | Лучше (специализированный OCR) |
| Мелкий печатный текст | Плохо с low, средне с high | Хорошо (Tesseract оптимизирован) |
| Семантический анализ изображения | Отлично | Невозможно |
| Описание содержания | Отлично | Невозможно |
| Сравнительный анализ | Отлично | Очень сложно |

Правило: если задача сводится к «извлеки текст точно» — используй OCR. Если задача «пойми смысл» — используй vision LLM. Часто лучший результат даёт комбинация: OCR для извлечения, LLM для анализа.

---

## Справочник API

### HumanMessage (multimodal content format)

**Описание:** Сообщение с ролью `human` в chat API. Для мультимодальных запросов `content` принимает не строку, а список блоков, каждый из которых описывает текст или изображение. LangChain автоматически конвертирует unified формат в нативный формат провайдера (Anthropic, OpenAI).

```python
HumanMessage(
    content: str | list[dict],
    additional_kwargs: dict = {},
    response_metadata: dict = {},
)
```

**Форматы блоков content:**

| Тип блока | Формат | Описание |
|---|---|---|
| Текст | `{"type": "text", "text": "..."}` | Текстовая часть сообщения |
| Изображение (URL) | `{"type": "image_url", "image_url": {"url": "https://..."}}` | Изображение по публичному URL |
| Изображение (Base64) | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` | Изображение inline |
| Изображение (detail) | `{"type": "image_url", "image_url": {"url": "...", "detail": "low"}}` | С указанием уровня детализации |

**Параметры image_url:**

| Параметр | Тип | Обязательный | Описание |
|---|---|---|---|
| `url` | `str` | Да | URL изображения или data URI с base64 |
| `detail` | `str` | Нет | `"auto"` (default), `"low"`, `"high"` — уровень детализации |

**Пример использования:**

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Ты — эксперт по оценке студенческих работ."),
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Оцени рукописную работу на изображении по критериям: содержание, почерк, аккуратность.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_data}",
                    "detail": "high",
                },
            },
        ]
    ),
]

result = await llm.ainvoke(messages)
```

---

### ChatAnthropic with vision

**Описание:** `ChatAnthropic` поддерживает vision без дополнительной конфигурации. Все модели Claude 3+ (Haiku, Sonnet, Opus) и Claude 4 (Sonnet, Opus) принимают изображения. Единственное требование — увеличить `max_tokens`, так как vision-ответы обычно длиннее текстовых.

```python
ChatAnthropic(
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 1.0,
    max_tokens: int = 1024,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int = 2,
)
```

**Параметры, важные для vision:**

| Параметр | Рекомендация для vision | Почему |
|---|---|---|
| `model` | `claude-sonnet-4-20250514` или `claude-4-opus-20250514` | Лучшее качество vision |
| `max_tokens` | 4096-8192 | Vision-ответы длиннее текстовых |
| `temperature` | 0.0-0.3 | Для оценки нужна воспроизводимость |

**Ограничения Anthropic vision:**

| Ограничение | Значение |
|---|---|
| Max изображений на запрос | 20 |
| Max размер одного изображения | 20 MB |
| Поддерживаемые форматы | JPEG, PNG, GIF, WebP |
| Max разрешение | Без лимита (resize автоматический) |

**Пример:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    temperature=0.0,
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Что изображено на картинке?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]
)

result = await llm.ainvoke([message])
print(result.content)
```

---

### ChatOpenAI with vision

**Описание:** `ChatOpenAI` с моделями GPT-4V и GPT-4o поддерживает vision. Формат сообщений идентичен LangChain unified — тот же `image_url` блок.

```python
ChatOpenAI(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int = 2,
)
```

**Модели с vision:**

| Модель | Vision | Стоимость (input/M) | Особенности |
|---|---|---|---|
| `gpt-4o` | Да | $2.50 | Быстрый, дешёвый, хорошее качество |
| `gpt-4o-mini` | Да | $0.15 | Самый дешёвый, приемлемое качество |
| `gpt-4-turbo` | Да | $10.00 | Legacy, не рекомендуется для новых проектов |

**Пример:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=4096,
    temperature=0.0,
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this diagram"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        },
    ]
)

result = await llm.ainvoke([message])
```

---

### base64 module

**Описание:** Стандартный модуль Python для кодирования/декодирования данных в Base64. В контексте vision — используется для конвертации бинарных данных изображений в строку, пригодную для передачи в JSON.

```python
import base64
```

**Основные функции:**

| Функция | Вход | Выход | Описание |
|---|---|---|---|
| `b64encode(data)` | `bytes` | `bytes` | Кодирует bytes → base64 bytes |
| `b64decode(data)` | `bytes \| str` | `bytes` | Декодирует base64 → bytes |

**Паттерны использования:**

```python
import base64

with open("image.jpg", "rb") as f:
    raw_bytes = f.read()

b64_bytes = base64.b64encode(raw_bytes)
b64_string = b64_bytes.decode("utf-8")

data_uri = f"data:image/jpeg;base64,{b64_string}"
```

**Формат data URI:**

```
data:<media_type>;base64,<data>
```

| Media type | Расширения |
|---|---|
| `image/jpeg` | .jpg, .jpeg |
| `image/png` | .png |
| `image/gif` | .gif |
| `image/webp` | .webp |

---

### with_structured_output() в мультимодальном контексте

**Описание:** Метод `with_structured_output()` работает с vision без изменений. Пydantic-модель передаётся как tool definition, мультимодальный контент — в сообщении. Constrained decoding применяется к ответу независимо от типа входных данных.

```python
structured_llm = llm.with_structured_output(
    schema: type[BaseModel] | dict,
    *,
    method: str = "function_calling",
    include_raw: bool = False,
)
```

**Ограничения в multimodal контексте:**

| Ограничение | Описание |
|---|---|
| Token budget | Vision tokens + structured output tokens должны вместиться в контекстное окно |
| Complexity | Чем сложнее Pydantic-схема, тем выше риск ошибок при vision input |
| Latency | Vision + structured output = ~2-5x latency обычного текстового запроса |

**Пример:**

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

class DiagramAnalysis(BaseModel):
    diagram_type: str = Field(description="UML, ER, flowchart, etc.")
    elements: list[str] = Field(description="List of identified elements")
    relationships: list[str] = Field(description="Relationships between elements")
    correctness_score: int = Field(ge=0, le=100, description="How correct is the diagram")
    issues: list[str] = Field(description="Found issues and errors")

structured_llm = llm.with_structured_output(DiagramAnalysis)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Проанализируй эту диаграмму. Определи тип, элементы, связи. Оцени корректность.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
)

result = await structured_llm.ainvoke([message])
```

---

### Pillow (PIL)

**Описание:** Библиотека для обработки изображений. В контексте vision используется для preprocessing: resize, конвертация формата, проверка валидности, получение метаданных.

```python
from PIL import Image
```

**Основные методы:**

| Метод | Описание |
|---|---|
| `Image.open(fp)` | Открыть изображение из файла или BytesIO |
| `image.resize((w, h))` | Изменить размер (exact) |
| `image.thumbnail((w, h))` | Уменьшить с сохранением пропорций |
| `image.convert(mode)` | Конвертировать цветовое пространство (RGB, RGBA, L) |
| `image.save(fp, format)` | Сохранить в файл или BytesIO |
| `image.size` | Кортеж `(width, height)` |
| `image.format` | Формат файла (JPEG, PNG, etc.) |
| `image.mode` | Цветовое пространство (RGB, RGBA, L) |

**Пример preprocessing для vision:**

```python
from PIL import Image
import io
import base64

def preprocess_for_vision(
    image_bytes: bytes,
    max_dimension: int = 1536,
    quality: int = 85,
) -> tuple[str, str]:
    image = Image.open(io.BytesIO(image_bytes))

    original_format = image.format or "JPEG"

    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded, "image/jpeg"
```

---

### httpx (AsyncClient)

**Описание:** Асинхронный HTTP-клиент для загрузки изображений по URL. В отличие от `requests`, httpx поддерживает `async/await` и лучше подходит для FastAPI-приложений.

```python
import httpx
```

**Основные методы AsyncClient:**

| Метод | Описание |
|---|---|
| `async with httpx.AsyncClient() as client:` | Контекстный менеджер для клиента |
| `await client.get(url)` | GET-запрос |
| `response.content` | Бинарное содержимое ответа (bytes) |
| `response.headers["content-type"]` | MIME-тип ответа |
| `response.raise_for_status()` | Бросить исключение при HTTP-ошибке |

**Пример загрузки изображения:**

```python
import httpx
import base64

async def fetch_image_as_base64(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "image/jpeg")
        media_type = content_type.split(";")[0].strip()

        encoded = base64.b64encode(response.content).decode("utf-8")
        return encoded, media_type
```

---

## Практика: роутер `/api/v1/multimodal`

### Шаг 1. Схемы — `app/schemas/multimodal.py`

Создаём Pydantic-модели для запросов и ответов всех эндпоинтов. Каждая схема ответа — это structured output, который vision LLM будет генерировать через `with_structured_output()`.

```python
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data (without data URI prefix)",
    )
    image_url: str | None = Field(
        default=None,
        description="Public URL of the image to analyze",
    )
    media_type: str = Field(
        default="image/jpeg",
        description="MIME type of the image: image/jpeg, image/png, image/gif, image/webp",
    )


class ImageAnalysisRequest(ImageInput):
    prompt: str = Field(description="What to analyze in the image")
    detail: str = Field(
        default="auto",
        description="Vision detail level: auto, low, high",
    )


class ImageAnalysisResult(BaseModel):
    description: str = Field(description="Detailed description of the image content")
    key_elements: list[str] = Field(description="Key visual elements identified")
    text_content: str | None = Field(
        default=None,
        description="Any text found in the image, None if no text present",
    )
    content_type: str = Field(
        description="Type of content: photo, screenshot, diagram, handwriting, chart, other",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the analysis 0.0-1.0",
    )


class ImageAnalysisResponse(BaseModel):
    analysis: ImageAnalysisResult
    model: str
    detail_level: str


class HandwritingOCRRequest(ImageInput):
    language_hint: str = Field(
        default="auto",
        description="Expected language: auto, en, ru, etc.",
    )


class HandwritingOCRResult(BaseModel):
    recognized_text: str = Field(description="Full recognized text from the image")
    legibility_score: int = Field(
        ge=1, le=10,
        description="Handwriting legibility: 1=illegible, 10=perfectly clear",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in text recognition accuracy",
    )
    language: str = Field(description="Detected language of the text")
    uncertain_fragments: list[str] = Field(
        default_factory=list,
        description="Fragments where recognition is uncertain, marked with [?]",
    )


class HandwritingOCRResponse(BaseModel):
    ocr: HandwritingOCRResult
    model: str


class DiagramAnalysisRequest(ImageInput):
    diagram_type_hint: str | None = Field(
        default=None,
        description="Expected diagram type: UML, ER, flowchart, etc.",
    )
    grading_criteria: str = Field(
        default="correctness, completeness, clarity, notation",
        description="Criteria to grade the diagram on",
    )


class DiagramElement(BaseModel):
    name: str = Field(description="Element name or label")
    element_type: str = Field(description="Type: class, entity, process, decision, etc.")


class DiagramRelationship(BaseModel):
    source: str = Field(description="Source element name")
    target: str = Field(description="Target element name")
    relationship_type: str = Field(
        description="Type: association, inheritance, dependency, flow, etc.",
    )
    label: str | None = Field(default=None, description="Relationship label if present")


class DiagramAnalysisResult(BaseModel):
    diagram_type: str = Field(description="Detected diagram type: UML class, ER, flowchart, sequence, etc.")
    elements: list[DiagramElement] = Field(description="All identified elements")
    relationships: list[DiagramRelationship] = Field(description="All identified relationships")
    correctness_score: int = Field(ge=0, le=100, description="Correctness of notation and semantics")
    completeness_score: int = Field(ge=0, le=100, description="How complete the diagram is")
    clarity_score: int = Field(ge=0, le=100, description="Visual clarity and readability")
    overall_score: int = Field(ge=0, le=100, description="Overall diagram quality score")
    issues: list[str] = Field(description="Found issues: incorrect notation, missing elements, etc.")
    suggestions: list[str] = Field(description="Improvement suggestions")


class DiagramAnalysisResponse(BaseModel):
    analysis: DiagramAnalysisResult
    model: str


class MultiImageCompareRequest(BaseModel):
    student_image: ImageInput
    reference_image: ImageInput
    comparison_prompt: str = Field(
        default="Compare the student work with the reference example. "
                "Evaluate accuracy, completeness, and quality.",
        description="Instructions for comparison",
    )


class ComparisonResult(BaseModel):
    similarity_score: int = Field(
        ge=0, le=100,
        description="How similar the student work is to the reference 0-100",
    )
    accuracy_score: int = Field(
        ge=0, le=100,
        description="Accuracy of the student work compared to reference",
    )
    completeness_score: int = Field(
        ge=0, le=100,
        description="How complete the student work is vs reference",
    )
    overall_score: int = Field(
        ge=0, le=100,
        description="Overall quality score",
    )
    matching_elements: list[str] = Field(
        description="Elements present in both student work and reference",
    )
    missing_elements: list[str] = Field(
        description="Elements in reference but missing in student work",
    )
    extra_elements: list[str] = Field(
        description="Elements in student work but not in reference",
    )
    feedback: str = Field(description="Detailed comparison feedback, 3-5 sentences")
    strengths: list[str] = Field(description="What the student did well")
    improvements: list[str] = Field(description="What needs improvement")


class MultiImageCompareResponse(BaseModel):
    comparison: ComparisonResult
    model: str
```

Обратите внимание: `ImageInput` — базовый класс, который принимает либо `image_base64`, либо `image_url`. Это позволяет эндпоинтам работать с обоими способами передачи изображений. Валидация (хотя бы одно из двух) выполняется в сервисе.

### Шаг 2. Сервис — `app/services/vision.py`

Сервисный слой инкапсулирует работу с изображениями: кодирование, загрузка, построение сообщений, вызов LLM.

```python
import base64
import io

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel

from app.config import Settings


VISION_SYSTEM_PROMPT = (
    "You are an expert visual analyst specializing in educational content assessment. "
    "Analyze images carefully and provide detailed, accurate observations. "
    "If text is present, read it carefully. If you are uncertain about any element, "
    "indicate your uncertainty rather than guessing. "
    "Focus on objective analysis over subjective interpretation."
)

SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024


def encode_image_file(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    image = Image.open(io.BytesIO(image_bytes))
    fmt = (image.format or "JPEG").upper()

    media_type_map = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }
    media_type = media_type_map.get(fmt, "image/jpeg")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return encoded, media_type


async def fetch_image(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10),
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "image/jpeg")
        media_type = content_type.split(";")[0].strip()

        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported media type: {media_type}")

        if len(response.content) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image too large: {len(response.content)} bytes "
                f"(max {MAX_IMAGE_SIZE_BYTES})"
            )

        encoded = base64.b64encode(response.content).decode("utf-8")
        return encoded, media_type


def preprocess_image(
    image_bytes: bytes,
    max_dimension: int = 1536,
    quality: int = 85,
) -> tuple[str, str]:
    image = Image.open(io.BytesIO(image_bytes))

    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1] if "A" in image.mode else None)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, "image/jpeg"


def build_vision_message(
    text: str,
    images: list[tuple[str, str]],
    detail: str = "auto",
) -> HumanMessage:
    content: list[dict] = [{"type": "text", "text": text}]

    for image_b64, media_type in images:
        data_uri = f"data:{media_type};base64,{image_b64}"
        block: dict = {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }
        if detail != "auto":
            block["image_url"]["detail"] = detail
        content.append(block)

    return HumanMessage(content=content)


async def resolve_image(
    image_base64: str | None,
    image_url: str | None,
    media_type: str = "image/jpeg",
) -> tuple[str, str]:
    if image_base64:
        return image_base64, media_type
    if image_url:
        return await fetch_image(image_url)
    raise ValueError("Either image_base64 or image_url must be provided")


async def analyze_image(
    llm: ChatAnthropic,
    images: list[tuple[str, str]],
    prompt: str,
    response_model: type[BaseModel],
    detail: str = "auto",
    system_prompt: str = VISION_SYSTEM_PROMPT,
) -> BaseModel:
    structured_llm = llm.with_structured_output(response_model)
    human_message = build_vision_message(prompt, images, detail=detail)
    messages = [SystemMessage(content=system_prompt), human_message]
    return await structured_llm.ainvoke(messages)
```

Разберём ключевые функции:

- `encode_image_file` — читает файл с диска, определяет формат через Pillow, кодирует в base64. Возвращает кортеж `(base64_data, media_type)`.
- `fetch_image` — асинхронно загружает изображение по URL через httpx. Проверяет media type и размер. Возвращает тот же кортеж.
- `preprocess_image` — уменьшает изображение до `max_dimension`, конвертирует RGBA→RGB (для JPEG), сжимает с заданным quality. Уменьшает размер в 4-16 раз.
- `build_vision_message` — собирает `HumanMessage` из текста и списка изображений. Формирует data URI для каждого изображения. Поддерживает `detail` параметр.
- `resolve_image` — единая точка входа: принимает либо base64, либо URL, возвращает base64. Используется в роутере для унификации.
- `analyze_image` — центральная функция: принимает LLM, изображения, промпт и Pydantic-модель ответа. Оборачивает LLM в `with_structured_output()`, собирает messages, вызывает `ainvoke`. Возвращает типизированный результат.

### Шаг 3. Router — `app/api/v1/multimodal.py`

Роутер с 4 эндпоинтами. Каждый эндпоинт демонстрирует отдельный аспект vision API.

```python
from fastapi import APIRouter, HTTPException

from app.dependencies import LLMDep, SettingsDep
from app.schemas.multimodal import (
    ComparisonResult,
    DiagramAnalysisRequest,
    DiagramAnalysisResponse,
    DiagramAnalysisResult,
    HandwritingOCRRequest,
    HandwritingOCRResponse,
    HandwritingOCRResult,
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    ImageAnalysisResult,
    MultiImageCompareRequest,
    MultiImageCompareResponse,
)
from app.services.vision import analyze_image, resolve_image

router = APIRouter(prefix="/multimodal", tags=["lesson-12-multimodal"])


@router.post("/analyze")
async def analyze(
    request: ImageAnalysisRequest,
    llm: LLMDep,
    settings: SettingsDep,
) -> ImageAnalysisResponse:
    try:
        image_b64, media_type = await resolve_image(
            request.image_base64, request.image_url, request.media_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt = (
        f"{request.prompt}\n\n"
        "Provide a detailed analysis of the image. Identify all key visual elements. "
        "If there is text in the image, read and include it. "
        "Determine the content type (photo, screenshot, diagram, handwriting, chart, other)."
    )

    try:
        result = await analyze_image(
            llm=llm,
            images=[(image_b64, media_type)],
            prompt=prompt,
            response_model=ImageAnalysisResult,
            detail=request.detail,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vision analysis failed: {e}")

    return ImageAnalysisResponse(
        analysis=result,
        model=settings.model_name,
        detail_level=request.detail,
    )


@router.post("/ocr")
async def ocr_handwriting(
    request: HandwritingOCRRequest,
    llm: LLMDep,
    settings: SettingsDep,
) -> HandwritingOCRResponse:
    try:
        image_b64, media_type = await resolve_image(
            request.image_base64, request.image_url, request.media_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    language_instruction = ""
    if request.language_hint != "auto":
        language_instruction = f"The text is expected to be in {request.language_hint}. "

    prompt = (
        "You are analyzing a handwritten document. "
        f"{language_instruction}"
        "Read ALL text from this handwritten image carefully. "
        "Transcribe exactly what is written, preserving paragraph breaks. "
        "If a word or fragment is illegible, write it as [неразборчиво] or [illegible]. "
        "Rate the overall legibility of the handwriting from 1 (completely illegible) "
        "to 10 (perfectly clear print-like writing). "
        "List any fragments where you are uncertain about the reading."
    )

    try:
        result = await analyze_image(
            llm=llm,
            images=[(image_b64, media_type)],
            prompt=prompt,
            response_model=HandwritingOCRResult,
            detail="high",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR analysis failed: {e}")

    return HandwritingOCRResponse(ocr=result, model=settings.model_name)


@router.post("/diagram")
async def analyze_diagram(
    request: DiagramAnalysisRequest,
    llm: LLMDep,
    settings: SettingsDep,
) -> DiagramAnalysisResponse:
    try:
        image_b64, media_type = await resolve_image(
            request.image_base64, request.image_url, request.media_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    type_hint = ""
    if request.diagram_type_hint:
        type_hint = f"This is expected to be a {request.diagram_type_hint} diagram. "

    prompt = (
        f"Analyze this technical diagram in detail. {type_hint}"
        "1. Determine the diagram type (UML class, ER, flowchart, sequence, etc.) "
        "2. Identify ALL elements (classes, entities, processes, decisions, etc.) "
        "3. Identify ALL relationships between elements "
        "4. Check for correctness of notation and semantics "
        "5. Evaluate completeness — are expected elements missing? "
        "6. Evaluate visual clarity and readability "
        f"Grading criteria: {request.grading_criteria}"
    )

    try:
        result = await analyze_image(
            llm=llm,
            images=[(image_b64, media_type)],
            prompt=prompt,
            response_model=DiagramAnalysisResult,
            detail="high",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Diagram analysis failed: {e}")

    return DiagramAnalysisResponse(analysis=result, model=settings.model_name)


@router.post("/compare")
async def compare_with_reference(
    request: MultiImageCompareRequest,
    llm: LLMDep,
    settings: SettingsDep,
) -> MultiImageCompareResponse:
    try:
        student_b64, student_media = await resolve_image(
            request.student_image.image_base64,
            request.student_image.image_url,
            request.student_image.media_type,
        )
        reference_b64, reference_media = await resolve_image(
            request.reference_image.image_base64,
            request.reference_image.image_url,
            request.reference_image.media_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt = (
        "You are comparing two images. "
        "IMAGE 1 is the student's work. IMAGE 2 is the reference/example. "
        f"{request.comparison_prompt}\n\n"
        "Identify elements that match between both, elements missing from the "
        "student work, and extra elements the student added. "
        "Score similarity, accuracy, completeness, and overall quality (0-100 each). "
        "Provide specific, constructive feedback."
    )

    try:
        result = await analyze_image(
            llm=llm,
            images=[
                (student_b64, student_media),
                (reference_b64, reference_media),
            ],
            prompt=prompt,
            response_model=ComparisonResult,
            detail="high",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Comparison failed: {e}")

    return MultiImageCompareResponse(comparison=result, model=settings.model_name)
```

Как каждый эндпоинт связан с теорией:

| Эндпоинт | Концепция из теории | Что демонстрирует |
|---|---|---|
| `POST /analyze` | §2, §3, §6 | Базовый vision: текст + изображение → structured analysis. Поддержка detail level. |
| `POST /ocr` | §4, §8 | Vision + structured output для OCR. `detail: high` обязателен. Ограничения распознавания. |
| `POST /diagram` | §4, §6 | Сложная Pydantic-схема с elements/relationships. Промпт-инжиниринг для vision. |
| `POST /compare` | §5 | Multi-image input: два изображения в одном запросе. Порядок важен. |

### Шаг 4. Регистрация в `app/api/router.py`

Добавляем импорт и подключение нового роутера:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, prompts, multimodal

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(multimodal.router)
```

### Шаг 5. Тестирование с curl

Для тестирования vision-эндпоинтов нужно реальное изображение в base64. Создадим минимальное тестовое изображение — белый квадрат с текстом «Hello World» через Python:

```bash
python3 -c "
from PIL import Image, ImageDraw, ImageFont
import base64, io, json

img = Image.new('RGB', (400, 200), 'white')
draw = ImageDraw.Draw(img)
draw.text((50, 30), 'Hello World!', fill='black')
draw.text((50, 60), 'This is a test image', fill='gray')
draw.text((50, 90), 'Score: 85/100', fill='blue')
draw.rectangle([20, 20, 380, 180], outline='black', width=2)

buf = io.BytesIO()
img.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()
print(b64)
" > /tmp/test_image_b64.txt
```

Сохраняем base64 в переменную:

```bash
TEST_IMAGE=$(cat /tmp/test_image_b64.txt)
```

**POST /multimodal/analyze** — базовый анализ изображения:

```bash
curl -s -X POST http://localhost:8000/api/v1/multimodal/analyze \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$TEST_IMAGE\",
    \"media_type\": \"image/png\",
    \"prompt\": \"Describe what you see in this image. Read any text.\",
    \"detail\": \"high\"
  }" | python -m json.tool
```

Ожидаемый результат: `content_type: "screenshot"` или `"other"`, `text_content` содержит «Hello World!» и «Score: 85/100», `key_elements` включает текст и рамку.

**POST /multimodal/ocr** — распознавание текста:

```bash
curl -s -X POST http://localhost:8000/api/v1/multimodal/ocr \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$TEST_IMAGE\",
    \"media_type\": \"image/png\",
    \"language_hint\": \"en\"
  }" | python -m json.tool
```

Ожидаемый результат: `recognized_text` содержит весь текст с изображения, `legibility_score` высокий (8-10 для печатного текста), `language: "en"`.

**POST /multimodal/diagram** — анализ диаграммы:

Для этого теста создадим простую диаграмму:

```bash
python3 -c "
from PIL import Image, ImageDraw
import base64, io

img = Image.new('RGB', (500, 300), 'white')
draw = ImageDraw.Draw(img)

draw.rectangle([50, 50, 200, 120], outline='black', width=2)
draw.text((80, 75), 'User', fill='black')

draw.rectangle([300, 50, 450, 120], outline='black', width=2)
draw.text((330, 75), 'Order', fill='black')

draw.line([200, 85, 300, 85], fill='black', width=2)
draw.text((220, 65), '1..*', fill='red')

draw.rectangle([300, 180, 450, 250], outline='black', width=2)
draw.text((320, 205), 'Product', fill='black')

draw.line([375, 120, 375, 180], fill='black', width=2)
draw.text((385, 140), '*..1', fill='red')

buf = io.BytesIO()
img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
" > /tmp/test_diagram_b64.txt
```

```bash
DIAGRAM_IMAGE=$(cat /tmp/test_diagram_b64.txt)

curl -s -X POST http://localhost:8000/api/v1/multimodal/diagram \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$DIAGRAM_IMAGE\",
    \"media_type\": \"image/png\",
    \"diagram_type_hint\": \"ER\",
    \"grading_criteria\": \"correctness, completeness, notation\"
  }" | python -m json.tool
```

Ожидаемый результат: `diagram_type: "ER"`, `elements` содержит User, Order, Product, `relationships` описывает связи, `overall_score` — оценка качества диаграммы.

**POST /multimodal/compare** — сравнение двух изображений:

```bash
curl -s -X POST http://localhost:8000/api/v1/multimodal/compare \
  -H "Content-Type: application/json" \
  -d "{
    \"student_image\": {
      \"image_base64\": \"$TEST_IMAGE\",
      \"media_type\": \"image/png\"
    },
    \"reference_image\": {
      \"image_base64\": \"$DIAGRAM_IMAGE\",
      \"media_type\": \"image/png\"
    },
    \"comparison_prompt\": \"Compare these two images. The first is the student submission, the second is the reference.\"
  }" | python -m json.tool
```

Ожидаемый результат: низкий `similarity_score` (изображения разные), `missing_elements` и `extra_elements` перечисляют различия.

### Связь с теорией

| Эндпоинт | Разделы теории | Ключевые концепции |
|---|---|---|
| `POST /analyze` | §1, §2, §3, §6 | HumanMessage с content-списком, Base64 кодирование, detail level |
| `POST /ocr` | §2, §4, §8 | Vision + structured output, промпт-инжиниринг для OCR, ограничения |
| `POST /diagram` | §4, §6, §7 | Сложная Pydantic-схема, high detail для мелких деталей, vision chain |
| `POST /compare` | §3, §5, §7 | Multi-image input, порядок изображений, сравнительный анализ |

---

## Чеклист самопроверки

- [ ] Объясни разницу между `content: str` и `content: list[dict]` в `HumanMessage`. Когда используется какой формат?
- [ ] Почему LangChain использует формат `image_url` с data URI, а не нативный формат Anthropic? Что это даёт?
- [ ] Когда лучше передать изображение по URL, а когда через Base64? Назови по 2 сценария для каждого.
- [ ] Как `detail: low` vs `detail: high` влияет на стоимость и качество? Приведи конкретные числа по токенам.
- [ ] Почему OCR-эндпоинт использует `detail: "high"` принудительно, а analyze-эндпоинт — `detail` из запроса?
- [ ] Что происходит, если отправить RGBA-изображение в JPEG? Зачем нужна конвертация в `preprocess_image`?
- [ ] Как работает multi-image input? Почему порядок изображений в `content` имеет значение?
- [ ] Назови 3 задачи, где vision LLM работает хорошо, и 3, где лучше использовать OCR + text LLM.
- [ ] Почему `with_structured_output()` работает с vision без изменений? Что общего у vision и текстовых запросов на уровне API?
- [ ] Какой `max_tokens` рекомендуется для vision-запросов и почему он больше, чем для текстовых?

---

## Частые ошибки

### 1. Забыть media_type в data URI

```python
url = f"data:base64,{b64_data}"
```

```python
url = f"data:image/jpeg;base64,{b64_data}"
```

Без `image/jpeg` (или другого media type) провайдер не знает, как интерпретировать данные. Anthropic API вернёт ошибку `invalid_request_error`. Всегда указывайте полный data URI: `data:<media_type>;base64,<data>`.

### 2. Отправка слишком большого изображения без resize

```python
with open("photo_4096x4096.jpg", "rb") as f:
    huge_b64 = base64.b64encode(f.read()).decode()
```

```python
from app.services.vision import preprocess_image

with open("photo_4096x4096.jpg", "rb") as f:
    optimized_b64, media_type = preprocess_image(f.read(), max_dimension=1536)
```

Изображение 4096×4096 JPEG может весить 5-15 MB. После base64 это 7-20 MB в JSON-запросе. Это увеличивает latency на 2-5 секунд только на передачу данных. Resize до 1536px сохраняет качество для vision-анализа и уменьшает размер в 5-10 раз.

### 3. detail: "high" для задач, где достаточно "low"

```python
result = await analyze_image(
    llm=llm,
    images=[(b64, media_type)],
    prompt="Is this a photo or a diagram?",
    response_model=ContentType,
    detail="high",
)
```

```python
result = await analyze_image(
    llm=llm,
    images=[(b64, media_type)],
    prompt="Is this a photo or a diagram?",
    response_model=ContentType,
    detail="low",
)
```

Для классификации типа контента `detail: low` (85 токенов) достаточно. `detail: high` (765+ токенов) — пустая трата бюджета. Используйте `high` только когда нужны мелкие детали: распознавание текста, анализ элементов диаграмм, чтение чисел.

### 4. Ожидание pixel-perfect OCR от vision модели

```python
class ExactOCR(BaseModel):
    text: str = Field(description="Exact character-by-character transcription")
    character_count: int = Field(description="Exact number of characters")
```

```python
class RealisticOCR(BaseModel):
    text: str = Field(
        description="Best-effort transcription. Use [неразборчиво] for illegible parts",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident in the transcription accuracy",
    )
```

Vision LLM — не OCR-движок. Для рукописного текста точность — 80-95%, для мелкого печатного — 90-98%. Всегда включайте поле `confidence` и механизм для обозначения неуверенных фрагментов. Для pixel-perfect OCR используйте Tesseract/EasyOCR, а результат анализируйте text LLM.

### 5. Один prompt для всех типов изображений

```python
prompt = "Analyze this image"
```

```python
prompts = {
    "handwriting": "Read ALL handwritten text carefully...",
    "diagram": "Identify diagram type, elements, relationships...",
    "screenshot": "Describe the UI elements, read all text...",
    "photo": "Describe the scene, identify objects...",
}
prompt = prompts.get(content_type, "Describe what you see in detail")
```

Разные типы изображений требуют разных промптов. Промпт для OCR рукописного текста должен акцентировать внимание на разборчивости и неуверенных фрагментах. Промпт для диаграммы — на элементах и связях. Универсальный промпт «analyze this» даст поверхностный результат.

### 6. Не учитывать стоимость: vision tokens значительно дороже текстовых

```python
for page in all_200_pages:
    result = await analyze_image(llm, [(page, "image/jpeg")], "OCR", Model, detail="high")
```

```python
pages_to_analyze = all_200_pages[:20]
for page in pages_to_analyze:
    result = await analyze_image(llm, [(page, "image/jpeg")], "OCR", Model, detail="high")
```

200 изображений × ~1000 токенов (high detail) = 200,000 input tokens только на изображения. При $3/M — $0.60 за один запрос. Плюс output tokens. Ограничивайте количество изображений, используйте `detail: low` где возможно, и помните о лимите Anthropic (20 изображений на запрос).

---

## Что читать дальше

- [Anthropic Vision Guide](https://docs.anthropic.com/en/docs/build-with-claude/vision) — официальная документация по vision в Claude
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) — документация GPT-4V/4o vision
- [LangChain Multimodal Messages](https://python.langchain.com/docs/concepts/messages/#humanmessage) — формат мультимодальных сообщений
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/) — обработка изображений в Python
- [httpx Documentation](https://www.python-httpx.org/) — асинхронный HTTP-клиент
- [Base64 Encoding](https://docs.python.org/3/library/base64.html) — стандартная библиотека Python

**Следующая тема:** [Тема 13: Advanced Agentic Patterns](topic_13_agentic_patterns.md)

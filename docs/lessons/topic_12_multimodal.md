# Тема 12: Multimodal AI — работа с изображениями

> **Пререквизиты:** [Тема 1-3](topic_01_prompt_engineering.md), рекомендуется [Тема 6 (LangGraph)](topic_06_langgraph_agents.md)
> **Зависимости:** `langchain-core`, `langchain-anthropic`, `langchain-openai`, `pillow`, `pymupdf`

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

Архитектурно это реализуется через LangGraph-граф, где vision-анализ — отдельный узел. Но для простых задач достаточно прямого вызова vision LLM без полноценного агента.

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

**Описание:** Асинхронный HTTP-клиент для загрузки изображений по URL. В отличие от `requests`, httpx поддерживает `async/await` и лучше подходит для асинхронных приложений.

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

## Практика

### Пример 1. Отправка изображения в LLM

Создадим тестовое изображение и отправим его в vision-модель двумя способами: через Base64 и по URL.

**Подготовка тестового изображения:**

```python
from PIL import Image, ImageDraw
import base64
import io

img = Image.new("RGB", (400, 200), "white")
draw = ImageDraw.Draw(img)
draw.text((50, 30), "Hello World!", fill="black")
draw.text((50, 60), "Тестовое изображение", fill="gray")
draw.text((50, 90), "Оценка: 85/100", fill="blue")
draw.rectangle([20, 20, 380, 180], outline="black", width=2)

buffer = io.BytesIO()
img.save(buffer, format="PNG")
test_image_bytes = buffer.getvalue()

b64_data = base64.b64encode(test_image_bytes).decode("utf-8")
print(f"Размер изображения: {len(test_image_bytes)} байт")
print(f"Длина Base64: {len(b64_data)} символов")
```

**Base64 — отправка локального файла:**

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
        {"type": "text", "text": "Опиши что изображено на картинке. Прочитай весь текст."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
        },
    ]
)

result = llm.invoke([message])
print(result.content)
```

**URL — публичное изображение:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm_openai = ChatOpenAI(model="gpt-4o", max_tokens=4096, temperature=0.0)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Опиши что изображено на этой картинке."},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            },
        },
    ]
)

result = llm_openai.invoke([message])
print(result.content)
```

Один и тот же формат `image_url` работает и с `ChatAnthropic`, и с `ChatOpenAI` — LangChain автоматически конвертирует в нативный формат провайдера.

**Preprocessing перед отправкой:**

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

    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    if image.mode in ("RGBA", "P"):
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

with open("large_photo.jpg", "rb") as f:
    raw_bytes = f.read()

b64_optimized, media_type = preprocess_for_vision(raw_bytes)
print(f"Оригинал: {len(raw_bytes)} байт")
print(f"После preprocessing: ~{len(b64_optimized) * 3 // 4} байт")
print(f"Media type: {media_type}")
```

**Управление detail level:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)

message_low = HumanMessage(
    content=[
        {"type": "text", "text": "Что это — фото, скриншот или диаграмма?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_data}",
                "detail": "low",
            },
        },
    ]
)

message_high = HumanMessage(
    content=[
        {"type": "text", "text": "Прочитай весь текст на изображении дословно."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_data}",
                "detail": "high",
            },
        },
    ]
)

result_low = llm.invoke([message_low])
print("=== detail=low (85 токенов) ===")
print(result_low.content)

result_high = llm.invoke([message_high])
print("\n=== detail=high (765+ токенов) ===")
print(result_high.content)
```

`detail: low` — 85 токенов, достаточно для классификации. `detail: high` — 765-1745 токенов, нужен для чтения мелкого текста и деталей диаграмм.

---

### Пример 2. Vision + Structured Output

Комбинация vision + `with_structured_output()`: модель анализирует изображение и возвращает типизированный результат.

**Анализ изображения со структурированным ответом:**

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

class ImageAnalysis(BaseModel):
    description: str = Field(description="Подробное описание содержания изображения")
    key_elements: list[str] = Field(description="Ключевые визуальные элементы")
    text_content: str | None = Field(
        default=None,
        description="Текст на изображении, None если текста нет",
    )
    content_type: str = Field(
        description="Тип: photo, screenshot, diagram, handwriting, chart, other",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в анализе")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)
structured_llm = llm.with_structured_output(ImageAnalysis)

messages = [
    SystemMessage(content="Ты — эксперт по анализу изображений. Анализируй внимательно и точно."),
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Проанализируй изображение. Определи тип контента, "
                        "опиши ключевые элементы, прочитай текст если есть.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_data}",
                    "detail": "high",
                },
            },
        ]
    ),
]

result = structured_llm.invoke(messages)
print(f"Тип контента: {result.content_type}")
print(f"Описание: {result.description}")
print(f"Элементы: {result.key_elements}")
print(f"Текст: {result.text_content}")
print(f"Уверенность: {result.confidence}")
```

**OCR рукописного текста:**

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

class HandwritingOCR(BaseModel):
    recognized_text: str = Field(
        description="Распознанный текст, [неразборчиво] для нечитаемых частей",
    )
    legibility_score: int = Field(
        ge=1, le=10,
        description="Разборчивость: 1=нечитаемо, 10=идеально",
    )
    language: str = Field(description="Язык текста")
    uncertain_fragments: list[str] = Field(
        default_factory=list,
        description="Фрагменты с неуверенным распознаванием",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в распознавании")

with open("handwriting.jpg", "rb") as f:
    hw_b64 = base64.b64encode(f.read()).decode("utf-8")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)
structured_llm = llm.with_structured_output(HandwritingOCR)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Прочитай рукописный текст на изображении. "
                    "Если слово неразборчиво — напиши [неразборчиво]. "
                    "Оцени разборчивость почерка от 1 до 10.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{hw_b64}",
                "detail": "high",
            },
        },
    ]
)

result = structured_llm.invoke([message])
print(f"Текст: {result.recognized_text}")
print(f"Разборчивость: {result.legibility_score}/10")
print(f"Язык: {result.language}")
print(f"Уверенность: {result.confidence}")
if result.uncertain_fragments:
    print(f"Неуверенные фрагменты: {result.uncertain_fragments}")
```

**Анализ диаграммы:**

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

class DiagramElement(BaseModel):
    name: str = Field(description="Имя элемента")
    element_type: str = Field(description="Тип: class, entity, process, decision и т.д.")

class DiagramRelationship(BaseModel):
    source: str = Field(description="Исходный элемент")
    target: str = Field(description="Целевой элемент")
    relationship_type: str = Field(
        description="Тип: association, inheritance, dependency, flow",
    )
    label: str | None = Field(default=None, description="Подпись связи")

class DiagramAnalysis(BaseModel):
    diagram_type: str = Field(
        description="Тип диаграммы: UML class, ER, flowchart, sequence и т.д.",
    )
    elements: list[DiagramElement] = Field(description="Все найденные элементы")
    relationships: list[DiagramRelationship] = Field(description="Все найденные связи")
    correctness_score: int = Field(ge=0, le=100, description="Корректность нотации")
    completeness_score: int = Field(ge=0, le=100, description="Полнота диаграммы")
    issues: list[str] = Field(description="Найденные проблемы")
    suggestions: list[str] = Field(description="Предложения по улучшению")


with open("diagram.png", "rb") as f:
    diagram_b64 = base64.b64encode(f.read()).decode("utf-8")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)
structured_llm = llm.with_structured_output(DiagramAnalysis)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Проанализируй эту техническую диаграмму. "
                    "Определи тип, все элементы, связи между ними. "
                    "Проверь корректность нотации. Оцени полноту.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{diagram_b64}",
                "detail": "high",
            },
        },
    ]
)

result = structured_llm.invoke([message])
print(f"Тип диаграммы: {result.diagram_type}")
print(f"Корректность: {result.correctness_score}/100")
print(f"Полнота: {result.completeness_score}/100")
print(f"\nЭлементы ({len(result.elements)}):")
for el in result.elements:
    print(f"  - {el.name} ({el.element_type})")
print(f"\nСвязи ({len(result.relationships)}):")
for rel in result.relationships:
    label = f" [{rel.label}]" if rel.label else ""
    print(f"  - {rel.source} → {rel.target} ({rel.relationship_type}){label}")
if result.issues:
    print(f"\nПроблемы:")
    for issue in result.issues:
        print(f"  - {issue}")
```

---

### Пример 3. Multi-image input

Отправка нескольких изображений в одном запросе — для сравнения, анализа многостраничных документов.

**Сравнение работы студента с эталоном:**

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

class ComparisonResult(BaseModel):
    similarity_score: int = Field(ge=0, le=100, description="Сходство с эталоном 0-100")
    matching_elements: list[str] = Field(
        description="Элементы, присутствующие в обеих работах",
    )
    missing_elements: list[str] = Field(
        description="Элементы эталона, отсутствующие в работе студента",
    )
    extra_elements: list[str] = Field(
        description="Элементы студента, отсутствующие в эталоне",
    )
    overall_score: int = Field(ge=0, le=100, description="Общая оценка")
    feedback: str = Field(description="Развёрнутая обратная связь, 3-5 предложений")


with open("student_work.png", "rb") as f:
    student_b64 = base64.b64encode(f.read()).decode("utf-8")
with open("reference.png", "rb") as f:
    reference_b64 = base64.b64encode(f.read()).decode("utf-8")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)
structured_llm = llm.with_structured_output(ComparisonResult)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Сравни две работы. ПЕРВОЕ изображение — работа студента. "
                    "ВТОРОЕ изображение — эталон. "
                    "Определи, какие элементы совпадают, чего не хватает, что лишнее.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{student_b64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{reference_b64}"},
        },
    ]
)

result = structured_llm.invoke([message])
print(f"Сходство: {result.similarity_score}/100")
print(f"Общая оценка: {result.overall_score}/100")
print(f"Совпадающие элементы: {result.matching_elements}")
print(f"Отсутствующие элементы: {result.missing_elements}")
print(f"Лишние элементы: {result.extra_elements}")
print(f"\nОбратная связь: {result.feedback}")
```

Порядок изображений в `content` имеет значение — модель обрабатывает блоки последовательно. Если промпт говорит «первое — студент, второе — эталон», расположение должно соответствовать.

**Многостраничный документ:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

pages_b64 = []
for i in range(1, 4):
    with open(f"page_{i}.jpg", "rb") as f:
        pages_b64.append(base64.b64encode(f.read()).decode("utf-8"))

content_blocks: list[dict] = [
    {
        "type": "text",
        "text": "Перед тобой 3 страницы рукописной работы. "
                "Проанализируй все страницы как единый документ.",
    },
]
for i, page_b64 in enumerate(pages_b64, 1):
    content_blocks.append({"type": "text", "text": f"--- Страница {i} ---"})
    content_blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{page_b64}"},
    })

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0.0)
result = llm.invoke([HumanMessage(content=content_blocks)])
print(result.content)
```

---

### Пример 4. Анализ PDF-документа

PDF-страницы конвертируются в изображения через `pymupdf` и отправляются в vision-модель.

```python
import fitz
import base64
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

class PDFPageSummary(BaseModel):
    page_number: int = Field(description="Номер страницы")
    content_type: str = Field(description="Тип контента: text, table, diagram, mixed")
    summary: str = Field(description="Краткое содержание страницы")
    key_points: list[str] = Field(description="Ключевые тезисы")

class PDFAnalysis(BaseModel):
    total_pages: int = Field(description="Количество проанализированных страниц")
    pages: list[PDFPageSummary] = Field(description="Анализ каждой страницы")
    overall_summary: str = Field(description="Общее резюме документа")
    document_type: str = Field(
        description="Тип документа: article, report, presentation, form, other",
    )

def pdf_pages_to_base64(
    pdf_path: str,
    max_pages: int = 5,
    dpi: int = 150,
) -> list[str]:
    doc = fitz.open(pdf_path)
    pages_b64 = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        pages_b64.append(base64.b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return pages_b64


pages = pdf_pages_to_base64("document.pdf", max_pages=5)
print(f"Сконвертировано страниц: {len(pages)}")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192, temperature=0.0)
structured_llm = llm.with_structured_output(PDFAnalysis)

content_blocks: list[dict] = [
    {
        "type": "text",
        "text": f"Проанализируй PDF-документ ({len(pages)} страниц). "
                "Определи тип документа, кратко опиши содержание каждой страницы, "
                "выдели ключевые тезисы и подготовь общее резюме.",
    },
]
for i, page_b64 in enumerate(pages, 1):
    content_blocks.append({"type": "text", "text": f"--- Страница {i} ---"})
    content_blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{page_b64}"},
    })

result = structured_llm.invoke([HumanMessage(content=content_blocks)])
print(f"Тип документа: {result.document_type}")
print(f"Общее резюме: {result.overall_summary}")
for page in result.pages:
    print(f"\nСтраница {page.page_number} ({page.content_type}):")
    print(f"  {page.summary}")
    for point in page.key_points:
        print(f"  • {point}")
```

При 5 страницах с `detail: auto` — примерно 4000-8000 input-токенов на изображения. Ограничивайте количество страниц и используйте `dpi=150` (а не 300) для баланса качества и стоимости.

### Связь с теорией

| Пример | Разделы теории | Ключевые концепции |
|---|---|---|
| Пример 1 (изображение в LLM) | §1, §2, §3, §6 | HumanMessage с content-списком, Base64 и URL, detail level, preprocessing |
| Пример 2 (structured output) | §4, §7, §8 | Vision + with_structured_output(), промпт-инжиниринг для vision, ограничения |
| Пример 3 (multi-image) | §5 | Multi-image input, порядок изображений, сравнительный анализ |
| Пример 4 (PDF-анализ) | §2, §4, §5 | Конвертация PDF в изображения, multi-page analysis, structured output |

---

## Чеклист самопроверки

- [ ] Объясни разницу между `content: str` и `content: list[dict]` в `HumanMessage`. Когда используется какой формат?
- [ ] Почему LangChain использует формат `image_url` с data URI, а не нативный формат Anthropic? Что это даёт?
- [ ] Когда лучше передать изображение по URL, а когда через Base64? Назови по 2 сценария для каждого.
- [ ] Как `detail: low` vs `detail: high` влияет на стоимость и качество? Приведи конкретные числа по токенам.
- [ ] Почему для OCR рукописного текста рекомендуется `detail: "high"`, а для классификации типа контента достаточно `detail: "low"`?
- [ ] Что происходит, если отправить RGBA-изображение в JPEG? Зачем нужна конвертация перед отправкой в vision API?
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
from PIL import Image
import io
import base64

with open("photo_4096x4096.jpg", "rb") as f:
    image = Image.open(io.BytesIO(f.read()))

image.thumbnail((1536, 1536), Image.LANCZOS)
if image.mode != "RGB":
    image = image.convert("RGB")

buffer = io.BytesIO()
image.save(buffer, format="JPEG", quality=85)
optimized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
```

Изображение 4096×4096 JPEG может весить 5-15 MB. После base64 это 7-20 MB в JSON-запросе. Это увеличивает latency на 2-5 секунд только на передачу данных. Resize до 1536px сохраняет качество для vision-анализа и уменьшает размер в 5-10 раз.

### 3. detail: "high" для задач, где достаточно "low"

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "Is this a photo or a diagram?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        },
    ]
)
result = structured_llm.invoke([message])
```

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "Is this a photo or a diagram?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        },
    ]
)
result = structured_llm.invoke([message])
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
for page_b64 in all_200_pages:
    message = HumanMessage(content=[
        {"type": "text", "text": "Распознай текст"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page_b64}", "detail": "high"}},
    ])
    result = structured_llm.invoke([message])
```

```python
pages_to_analyze = all_200_pages[:20]
for page_b64 in pages_to_analyze:
    message = HumanMessage(content=[
        {"type": "text", "text": "Распознай текст"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page_b64}", "detail": "high"}},
    ])
    result = structured_llm.invoke([message])
```

200 изображений × ~1000 токенов (high detail) = 200,000 input tokens только на изображения. При $3/M — $0.60 за один запрос. Плюс output tokens. Ограничивайте количество изображений, используйте `detail: low` где возможно, и помните о лимите Anthropic (20 изображений на запрос).

---

## Что читать дальше

- [Anthropic Vision Guide](https://docs.anthropic.com/en/docs/build-with-claude/vision) — официальная документация по vision в Claude
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) — документация GPT-4V/4o vision
- [LangChain Multimodal Messages](https://python.langchain.com/docs/concepts/messages/#humanmessage) — формат мультимодальных сообщений
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/) — обработка изображений в Python
- [httpx Documentation](https://www.python-httpx.org/) — асинхронный HTTP-клиент
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/) — работа с PDF-документами
- [Base64 Encoding](https://docs.python.org/3/library/base64.html) — стандартная библиотека Python

**Следующая тема:** [Тема 13: Advanced Agentic Patterns](topic_13_agentic_patterns.md)

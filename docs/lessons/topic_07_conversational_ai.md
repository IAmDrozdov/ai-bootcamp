# Тема 7: Conversational AI

> **Пререквизиты:** [Тема 6: LangGraph](topic_06_langgraph_agents.md)
> **Что добавим в проект:** `app/schemas/chat.py`, `app/graph/conversation_graph.py`, `app/api/v1/chat.py`
> **Зависимости:** `langgraph` (группа `agents`), `langchain-core`, `langchain-anthropic`

---

## Теория

### 1. LLM не помнит предыдущие сообщения

Каждый вызов LLM API — это **stateless HTTP-запрос**. В отличие от классических веб-приложений, где сессия пользователя живёт на сервере (cookies, JWT, Redis-сессии), LLM-провайдер не хранит никакого состояния между запросами. Для модели каждый запрос — первый и единственный.

Чтобы вести диалог, приложение должно **каждый раз** отправлять полную историю сообщений:

```
Вызов 1: [system, human("Оцени мою работу")]          → AI("Оценка: 75/100...")
Вызов 2: [system, human("Оцени"), AI("75/100"), human("Почему не 80?")] → AI("Потому что...")
Вызов 3: [system, human, AI, human, AI, human("А как улучшить?")] → AI("Рекомендации...")
```

С каждым обменом контекст **растёт линейно**. Это создаёт три проблемы:

1. **Стоимость.** API тарифицирует и input-, и output-токены. Через 20 сообщений вы платите за повторную обработку тысяч токенов, которые модель уже "видела". Для Claude Sonnet при контексте 10k input-токенов это ~$0.03 за каждый вызов — и расходы растут с каждым сообщением.

2. **Лимит контекста.** У Claude — 200k токенов, у GPT-4 — 128k. Даже при большом лимите, длинный контекст деградирует качество из-за эффекта **"lost in the middle"**: модель хуже внимает информации в середине длинного контекста, чем в начале или конце.

3. **Латентность.** Чем длиннее input, тем дольше модель обрабатывает запрос. Time-to-first-token растёт пропорционально длине контекста, потому что attention-механизм вычисляет попарные связи между всеми токенами (O(n²) по длине).

Архитектурное следствие: приложению нужен **менеджер памяти** — компонент, который решает, что хранить, что сжимать и что забывать.

### 2. Типы сообщений в LangChain

LangChain моделирует диалог через типизированные объекты сообщений, наследующие `BaseMessage`:

| Класс | Роль в API | Назначение |
|-------|-----------|------------|
| `SystemMessage` | `system` | Инструкции, роль, ограничения. Повышенный вес в attention. |
| `HumanMessage` | `user` | Ввод пользователя. Untrusted input в production. |
| `AIMessage` | `assistant` | Ответ модели. Содержит `response_metadata` (токены, stop reason). |
| `ToolMessage` | `tool` | Результат вызова инструмента. Привязан к `tool_call_id`. |

Каждое сообщение имеет поля `content` (текст или список блоков), `type` (строковый идентификатор роли) и `id` (уникальный идентификатор для трекинга). `AIMessage` дополнительно содержит `tool_calls` — список запросов на вызов инструментов, и `usage_metadata` — количество потреблённых токенов.

Типизация критична: когда вы отправляете список `[SystemMessage, HumanMessage, AIMessage, HumanMessage]`, API получает корректные роли для каждого сообщения. Если склеить всё в одну строку — модель не сможет отличить свои ответы от вопросов пользователя.

### 3. Стратегии управления памятью

#### Buffer memory — хранить всё

Простейший подход: сохраняем все сообщения, отправляем всё каждый раз.

| Параметр | Значение |
|----------|----------|
| Плюсы | Полный контекст, ничего не теряется, нулевая сложность реализации |
| Минусы | Растущая стоимость, лимит контекста, "lost in the middle" |
| Когда | Короткие диалоги (< 20 сообщений), прототипы |

Buffer memory — это дефолтное поведение, когда вы просто складываете все сообщения в список. Работает идеально для коротких разговоров, но непригодно для production-систем с длинными сессиями.

#### Window memory — скользящее окно

Хранить последние N сообщений. Старые — удаляются безвозвратно.

| Параметр | Значение |
|----------|----------|
| Плюсы | Предсказуемая стоимость, фиксированный размер контекста |
| Минусы | Теряется ранний контекст ("О чём мы говорили 10 минут назад?") |
| Когда | Длинные сессии, где важен недавний контекст |

Важная деталь: окно лучше считать не в сообщениях, а в **токенах**. Одно сообщение может содержать 50 токенов или 5000 — фиксированное количество сообщений не гарантирует фиксированный размер контекста. Практический подход: хранить последние K сообщений, но проверять суммарное количество токенов и урезать при необходимости.

#### Summary memory — сжатие через LLM

Старые сообщения сжимаются в краткое резюме через отдельный LLM-вызов. Контекст = резюме + последние N сообщений.

| Параметр | Значение |
|----------|----------|
| Плюсы | Сохраняется суть всего разговора при фиксированном размере |
| Минусы | Дополнительный вызов LLM для суммаризации, потеря деталей и нюансов |
| Когда | Длинные диалоги, где важна тема, но не точные формулировки |

Алгоритм summary memory:
1. Когда количество сообщений превышает порог (например, 10) — выделить старые (все кроме последних 5)
2. Отправить старые сообщения в LLM с промптом "Summarize this conversation, preserving key decisions and scores"
3. Заменить старые сообщения на `SystemMessage(content=summary)`
4. Новый контекст: `[system_prompt, summary_message, last_5_messages, new_human_message]`

Экономия значительная: 20 сообщений (~4000 токенов) сжимаются до резюме в ~200 токенов. Но каждая суммаризация — это дополнительный LLM-вызов (~1-2 секунды, ~$0.005).

#### Комбинированный подход

В production обычно комбинируют стратегии: buffer memory для текущей сессии, window для длинных разговоров, summary для архивных. Выбор зависит от требований к полноте контекста, бюджета на токены и допустимой латентности.

### 4. MessagesPlaceholder — вставка истории

`MessagesPlaceholder` — специальный элемент в `ChatPromptTemplate`, который принимает **список типизированных сообщений** и вставляет их в промпт с сохранением ролей.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assessment discussion assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
```

Как это работает внутри: при вызове `prompt.invoke({"chat_history": [...], "question": "..."})`, шаблон берёт переменную `chat_history` (список BaseMessage-объектов) и **разворачивает** её в последовательность сообщений с правильными ролями. Итоговый prompt выглядит так:

```
[SystemMessage("You are..."),
 HumanMessage("Оцени мою работу"),
 AIMessage("Оценка: 75/100..."),
 HumanMessage("Почему не 80?")]
```

Без `MessagesPlaceholder` пришлось бы склеивать историю в строку вроде `"User: ...\nAssistant: ..."` — модель бы не могла надёжно отличить роли, и API не получил бы корректную чередующуюся структуру ролей, которая критична для качества ответов.

Параметр `optional=True` позволяет не передавать историю на первом ходу: если переменная отсутствует, placeholder просто пропускается. Без этого флага отсутствие переменной вызовет `KeyError`.

### 5. LangGraph для диалога

LangGraph решает фундаментальную проблему conversational AI: **где и как хранить состояние разговора между HTTP-запросами**.

Ключевые свойства LangGraph для диалога:

**Stateful граф.** Состояние (`TypedDict`) хранит историю сообщений и любые дополнительные данные (результат оценки, контекст работы студента). Каждый узел получает текущее состояние и возвращает обновления.

**Reducers.** Обычное поведение TypedDict — замена значения. Но для сообщений нужно **добавление**: новое сообщение добавляется к списку, а не заменяет его. Это делает паттерн `Annotated[list, add_messages]`:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
```

Когда узел возвращает `{"messages": [AIMessage("...")]}`, reducer `add_messages` добавляет это сообщение к существующему списку, а не заменяет его. `add_messages` также поддерживает дедупликацию по `id` и удаление через `RemoveMessage`.

**Checkpointing.** `MemorySaver` автоматически сохраняет состояние после каждого шага графа и восстанавливает его при следующем вызове с тем же `thread_id`. Это означает, что второй вызов `graph.ainvoke({"messages": [HumanMessage("...")]}, config)` автоматически получит всю предыдущую историю — без ручного управления.

**Thread isolation.** `thread_id` в конфигурации изолирует разговоры. Два пользователя с разными `thread_id` работают с независимыми состояниями, даже если используют один и тот же скомпилированный граф. Это позволяет создать один граф при старте приложения и использовать его для всех пользователей.

```python
config_user_1 = {"configurable": {"thread_id": "user-1-session"}}
config_user_2 = {"configurable": {"thread_id": "user-2-session"}}
```

**Условная маршрутизация.** В conversational AI часто нужна логика: если оценка ещё не проведена — оценить; если уже есть — обсудить. `add_conditional_edges` позволяет направлять поток в зависимости от состояния, создавая динамические сценарии диалога.

### 6. WebSocket vs SSE для чата

| Характеристика | SSE (Server-Sent Events) | WebSocket |
|---------------|--------------------------|-----------|
| Направление | Только сервер → клиент | Двунаправленное |
| Протокол | HTTP (Content-Type: text/event-stream) | WS (отдельный протокол, upgrade от HTTP) |
| Подключение | Новый HTTP-запрос на каждое сообщение пользователя | Одно постоянное соединение |
| Реконнект | Встроенный (EventSource API) | Ручной (нужна логика переподключения) |
| Для LLM | Стриминг одного ответа | Полноценный real-time чат |
| Сложность сервера | Простая (обычный HTTP endpoint) | Средняя (WebSocket handler, lifecycle) |
| Firewall/Proxy | Работает везде (обычный HTTP) | Иногда блокируется корпоративными прокси |

**SSE** идеален, когда клиент отправляет один запрос и получает потоковый ответ — как в нашем endpoint `POST /assess/stream`. Клиент делает POST, сервер стримит токены через SSE, соединение закрывается.

**WebSocket** нужен для полноценного чата: клиент отправляет сообщения и получает ответы через одно постоянное соединение, без пересоздания HTTP-запросов. Жизненный цикл WebSocket:

1. **Handshake** — клиент отправляет HTTP-запрос с заголовком `Upgrade: websocket`, сервер отвечает `101 Switching Protocols`
2. **Open** — соединение установлено, обе стороны могут отправлять фреймы
3. **Message exchange** — JSON-сообщения в обе стороны
4. **Close** — любая сторона инициирует закрытие, отправляя close-фрейм

FastAPI имеет встроенную поддержку WebSocket через Starlette. Декоратор `@router.websocket("/path")` создаёт endpoint, который принимает `WebSocket` объект с методами `accept()`, `receive_json()`, `send_json()` и `close()`.

Для нашего проекта WebSocket даёт два преимущества: (1) двунаправленность — студент и AI-ассессор обмениваются сообщениями в реальном времени; (2) стриминг — ответ LLM приходит токен за токеном через то же соединение.

---

## Справочник API

### MessagesPlaceholder

**Описание:** элемент шаблона `ChatPromptTemplate`, который вставляет динамический список сообщений с сохранением ролей.

**Импорт:**

```python
from langchain_core.prompts import MessagesPlaceholder
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `variable_name` | `str` | обязательный | Имя переменной, из которой берётся список сообщений |
| `optional` | `bool` | `False` | Если `True`, отсутствие переменной не вызовет ошибку |
| `n_messages` | `int \| None` | `None` | Ограничить количество вставляемых сообщений (последние N) |

**Пример:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

result = prompt.invoke({
    "chat_history": [
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4"),
    ],
    "input": "And 3+3?",
})
```

---

### HumanMessage / AIMessage / SystemMessage

**Описание:** типизированные сообщения диалога. Все наследуют `BaseMessage` и различаются полем `type`, которое мапится на роль в API провайдера.

**Импорт:**

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
```

**Конструктор (общий для всех):**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `content` | `str \| list[str \| dict]` | обязательный | Текст или мультимодальный контент (текст + изображения) |
| `additional_kwargs` | `dict` | `{}` | Дополнительные параметры от провайдера |
| `response_metadata` | `dict` | `{}` | Метаданные ответа (только для `AIMessage`) |
| `id` | `str \| None` | `None` | Уникальный идентификатор сообщения |
| `name` | `str \| None` | `None` | Имя автора (для мульти-агентных сценариев) |

**Дополнительные поля `AIMessage`:**

| Поле | Тип | Описание |
|------|-----|----------|
| `tool_calls` | `list[ToolCall]` | Запросы на вызов инструментов |
| `usage_metadata` | `UsageMetadata \| None` | Токены: `input_tokens`, `output_tokens`, `total_tokens` |
| `invalid_tool_calls` | `list[InvalidToolCall]` | Невалидные tool calls (ошибка парсинга) |

**Пример:**

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

system = SystemMessage(content="You are an expert assessor.")

human = HumanMessage(content="Rate my essay on climate change.")

ai = AIMessage(
    content="Your essay scores 75/100...",
    response_metadata={"model": "claude-sonnet-4-20250514", "stop_reason": "end_turn"},
)

multimodal = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
])
```

---

### StateGraph (паттерн conversation)

**Описание:** граф состояний LangGraph. Для conversational AI используется с reducer `add_messages`, который автоматически накапливает сообщения вместо замены.

**Импорт:**

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `state_schema` | `type` | обязательный | TypedDict-класс, определяющий структуру состояния |

**Основные методы:**

| Метод | Сигнатура | Описание |
|-------|-----------|----------|
| `add_node` | `(name: str, action: Callable)` | Добавить узел — функцию обработки |
| `add_edge` | `(start: str, end: str)` | Безусловное ребро между узлами |
| `add_conditional_edges` | `(source: str, path: Callable, path_map: dict \| None)` | Условное ребро: `path(state)` возвращает имя следующего узла |
| `compile` | `(checkpointer: BaseCheckpointSaver \| None)` | Скомпилировать граф в исполняемый `CompiledStateGraph` |

**Паттерн `Annotated[list, add_messages]`:**

```python
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
```

Reducer `add_messages` при обновлении состояния:
- Добавляет новые сообщения к существующему списку (не заменяет)
- Дедуплицирует по `id` — если сообщение с таким `id` уже есть, оно обновляется
- Поддерживает `RemoveMessage(id=...)` для удаления сообщений из истории

Альтернатива — `operator.add` — простая конкатенация списков без дедупликации:

```python
from operator import add

class SimpleState(TypedDict):
    messages: Annotated[list, add]
```

**Пример:**

```python
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(llm: ChatAnthropic):
    async def respond(state: ConversationState):
        all_messages = [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
        response = await llm.ainvoke(all_messages)
        return {"messages": [response]}

    graph = StateGraph(ConversationState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile()
```

---

### MemorySaver

**Описание:** in-memory checkpointer для LangGraph. Сохраняет состояние графа между вызовами в оперативной памяти процесса. Каждый `thread_id` имеет изолированную историю.

**Импорт:**

```python
from langgraph.checkpoint.memory import MemorySaver
```

**Конструктор:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `serde` | `SerializerProtocol \| None` | `None` | Кастомный сериализатор (обычно не нужен) |

**Ключевые методы (наследует `BaseCheckpointSaver`):**

| Метод | Описание |
|-------|----------|
| `get_tuple(config)` | Получить последний checkpoint для данного thread_id |
| `put(config, checkpoint, metadata, new_versions)` | Сохранить checkpoint |
| `list(config)` | Список всех checkpoints для thread_id |

**Ограничения:** данные живут только пока жив процесс. Перезапуск сервера = потеря всех разговоров. Для production используйте `SqliteSaver`, `PostgresSaver` или `AsyncPostgresSaver`.

**Пример:**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-42"}}
result1 = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]}, config)
result2 = await graph.ainvoke({"messages": [HumanMessage(content="What did I say?")]}, config)
```

---

### WebSocket (FastAPI)

**Описание:** класс WebSocket из Starlette/FastAPI для двунаправленной real-time коммуникации. Используется в endpoint с декоратором `@router.websocket("/path")`.

**Импорт:**

```python
from fastapi import WebSocket, WebSocketDisconnect
```

**Основные методы:**

| Метод | Сигнатура | Описание |
|-------|-----------|----------|
| `accept` | `async accept()` | Принять WebSocket-соединение (обязательно вызвать первым) |
| `receive_json` | `async receive_json() -> Any` | Получить JSON-сообщение от клиента |
| `receive_text` | `async receive_text() -> str` | Получить текстовое сообщение |
| `send_json` | `async send_json(data: Any)` | Отправить JSON клиенту |
| `send_text` | `async send_text(data: str)` | Отправить текст клиенту |
| `close` | `async close(code: int = 1000, reason: str \| None = None)` | Закрыть соединение |

**Коды закрытия WebSocket:**

| Код | Значение |
|-----|----------|
| 1000 | Нормальное закрытие |
| 1001 | Уход (going away) |
| 1008 | Нарушение политики |
| 1011 | Внутренняя ошибка сервера |

**Пример:**

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            response = f"Echo: {data['message']}"
            await websocket.send_json({"reply": response})
    except WebSocketDisconnect:
        pass
```

---

### ChatPromptTemplate с MessagesPlaceholder

**Описание:** комбинированный паттерн — шаблон, содержащий фиксированные сообщения (system), динамическую историю (MessagesPlaceholder) и текущий ввод (human). Стандартный шаблон для conversational AI.

**Пример полного шаблона:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assessor. Context:\n{context}"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

partial_prompt = prompt.partial(context="Assessment rubric: ...")

messages = partial_prompt.invoke({
    "chat_history": [
        HumanMessage(content="Assess my essay"),
        AIMessage(content="Score: 75/100..."),
    ],
    "input": "Why did I lose points on evidence?",
})
```

**Как формируется итоговый промпт:**

```
SystemMessage("You are an expert assessor. Context:\nAssessment rubric: ...")
HumanMessage("Assess my essay")
AIMessage("Score: 75/100...")
HumanMessage("Why did I lose points on evidence?")
```

Порядок сообщений критичен: system задаёт контекст, затем идёт история (от старых к новым), затем текущий вопрос пользователя. Модель "видит" последнее сообщение как текущий запрос.

---

## Практика: роутер `/api/v1/chat`

В этой практике мы создадим полноценный conversational AI backend: граф для диалога с checkpointing, REST-эндпоинты для управления разговорами и WebSocket для стриминга. Все компоненты используют LangGraph `MemorySaver` для автоматического сохранения истории по `thread_id`.

### Шаг 1: Схемы данных

Определим Pydantic-модели для request/response чата. `ChatMessage` — унифицированный формат сообщения с ролью. `AssessmentChatRequest` поддерживает два режима: первый вызов с `student_work` запускает оценку, последующие — обсуждение.

**Файл: `app/schemas/chat.py`**

```python
from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    message: str = Field(description="User message text")


class ChatMessage(BaseModel):
    role: str = Field(description="Message role: human, ai, or system")
    content: str = Field(description="Message content")


class ChatMessageResponse(BaseModel):
    reply: str = Field(description="AI response text")
    history: list[ChatMessage] = Field(description="Full conversation history")


class AssessmentChatRequest(BaseModel):
    message: str = Field(description="User message text")
    student_work: str | None = Field(
        default=None,
        description="Student work to assess (only first message)",
    )
    rubric_id: str | None = Field(default="essay_default")


class AssessmentChatResponse(BaseModel):
    reply: str = Field(description="AI response text")
    history: list[ChatMessage] = Field(description="Full conversation history")
    has_assessment: bool = Field(description="Whether assessment has been completed")
```

**Связь с теорией:** `ChatMessage` с полем `role` отражает структуру типизированных сообщений LangChain (раздел 2). Клиент получает историю в унифицированном формате, который легко отобразить в UI.

---

### Шаг 2: Граф разговора

Создадим два графа: `build_conversation_graph` — для общего чата, `build_assessment_conversation_graph` — для обсуждения оценки с условной маршрутизацией. Оба используют `MemorySaver` для персистентности.

**Файл: `app/graph/conversation_graph.py`**

```python
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]


class AssessmentConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    student_work: str
    assessment: str


CONVERSATION_SYSTEM = (
    "You are a helpful academic assistant. "
    "Answer questions clearly and concisely in the same language the student uses."
)

ASSESSMENT_SYSTEM = """\
You are an expert academic assessor. Evaluate the student work below \
thoroughly, providing scores and specific feedback for each criterion.

## Student Work
{student_work}"""

DISCUSSION_SYSTEM = """\
You are an expert academic assessor discussing a student's work and assessment.

## Student Work
{student_work}

## Completed Assessment
{assessment}

Answer questions about the assessment. Reference specific parts of the work \
when explaining scores. Be constructive and specific."""


def build_conversation_graph(llm: ChatAnthropic):
    async def respond(state: ConversationState):
        all_messages = [SystemMessage(content=CONVERSATION_SYSTEM)] + state["messages"]
        response = await llm.ainvoke(all_messages)
        return {"messages": [response]}

    graph = StateGraph(ConversationState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver())


def build_assessment_conversation_graph(llm: ChatAnthropic):
    async def assess(state: AssessmentConversationState):
        system_text = ASSESSMENT_SYSTEM.format(student_work=state["student_work"])
        all_messages = [SystemMessage(content=system_text)] + state["messages"]
        response = await llm.ainvoke(all_messages)
        return {"messages": [response], "assessment": response.content}

    async def discuss(state: AssessmentConversationState):
        system_text = DISCUSSION_SYSTEM.format(
            student_work=state["student_work"],
            assessment=state["assessment"],
        )
        all_messages = [SystemMessage(content=system_text)] + state["messages"]
        response = await llm.ainvoke(all_messages)
        return {"messages": [response]}

    def route(state: AssessmentConversationState) -> str:
        if not state.get("assessment"):
            return "assess"
        return "discuss"

    graph = StateGraph(AssessmentConversationState)
    graph.add_node("assess", assess)
    graph.add_node("discuss", discuss)
    graph.add_conditional_edges(START, route)
    graph.add_edge("assess", END)
    graph.add_edge("discuss", END)

    return graph.compile(checkpointer=MemorySaver())
```

**Связь с теорией:** `ConversationState` использует `Annotated[list, add_messages]` — паттерн reducer из раздела 5. `AssessmentConversationState` добавляет `student_work` и `assessment` — обычные строки без reducer, они замещаются при обновлении. Функция `route` реализует условную маршрутизацию: при первом вызове (assessment пуст) — оценка, далее — обсуждение. `MemorySaver` в `compile()` обеспечивает автоматическую персистентность по `thread_id`.

---

### Шаг 3: Роутер

Создадим роутер с пятью endpoints: отправка сообщения, получение истории, очистка, оценка с обсуждением, WebSocket со стримингом.

**Файл: `app/api/v1/chat.py`**

```python
from functools import lru_cache

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.dependencies import LLMDep, RubricStoreDep, get_llm
from app.graph.conversation_graph import (
    build_assessment_conversation_graph,
    build_conversation_graph,
)
from app.schemas.chat import (
    AssessmentChatRequest,
    AssessmentChatResponse,
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
)

router = APIRouter(prefix="/chat", tags=["lesson-7-chat"])

_thread_versions: dict[str, int] = {}


@lru_cache
def _get_graph():
    return build_conversation_graph(get_llm())


@lru_cache
def _get_assess_graph():
    return build_assessment_conversation_graph(get_llm())


def _effective_config(thread_id: str) -> dict:
    version = _thread_versions.get(thread_id, 0)
    effective_id = f"{thread_id}__v{version}" if version else thread_id
    return {"configurable": {"thread_id": effective_id}}


def _format_history(messages: list) -> list[ChatMessage]:
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "human"
        elif isinstance(m, AIMessage):
            role = "ai"
        elif isinstance(m, SystemMessage):
            continue
        else:
            role = "unknown"
        result.append(ChatMessage(role=role, content=m.content))
    return result


@router.post("/{thread_id}/message")
async def send_message(thread_id: str, request: ChatMessageRequest) -> ChatMessageResponse:
    graph = _get_graph()
    config = _effective_config(thread_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    messages = result["messages"]
    reply = messages[-1].content if messages else ""

    return ChatMessageResponse(
        reply=reply,
        history=_format_history(messages),
    )


@router.get("/{thread_id}/history")
async def get_history(thread_id: str) -> ChatMessageResponse:
    graph = _get_graph()
    config = _effective_config(thread_id)

    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])

    if not messages:
        raise HTTPException(status_code=404, detail=f"No history for thread '{thread_id}'")

    return ChatMessageResponse(
        reply=messages[-1].content if messages and isinstance(messages[-1], AIMessage) else "",
        history=_format_history(messages),
    )


@router.delete("/{thread_id}")
async def clear_thread(thread_id: str) -> dict[str, str]:
    _thread_versions[thread_id] = _thread_versions.get(thread_id, 0) + 1
    return {"status": "cleared", "thread_id": thread_id}


@router.post("/{thread_id}/assess")
async def assess_and_discuss(
    thread_id: str,
    request: AssessmentChatRequest,
    rubrics: RubricStoreDep,
) -> AssessmentChatResponse:
    graph = _get_assess_graph()
    config = _effective_config(thread_id)

    input_data: dict = {"messages": [HumanMessage(content=request.message)]}

    if request.student_work:
        rubric = rubrics.get(request.rubric_id or "essay_default")
        work_context = request.student_work
        if rubric:
            rubric_lines = [f"Rubric: {rubric.name}"]
            for c in rubric.criteria:
                rubric_lines.append(
                    f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}"
                )
            work_context = "\n".join(rubric_lines) + "\n\n" + request.student_work

        input_data["student_work"] = work_context
        input_data["assessment"] = ""

    result = await graph.ainvoke(input_data, config=config)

    messages = result["messages"]
    reply = messages[-1].content if messages else ""
    has_assessment = bool(result.get("assessment"))

    return AssessmentChatResponse(
        reply=reply,
        history=_format_history(messages),
        has_assessment=has_assessment,
    )


@router.websocket("/ws/{thread_id}")
async def websocket_chat(websocket: WebSocket, thread_id: str):
    await websocket.accept()
    graph = _get_graph()
    config = _effective_config(thread_id)

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            if not user_message:
                await websocket.send_json({"type": "error", "data": "Empty message"})
                continue

            input_data = {"messages": [HumanMessage(content=user_message)]}
            full_response = ""

            async for event in graph.astream_events(
                input_data, config=config, version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_response += token
                        await websocket.send_json({"type": "token", "data": token})

            await websocket.send_json({
                "type": "end",
                "data": {"message": full_response},
            })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json({"type": "error", "data": str(exc)})
        await websocket.close(code=1011)
```

**Связь с теорией:**

- `_effective_config` реализует thread isolation (раздел 5): каждый `thread_id` получает свою изолированную историю. Версионирование (`_thread_versions`) позволяет "очистить" разговор без удаления данных из MemorySaver — при инкременте версии эффективный `thread_id` меняется, и граф начинает новый разговор.
- `send_message` показывает stateless API поверх stateful графа (раздел 1): HTTP POST-запрос без состояния, но LangGraph восстанавливает историю из checkpointer автоматически.
- `websocket_chat` использует `astream_events` для token-by-token стриминга (раздел 6): событие `on_chat_model_stream` генерируется при каждом новом токене от LLM, мы пересылаем его клиенту через WebSocket.
- `assess_and_discuss` демонстрирует условную маршрутизацию (раздел 5): при первом вызове (`student_work` задан) граф идёт в узел `assess`, при последующих — в `discuss`.

---

### Шаг 4: Регистрация роутера

Добавим chat-роутер в главный router приложения.

**Обновление файла `app/api/router.py`:**

```python
from fastapi import APIRouter

from app.api.v1 import assessment, chat, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(chat.router)
```

---

### Шаг 5: Тестирование

Запустите сервер:

```bash
uvicorn app.main:app --reload
```

**Отправить сообщение в чат:**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat/thread-1/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What makes a strong thesis statement?"}' | python -m json.tool
```

**Продолжить разговор (история сохраняется автоматически):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat/thread-1/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you give me an example?"}' | python -m json.tool
```

**Получить историю:**

```bash
curl -s http://localhost:8000/api/v1/chat/thread-1/history | python -m json.tool
```

**Очистить разговор:**

```bash
curl -s -X DELETE http://localhost:8000/api/v1/chat/thread-1 | python -m json.tool
```

**Оценка с обсуждением (первое сообщение с работой):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat/assess-1/assess \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Please assess my essay",
    "student_work": "Climate change is the defining challenge of our generation. Rising global temperatures threaten ecosystems, economies, and human health. According to the IPCC 2023 report, we must reduce emissions by 45% by 2030 to limit warming to 1.5°C.",
    "rubric_id": "essay_default"
  }' | python -m json.tool
```

**Обсудить оценку (последующие сообщения без student_work):**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat/assess-1/assess \
  -H "Content-Type: application/json" \
  -d '{"message": "Why did I lose points? How can I improve?"}' | python -m json.tool
```

**WebSocket чат (установите `wscat`: `npm i -g wscat`):**

```bash
wscat -c ws://localhost:8000/api/v1/chat/ws/ws-thread-1
```

После подключения отправляйте JSON-сообщения:

```
> {"message": "What is prompt engineering?"}
< {"type":"token","data":"Prompt"}
< {"type":"token","data":" engineering"}
< {"type":"token","data":" is"}
...
< {"type":"end","data":{"message":"Prompt engineering is..."}}
```

---

## Чеклист самопроверки

- [ ] Почему LLM API stateless, и как это влияет на архитектуру чат-приложения?
- [ ] В чём разница между `HumanMessage`, `AIMessage` и `SystemMessage`? Какую роль каждый играет?
- [ ] Сравни buffer, window и summary memory: стоимость, качество контекста, сложность реализации.
- [ ] Зачем `MessagesPlaceholder` вместо f-string конкатенации истории в промпт?
- [ ] Что делает `Annotated[list, add_messages]`? Чем отличается от `Annotated[list, operator.add]`?
- [ ] Как `thread_id` изолирует разговоры в LangGraph с MemorySaver?
- [ ] WebSocket vs SSE: когда какой протокол выбрать для LLM-приложения?
- [ ] Что произойдёт, если checkpointer не указан при `compile()`? Будет ли сохраняться история между вызовами?

---

## Частые ошибки

### 1. Забыть про рост контекста

```python
messages.append(new_message)
result = llm.invoke(messages)
```

Через 50 сообщений контекст = 100k+ токенов. Стоимость одного вызова превышает $0.30. Используйте window или summary memory:

```python
if len(messages) > WINDOW_SIZE:
    summary = await summarize(messages[:-WINDOW_SIZE])
    messages = [SystemMessage(content=summary)] + messages[-WINDOW_SIZE:]
```

### 2. Потеря ролей при склейке истории

```python
history = "\n".join([f"{m.type}: {m.content}" for m in messages])
```

Модель не может надёжно парсить текстовую разметку ролей. Используйте `MessagesPlaceholder`, который сохраняет типизированную структуру:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

### 3. Без thread_id — все пользователи в одном разговоре

```python
result = await app.ainvoke(input)
```

Без `thread_id` граф использует единое состояние. Все пользователи видят сообщения друг друга. Всегда передавайте уникальный `thread_id`:

```python
config = {"configurable": {"thread_id": f"user-{user_id}-{session_id}"}}
result = await app.ainvoke(input, config)
```

### 4. SystemMessage в state вместо node

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]

await graph.ainvoke({
    "messages": [
        SystemMessage(content="You are an assessor"),
        HumanMessage(content="Hi"),
    ]
}, config)
```

System message добавляется в историю при каждом вызове — через 5 ходов в списке 5 одинаковых системных сообщений. Правильно: добавлять system message **внутри node-функции**, вне персистируемого state:

```python
async def respond(state):
    all_messages = [SystemMessage(content="...")] + state["messages"]
    response = await llm.ainvoke(all_messages)
    return {"messages": [response]}
```

### 5. Блокирующие вызовы в WebSocket handler

```python
@router.websocket("/ws/{thread_id}")
async def ws_chat(websocket: WebSocket, thread_id: str):
    result = graph.invoke(input, config)
```

`invoke` (не `ainvoke`) блокирует event loop. Все остальные WebSocket-соединения "замерзают" на время вызова LLM (2-30 секунд). Всегда используйте async-варианты: `ainvoke`, `astream`, `astream_events`.

---

## Что читать дальше

- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) — checkpointing и state management
- [LangGraph Message Handling](https://langchain-ai.github.io/langgraph/how-tos/manage-conversation-history/) — управление историей в графе
- [Chat History](https://python.langchain.com/docs/how_to/message_history/) — управление историей в LangChain
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/) — WebSocket в FastAPI
- [add_messages reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) — документация reducers в LangGraph

**Следующая тема:** [Тема 8: Observability](topic_08_observability.md) — как наблюдать за LLM в production.

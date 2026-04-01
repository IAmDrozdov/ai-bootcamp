# Тема 7: Conversational AI

> **Пререквизиты:** [Тема 6: LangGraph](topic_06_langgraph_agents.md)
> **Зависимости:** `langgraph`, `langchain-core`, `langchain-anthropic`

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

LangGraph решает фундаментальную проблему conversational AI: **где и как хранить состояние разговора между вызовами**.

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

**SSE** идеален, когда клиент отправляет один запрос и получает потоковый ответ. Клиент делает POST, сервер стримит токены через SSE, соединение закрывается.

**WebSocket** нужен для полноценного чата: клиент отправляет сообщения и получает ответы через одно постоянное соединение, без пересоздания HTTP-запросов. Жизненный цикл WebSocket:

1. **Handshake** — клиент отправляет HTTP-запрос с заголовком `Upgrade: websocket`, сервер отвечает `101 Switching Protocols`
2. **Open** — соединение установлено, обе стороны могут отправлять фреймы
3. **Message exchange** — JSON-сообщения в обе стороны
4. **Close** — любая сторона инициирует закрытие, отправляя close-фрейм

WebSocket даёт два преимущества для чат-приложений: (1) двунаправленность — пользователь и AI обмениваются сообщениями в реальном времени; (2) стриминг — ответ LLM приходит токен за токеном через то же соединение.

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

**Ограничения:** данные живут только пока жив процесс. Перезапуск = потеря всех разговоров. Для production используйте `SqliteSaver`, `PostgresSaver` или `AsyncPostgresSaver`.

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

## Практика

### Пример 1: Многоходовой диалог (buffer memory)

Простейший подход — накапливаем все сообщения в списке и отправляем полную историю при каждом вызове LLM.

```python
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

messages = [
    SystemMessage(content="You are a helpful academic assistant. Answer concisely."),
]

user_inputs = [
    "What makes a strong thesis statement?",
    "Can you give me an example about climate change?",
    "How would you improve that example?",
]

for user_text in user_inputs:
    messages.append(HumanMessage(content=user_text))
    response = llm.invoke(messages)
    messages.append(response)
    print(f"Human: {user_text}")
    print(f"AI: {response.content}\n")

print(f"Всего сообщений в истории: {len(messages)}")
print(f"Токенов в последнем ответе: {response.usage_metadata}")
```

---

### Пример 2: Window memory — скользящее окно

Хранить все сообщения, но отправлять в LLM только последние N. Ранний контекст теряется — зато стоимость и размер контекста фиксированы.

```python
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

SYSTEM = SystemMessage(content="You are a helpful assistant. Answer concisely.")
WINDOW_SIZE = 4

messages: list = []

user_inputs = [
    "My name is Alex.",
    "I study computer science.",
    "My favorite language is Python.",
    "I want to learn about LLMs.",
    "What should I start with?",
    "Do you remember my name?",
]

for user_text in user_inputs:
    messages.append(HumanMessage(content=user_text))

    window = messages[-WINDOW_SIZE:]
    to_send = [SYSTEM] + window

    response = llm.invoke(to_send)
    messages.append(response)

    print(f"Human: {user_text}")
    print(f"AI: {response.content}")
    print(f"  (окно: {len(window)} из {len(messages)} сообщений)\n")
```

---

### Пример 3: Summary memory — сжатие через LLM

Когда сообщений становится слишком много, старые сжимаются в краткое резюме. Контекст = системное сообщение + резюме + последние N сообщений.

```python
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

SYSTEM = SystemMessage(content="You are a helpful academic assistant.")
SUMMARIZE_THRESHOLD = 6
KEEP_RECENT = 2


def summarize_messages(msgs: list) -> str:
    text = "\n".join(f"{m.type}: {m.content}" for m in msgs)
    summary_response = llm.invoke([
        SystemMessage(
            content="Summarize this conversation in 2-3 sentences, "
            "preserving key facts and decisions."
        ),
        HumanMessage(content=text),
    ])
    return summary_response.content


messages: list = []
summary: str | None = None

user_inputs = [
    "Hi, I'm working on an essay about renewable energy.",
    "My thesis is that solar power will dominate by 2040.",
    "I have three supporting arguments.",
    "First, the cost of solar panels dropped 90% since 2010.",
    "Can you assess my thesis so far?",
    "How can I strengthen my first argument?",
]

for user_text in user_inputs:
    messages.append(HumanMessage(content=user_text))

    if len(messages) > SUMMARIZE_THRESHOLD:
        old = messages[:-KEEP_RECENT]
        summary = summarize_messages(old)
        messages = messages[-KEEP_RECENT:]
        print(f"  [Суммаризовано {len(old)} сообщений → {len(summary)} символов]\n")

    to_send = [SYSTEM]
    if summary:
        to_send.append(SystemMessage(content=f"Previous conversation summary:\n{summary}"))
    to_send.extend(messages)

    response = llm.invoke(to_send)
    messages.append(response)

    print(f"Human: {user_text}")
    print(f"AI: {response.content[:120]}...\n")
```

---

### Пример 4: MessagesPlaceholder в ChatPromptTemplate

`MessagesPlaceholder` вставляет типизированную историю в промпт, сохраняя роли. Это правильный способ передать историю — вместо склейки в строку.

```python
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assessor. Assessment context:\n{context}"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

chain = prompt | llm

context = "Rubric: Essay (clarity 30%, evidence 40%, structure 30%)"

result1 = chain.invoke({
    "context": context,
    "chat_history": [],
    "input": "Assess this thesis: Solar power will dominate energy by 2040.",
})
print(f"Ход 1: {result1.content[:150]}...\n")

history = [
    HumanMessage(content="Assess this thesis: Solar power will dominate energy by 2040."),
    result1,
]

result2 = chain.invoke({
    "context": context,
    "chat_history": history,
    "input": "Why did I lose points on evidence?",
})
print(f"Ход 2: {result2.content[:150]}...\n")

formatted = prompt.invoke({
    "context": context,
    "chat_history": history,
    "input": "How can I improve?",
})
print("Итоговые сообщения:")
for msg in formatted.messages:
    print(f"  {msg.__class__.__name__}: {msg.content[:80]}...")
```

---

### Пример 5: LangGraph — stateful диалог с checkpointing

LangGraph + `MemorySaver` автоматически сохраняет и восстанавливает историю по `thread_id`. Не нужно вручную управлять списком сообщений — checkpointer делает это за вас.

```python
import asyncio
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


async def respond(state: ConversationState):
    all_messages = [
        SystemMessage(content="You are a helpful academic assistant. Answer concisely.")
    ] + state["messages"]
    response = await llm.ainvoke(all_messages)
    return {"messages": [response]}


graph = StateGraph(ConversationState)
graph.add_node("respond", respond)
graph.add_edge(START, "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=MemorySaver())


async def main():
    config_alice = {"configurable": {"thread_id": "alice-session"}}
    config_bob = {"configurable": {"thread_id": "bob-session"}}

    r1 = await app.ainvoke(
        {"messages": [HumanMessage(content="Hi, I'm Alice. What makes a good thesis?")]},
        config=config_alice,
    )
    print(f"Alice ход 1: {r1['messages'][-1].content[:100]}...\n")

    r2 = await app.ainvoke(
        {"messages": [HumanMessage(content="Hi, I'm Bob. Explain window memory.")]},
        config=config_bob,
    )
    print(f"Bob ход 1: {r2['messages'][-1].content[:100]}...\n")

    r3 = await app.ainvoke(
        {"messages": [HumanMessage(content="Can you give me an example?")]},
        config=config_alice,
    )
    print(f"Alice ход 2: {r3['messages'][-1].content[:100]}...\n")

    state = await app.aget_state(config_alice)
    print(f"Alice — история: {len(state.values['messages'])} сообщений")
    for m in state.values["messages"]:
        print(f"  {m.__class__.__name__}: {m.content[:60]}...")

    state_bob = await app.aget_state(config_bob)
    print(f"\nBob — история: {len(state_bob.values['messages'])} сообщений")


asyncio.run(main())
```

---

### Пример 6: LangGraph — условная маршрутизация в диалоге

Граф с двумя путями: при первом вызове (оценка ещё не проведена) — узел `assess`; при последующих — узел `discuss`. `MemorySaver` запоминает, что оценка уже была.

```python
import asyncio
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

ASSESSMENT_SYSTEM = """\
You are an expert academic assessor. Evaluate the student work below \
thoroughly, providing scores and specific feedback.

## Student Work
{student_work}"""

DISCUSSION_SYSTEM = """\
You are an expert academic assessor discussing a student's work.

## Student Work
{student_work}

## Completed Assessment
{assessment}

Answer questions about the assessment. Be constructive and specific."""


class AssessmentState(TypedDict):
    messages: Annotated[list, add_messages]
    student_work: str
    assessment: str


llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


async def assess(state: AssessmentState):
    system = SystemMessage(
        content=ASSESSMENT_SYSTEM.format(student_work=state["student_work"])
    )
    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response], "assessment": response.content}


async def discuss(state: AssessmentState):
    system = SystemMessage(
        content=DISCUSSION_SYSTEM.format(
            student_work=state["student_work"],
            assessment=state["assessment"],
        )
    )
    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}


def route(state: AssessmentState) -> str:
    if not state.get("assessment"):
        return "assess"
    return "discuss"


graph = StateGraph(AssessmentState)
graph.add_node("assess", assess)
graph.add_node("discuss", discuss)
graph.add_conditional_edges(START, route)
graph.add_edge("assess", END)
graph.add_edge("discuss", END)

app = graph.compile(checkpointer=MemorySaver())


async def main():
    config = {"configurable": {"thread_id": "assessment-session-1"}}

    r1 = await app.ainvoke(
        {
            "messages": [HumanMessage(content="Please assess my essay.")],
            "student_work": (
                "Climate change is the defining challenge of our generation. "
                "Rising temperatures threaten ecosystems and economies. "
                "According to the IPCC 2023 report, we must reduce emissions "
                "by 45% by 2030."
            ),
            "assessment": "",
        },
        config=config,
    )
    print(f"Оценка: {r1['messages'][-1].content[:200]}...\n")

    r2 = await app.ainvoke(
        {"messages": [HumanMessage(content="Why did I lose points? How can I improve?")]},
        config=config,
    )
    print(f"Обсуждение: {r2['messages'][-1].content[:200]}...\n")

    state = await app.aget_state(config)
    print(f"Есть оценка: {bool(state.values.get('assessment'))}")
    print(f"Всего сообщений: {len(state.values['messages'])}")


asyncio.run(main())
```

---

## Чеклист самопроверки

- [ ] Почему LLM API stateless, и как это влияет на архитектуру чат-приложения?
- [ ] В чём разница между `HumanMessage`, `AIMessage` и `SystemMessage`? Какую роль каждый играет?
- [ ] Сравни buffer, window и summary memory: стоимость, качество контекста, сложность реализации.
- [ ] Зачем `MessagesPlaceholder` вместо f-string конкатенации истории в промпт?
- [ ] Что делает `Annotated[list, add_messages]`? Чем отличается от `Annotated[list, operator.add]`?
- [ ] Как `thread_id` изолирует разговоры в LangGraph с MemorySaver?
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

---

## Что читать дальше

- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) — checkpointing и state management
- [LangGraph Message Handling](https://langchain-ai.github.io/langgraph/how-tos/manage-conversation-history/) — управление историей в графе
- [Chat History](https://python.langchain.com/docs/how_to/message_history/) — управление историей в LangChain
- [add_messages reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) — документация reducers в LangGraph

**Следующая тема:** [Тема 8: Observability](topic_08_observability.md) — как наблюдать за LLM в production.

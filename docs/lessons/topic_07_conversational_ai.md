# Тема 7: Conversational AI

> **Пререквизиты:** [Тема 6: LangGraph](topic_06_langgraph_agents.md)
> **Где в проекте:** `app/graph/`, `app/api/v1/`
> **Зависимости:** `langgraph`, `langchain-core`

---

## Теория

### 1. LLM не помнит предыдущие сообщения

Каждый вызов LLM API — **stateless**. Модель не "помнит" предыдущий разговор. Чтобы вести диалог, нужно **каждый раз** отправлять полную историю сообщений:

```
Вызов 1: [system, human("Оцени мою работу")]          → AI("Оценка: 75/100...")
Вызов 2: [system, human("Оцени"), AI("75/100"), human("Почему не 80?")] → AI("Потому что...")
Вызов 3: [system, human, AI, human, AI, human("А как улучшить?")] → AI("Рекомендации...")
```

С каждым обменом контекст **растёт**. Через 20 сообщений ты можешь превысить лимит контекста или платить за повторную обработку тысяч токенов.

### 2. Стратегии управления памятью

#### Buffer memory — хранить всё

Простейший подход: сохраняем все сообщения, отправляем всё каждый раз.

```
Плюсы: полный контекст, ничего не теряется
Минусы: растущая стоимость, лимит контекста, "lost in the middle"
Когда: короткие диалоги (< 20 сообщений)
```

#### Window memory — скользящее окно

Хранить последние N сообщений. Старые — удаляются.

```
Плюсы: предсказуемая стоимость, фиксированный размер контекста
Минусы: теряется ранний контекст ("О чём мы говорили 10 минут назад?")
Когда: длинные сессии, где важен недавний контекст
```

#### Summary memory — сжатие через LLM

Старые сообщения сжимаются в краткое резюме через LLM. Резюме + последние N сообщений отправляются в контекст.

```
Плюсы: сохраняется суть всего разговора при фиксированном размере
Минусы: дополнительный вызов LLM для суммаризации, потеря деталей
Когда: длинные диалоги, где важна тема, но не точные формулировки
```

### 3. MessagesPlaceholder — вставка истории

`MessagesPlaceholder` — специальный элемент в `ChatPromptTemplate`, который принимает **список сообщений** и вставляет их в промпт с сохранением ролей:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assessment discussion assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
```

Без `MessagesPlaceholder` пришлось бы склеивать историю в одну строку, теряя разделение по ролям. Модель бы не могла отличить свои ответы от вопросов пользователя.

### 4. LangGraph для диалога

LangGraph идеален для conversational AI, потому что:
- **Stateful**: state хранит историю сообщений между вызовами
- **Checkpointing**: история персистится (MemorySaver, SqliteSaver)
- **Thread-based**: каждый разговор — отдельный thread_id

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ConversationState(TypedDict):
    messages: Annotated[list, add]

graph = StateGraph(ConversationState)
graph.add_node("respond", respond_node)
graph.add_edge(START, "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "conversation-1"}}
result = await app.ainvoke({"messages": [HumanMessage("Why 7/10?")]}, config)
result = await app.ainvoke({"messages": [HumanMessage("How to improve?")]}, config)
```

Второй вызов **автоматически** получает историю из первого через checkpointer.

### 5. WebSocket vs SSE для чата

| | SSE | WebSocket |
|---|---|---|
| Направление | Только сервер → клиент | Двунаправленное |
| Подключение | Новый запрос на каждое сообщение | Одно постоянное соединение |
| Для LLM чата | Стриминг ответа | Полноценный real-time чат |
| Сложность | Простая (HTTP) | Средняя (WS протокол) |

Для полноценного чата WebSocket лучше: клиент отправляет сообщения и получает ответы через одно соединение, без пересоздания HTTP-запросов.

---

## Практические задания

### Задание 1: Обсуждение оценки

**Цель:** построить conversation graph для обсуждения выставленной оценки.

**Файлы:** `app/graph/conversation_graph.py`, `experiments/t7_conversation.py`

**Критерии успеха:**
- После оценки преподаватель задаёт вопросы ("Почему 7/10?")
- Агент отвечает со ссылками на фрагменты работы
- История сохраняется между вызовами через checkpointer

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a conversation graph for discussing assessment results.

1. Create app/graph/conversation_graph.py:
   - State: messages (Annotated[list, add]), student_work (str), assessment (str | None)
   - Node "respond": 
     - If no assessment in state: run assessment first, save to state
     - If assessment exists: answer questions about it, referencing specific parts of student_work
   - System prompt should include the student_work and assessment context
   - Use MessagesPlaceholder for chat history
   - Compile with MemorySaver checkpointer
   - Function build_conversation_graph(llm) -> CompiledGraph

2. Create experiments/t7_conversation.py:
   - Initialize conversation with student work
   - Turn 1: "Please assess my essay" → agent provides assessment
   - Turn 2: "Why did I get 12/25 on Evidence?" → agent explains with quotes from the essay
   - Turn 3: "How can I improve my thesis?" → agent gives specific advice
   - Turn 4: "What if I added 3 more citations?" → agent discusses potential impact
   - Each turn uses the same thread_id
   - Print full conversation history at the end

Run with: python -m experiments.t7_conversation
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/graph/conversation_graph.py:

1. STATEFULNESS: Does the graph maintain conversation history across calls?
2. CONTEXT: Does the system prompt include the student work and assessment?
3. CHECKPOINTER: Is MemorySaver used for persistence?
4. MESSAGES: Is messages field using Annotated[list, add] reducer?
5. REFERENCES: Does the agent reference specific parts of the essay in answers?
6. THREAD_ID: Are calls using consistent thread_id?

Common mistakes:
- Not persisting assessment result in state (re-assessed each turn)
- Not including student_work in the system prompt (agent can't reference specifics)
- messages reducer using replace instead of add (loses history)
- Forgetting thread_id in config (no persistence between calls)

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Summarization memory

**Цель:** реализовать сжатие длинных диалогов для экономии контекста.

**Файлы:** `experiments/t7_summary_memory.py`

**Критерии успеха:**
- После 10 сообщений старые сжимаются в резюме через LLM
- Резюме + последние 5 сообщений отправляются в контекст
- Агент "помнит" суть ранних сообщений, но не тратит контекст

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t7_summary_memory.py implementing summarization memory.

1. Implement summarization logic:
   - Function summarize_messages(messages: list, llm) -> str:
     Takes old messages, asks LLM to create a brief summary
   - Summary prompt: "Summarize this conversation, preserving key decisions, scores, and topics discussed."

2. Implement memory management:
   - Keep track of all messages
   - When message count > 10: summarize oldest messages (all except last 5)
   - Replace old messages with a SystemMessage containing the summary
   - New context = [system_prompt, summary_message, last_5_messages, new_human_message]

3. Demonstrate with a 15+ turn conversation:
   - Turns 1-5: Initial assessment and questions
   - Turn 6-10: Discussion about specific criteria
   - Turn 11-15: Questions about improvement strategies
   - After turn 10: summarization triggers
   - Show: total tokens before/after summarization
   - Verify: agent still knows what was discussed in early turns

4. Compare token usage: full history vs summarized

Run with: python -m experiments.t7_summary_memory
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t7_summary_memory.py:

1. SUMMARIZATION: Is there an LLM call that summarizes old messages?
2. WINDOW: Are recent messages preserved (not summarized)?
3. TOKEN SAVINGS: Is token count compared before/after summarization?
4. QUALITY: Does the agent still know earlier context after summarization?
5. TRIGGER: Is summarization triggered at the right time (after N messages)?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: WebSocket

**Цель:** добавить WebSocket endpoint для real-time чата.

**Файлы:** `app/api/v1/chat.py`, `experiments/t7_websocket.py`

**Критерии успеха:**
- WebSocket endpoint `/api/v1/chat/ws/{thread_id}`
- Клиент отправляет сообщения, получает ответы в реальном времени
- История сохраняется по thread_id
- Streaming: ответ приходит токен за токеном

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Add a WebSocket chat endpoint.

1. Create app/api/v1/chat.py:
   - WebSocket endpoint: @router.websocket("/ws/{thread_id}")
   - On connect: load/create conversation graph with thread_id
   - On message: 
     a. Parse JSON: {"message": "user text"}
     b. Run conversation graph with the new message
     c. Stream response tokens back via WebSocket
     d. Send final JSON: {"type": "end", "data": {"message": full_response}}
   - On disconnect: cleanup

2. Register the router in app/api/router.py

3. Create experiments/t7_websocket.py — a CLI WebSocket client:
   - Connect to ws://localhost:8000/api/v1/chat/ws/test-1
   - Interactive loop: type message → send → print streaming response
   - Handle Ctrl+C gracefully

4. Dependencies: FastAPI has built-in WebSocket support, no extra packages needed

Run server: uvicorn app.main:app --reload
Run client: python -m experiments.t7_websocket
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review app/api/v1/chat.py:

1. WEBSOCKET: Is @router.websocket used correctly?
2. THREAD_ID: Is conversation state maintained per thread_id?
3. STREAMING: Are response tokens sent incrementally (not as one big message)?
4. JSON FORMAT: Are messages properly structured JSON?
5. ERROR HANDLING: What happens on disconnect, invalid JSON, LLM error?
6. CLEANUP: Are resources cleaned up on disconnect?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Почему LLM не помнит предыдущие сообщения? Как это решается?
- [ ] Сравни buffer, window и summary memory: плюсы, минусы, когда какой.
- [ ] Зачем MessagesPlaceholder? Почему не просто строка с историей?
- [ ] Как LangGraph checkpointer обеспечивает персистентность диалога?
- [ ] WebSocket vs SSE для чата: когда что использовать?

---

## Частые ошибки

### 1. Забыть про рост контекста

```python
# Плохо: через 50 сообщений контекст = 100k токенов
messages.append(new_message)
result = llm.invoke(messages)

# Хорошо: ограничить + суммаризировать
if len(messages) > WINDOW_SIZE:
    summary = await summarize(messages[:-WINDOW_SIZE])
    messages = [SystemMessage(summary)] + messages[-WINDOW_SIZE:]
```

### 2. Потеря ролей при склейке истории

```python
# Плохо: всё в одну строку — модель не понимает, кто что сказал
history = "\n".join([f"{m.type}: {m.content}" for m in messages])

# Хорошо: MessagesPlaceholder сохраняет роли
prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

### 3. Без thread_id — все пользователи в одном разговоре

```python
# Плохо: один глобальный граф на всех
result = await app.ainvoke(input)

# Хорошо: thread_id изолирует разговоры
config = {"configurable": {"thread_id": f"user-{user_id}"}}
result = await app.ainvoke(input, config)
```

---

## Что читать дальше

- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) — checkpointing
- [Chat History](https://python.langchain.com/docs/how_to/message_history/) — управление историей
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/) — WebSocket в FastAPI

**Следующая тема:** [Тема 8: Observability](topic_08_observability.md) — как наблюдать за LLM в production.

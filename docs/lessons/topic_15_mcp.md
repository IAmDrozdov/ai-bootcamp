# Тема 15: MCP (Model Context Protocol) — стандарт интеграции LLM с инструментами

> **Пререквизиты:** [Тема 11 (Tool Use)](topic_11_tool_use.md), рекомендуется [Тема 6 (LangGraph)](topic_06_langgraph_agents.md)
> **Что добавляем в проект:** `mcp_servers/rubrics_server.py`, `mcp_servers/assessment_server.py`, `app/services/mcp_client.py`, `app/api/v1/mcp_endpoints.py`, `app/schemas/mcp.py`
> **Зависимости:** `mcp[cli]`, `langchain-mcp-adapters`, `httpx`

---

## Теория

### 1. Что такое MCP и зачем он нужен

Каждый LLM-провайдер определяет свой формат описания инструментов. У Anthropic — `tool_use` блоки с `input_schema`. У OpenAI — `functions` (потом `tools`) с `parameters`. У Google — `function_declarations`. Когда приложение интегрируется с внешними сервисами — базами данных, API, файловой системой — приходится писать адаптеры под каждый провайдер и каждый сервис отдельно.

Если у тебя 3 LLM-провайдера и 5 внешних сервисов — это 15 интеграций. Добавляешь шестой сервис — ещё 3 интеграции. Проблема масштабируется как M × N.

**MCP (Model Context Protocol)** — открытый протокол, выпущенный Anthropic в ноябре 2024 года. Его цель — стандартизировать способ подключения LLM к данным и инструментам. Вместо M × N интеграций получается M + N: каждый провайдер реализует MCP-клиент, каждый сервис — MCP-сервер.

Аналогия: USB для AI-приложений. До USB каждое устройство имело свой разъём — принтер, сканер, камера, клавиатура подключались по-разному. USB стандартизировал интерфейс: один порт, любое устройство. MCP делает то же самое для LLM: один протокол, любой инструмент.

Кто уже использует MCP:

| Продукт | Роль | Что делает |
|---------|------|------------|
| Claude Desktop | Host | Подключается к локальным MCP-серверам через stdio |
| Cursor | Host | Использует MCP для доступа к codebase tools |
| VS Code (Copilot) | Host | MCP-серверы как расширения возможностей |
| Zed | Host | Встроенная поддержка MCP |
| Windsurf | Host | MCP для AI-ассистента |

**MCP vs OpenAI function calling vs LangChain tools:**

| Характеристика | OpenAI Function Calling | LangChain Tools | MCP |
|---|---|---|---|
| Scope | Один провайдер (OpenAI API) | Один framework (LangChain) | Универсальный протокол |
| Формат | JSON Schema в API-запросе | Python-классы с `_run()` | JSON-RPC 2.0 поверх транспорта |
| Транспорт | HTTP (API) | In-process | stdio, HTTP+SSE |
| Discovery | Нет (tools передаются в запросе) | Нет (tools передаются в коде) | Да (`tools/list`, `resources/list`) |
| Серверная сторона | Нет (tools на стороне клиента) | Нет | Да (отдельный процесс/сервис) |
| Межпроцессное взаимодействие | Нет | Нет | Да |
| Стандарт | Проприетарный | Проприетарный | Открытый (MIT) |

Ключевое отличие MCP — **отделение серверной логики от клиента**. MCP-сервер — это отдельный процесс, который может быть написан на любом языке, запущен где угодно и использоваться любым MCP-совместимым хостом. OpenAI function calling и LangChain tools существуют только внутри одного процесса.

### 2. Архитектура MCP: Host, Client, Server

MCP определяет три роли:

**Host** — приложение верхнего уровня, с которым взаимодействует пользователь. Примеры: Claude Desktop, Cursor, твой FastAPI-сервер. Host управляет жизненным циклом клиентов, решает какие серверы подключать, контролирует политики безопасности.

**Client** — компонент внутри Host, который поддерживает 1:1 соединение с одним MCP-сервером. Говорит на JSON-RPC 2.0. Один Host может создать несколько клиентов для подключения к разным серверам одновременно.

**Server** — процесс или сервис, предоставляющий инструменты, ресурсы и промпты через стандартный MCP-интерфейс. Может быть локальным (subprocess через stdio) или удалённым (HTTP).

```
┌─────────────────────────────────────────┐
│              Host (FastAPI app)          │
│                                         │
│  ┌──────────┐  ┌──────────┐             │
│  │ Client A │  │ Client B │             │
│  └────┬─────┘  └────┬─────┘             │
│       │              │                  │
└───────┼──────────────┼──────────────────┘
        │              │
   JSON-RPC 2.0   JSON-RPC 2.0
   (stdio)        (HTTP+SSE)
        │              │
  ┌─────┴──────┐ ┌─────┴──────┐
  │  Server A  │ │  Server B  │
  │ (rubrics)  │ │ (assess)   │
  └────────────┘ └────────────┘
```

**Жизненный цикл соединения:**

1. **Initialize** — клиент отправляет `initialize` request с информацией о своих capabilities. Сервер отвечает своими capabilities (какие примитивы поддерживает: tools, resources, prompts).
2. **Initialized** — клиент отправляет `initialized` notification, подтверждая что handshake завершён.
3. **Operation** — нормальная работа: вызовы tools, чтение resources, получение prompts.
4. **Shutdown** — корректное завершение: клиент вызывает `close()`, ресурсы освобождаются.

**Capability negotiation** — важный шаг. Не каждый сервер поддерживает все три примитива. При инициализации сервер сообщает, что именно он предоставляет:

```json
{
  "capabilities": {
    "tools": {"listChanged": true},
    "resources": {"subscribe": true, "listChanged": true},
    "prompts": {"listChanged": true}
  }
}
```

Клиент использует эту информацию, чтобы знать какие методы доступны. Если сервер не объявил `tools` — вызов `tools/list` вернёт ошибку.

### 3. Три примитива MCP

MCP определяет три типа данных, которые сервер может предоставить. Каждый примитив имеет свою семантику и паттерн использования.

#### Tools — функции для LLM

Tools — это функции, которые LLM может вызвать для выполнения действий или получения данных. Аналог LangChain tools и OpenAI functions, но стандартизированные через протокол.

Характеристики:
- **Model-controlled** — LLM сама решает, когда вызвать tool, на основе описания и контекста разговора
- Описываются через JSON Schema (как OpenAI functions)
- Могут иметь side effects (запись в БД, вызов API)
- Результат возвращается в контекст LLM для формирования ответа

```json
{
  "name": "assess_work",
  "description": "Assess a student's work against a rubric and return detailed scores",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The student's work to assess"
      },
      "rubric": {
        "type": "string",
        "description": "Name of the rubric to use for assessment"
      }
    },
    "required": ["text", "rubric"]
  }
}
```

#### Resources — данные для чтения

Resources — данные, доступные по URI. В отличие от tools, resources — read-only и не вызываются LLM напрямую.

Характеристики:
- **Application-controlled** — приложение (не LLM) решает когда и какие ресурсы читать
- URI-адресация: `rubric://math/algebra`, `assessment://2024/student-123`
- Могут иметь MIME-тип: `text/plain`, `application/json`
- Могут быть статическими (файл) или динамическими (результат запроса к БД)

Типичные use cases:
- Предоставить рубрику как контекст для оценки
- Прочитать конфигурацию оценочной системы
- Получить пример хорошо оценённой работы

#### Prompts — шаблоны промптов

Prompts — параметризованные шаблоны промптов, предоставляемые сервером.

Характеристики:
- **User-controlled** — пользователь (не LLM, не код) выбирает промпт из каталога
- Принимают аргументы: `assess(student_work="...", rubric_name="essay")`
- Возвращают массив сообщений (`messages`) — готовый input для LLM
- Позволяют серверу инкапсулировать промпт-инжиниринг

| Примитив | Кто контролирует | Направление данных | Аналогия |
|---|---|---|---|
| Tool | LLM (model-controlled) | Вызов функции → результат | RPC / API call |
| Resource | Приложение (application-controlled) | URI → данные | GET запрос / файл |
| Prompt | Пользователь (user-controlled) | Аргументы → сообщения | Шаблон / макрос |

### 4. Транспорт: stdio vs Streamable HTTP

MCP не привязан к конкретному транспорту. Спецификация определяет два стандартных варианта.

#### stdio — для локальных серверов

При использовании stdio MCP-сервер запускается как дочерний процесс (subprocess). Общение происходит через стандартные потоки: клиент пишет JSON-RPC в stdin сервера, сервер отвечает через stdout.

```
Host process                    Server process
     │                              │
     │──── JSON-RPC request ───────>│ (stdin)
     │                              │
     │<─── JSON-RPC response ───────│ (stdout)
     │                              │
     │     (stderr → logs)          │
```

Преимущества:
- Не требует сетевого стека — нет портов, TLS, CORS
- Надёжный: ОС управляет потоками, нет проблем с reconnect
- Изоляция: каждый сервер — отдельный процесс с собственной памятью
- Простота: для локальной разработки — запустил и работает

Используется в: Claude Desktop, Cursor, Zed — все запускают MCP-серверы как subprocesses.

#### Streamable HTTP (SSE) — для удалённых серверов

Для серверов, работающих удалённо (другая машина, облако, Docker), используется HTTP-транспорт: клиент отправляет POST-запросы, сервер отвечает через Server-Sent Events (SSE).

```
Client                         Remote Server
  │                                  │
  │── POST /mcp (JSON-RPC) ────────>│
  │                                  │
  │<── SSE stream (responses) ──────│
  │                                  │
```

Преимущества:
- Работает через сеть — сервер может быть где угодно
- Поддерживает аутентификацию (OAuth, API keys в headers)
- Совместим с load balancers, proxies, firewalls
- Можно использовать существующую HTTP-инфраструктуру

**Когда что использовать:**

| Сценарий | Транспорт | Причина |
|---|---|---|
| Локальная разработка | stdio | Просто, надёжно, без настройки сети |
| CI/CD | stdio | Запуск как subprocess в pipeline |
| Production (monolith) | stdio | Сервер рядом с хостом |
| Production (микросервисы) | HTTP+SSE | Серверы на разных машинах |
| Облачный deployment | HTTP+SSE | Сервер за load balancer |
| Claude Desktop / Cursor | stdio | Стандарт для desktop apps |

### 5. Написание MCP-сервера на Python

Библиотека `mcp` предоставляет высокоуровневый API через класс `FastMCP` — его дизайн вдохновлён FastAPI: декораторы для регистрации tools, resources, prompts.

Установка:

```bash
pip install "mcp[cli]"
```

#### Минимальный сервер с tool, resource и prompt

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


@mcp.resource("config://app")
def get_config() -> str:
    return "max_score=100\npassing_score=60"


@mcp.prompt()
def review(code: str) -> str:
    return f"Review this code and suggest improvements:\n\n{code}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Разберём каждый элемент.

**`FastMCP("demo-server")`** — создание экземпляра сервера. Строка — имя сервера, которое видит клиент при инициализации.

**`@mcp.tool()`** — регистрация функции как MCP tool. `FastMCP` автоматически:
- Извлекает имя функции как имя tool
- Генерирует `inputSchema` из type hints аргументов
- Использует docstring как описание (или можно передать `description=`)
- Конвертирует return value в строку для ответа

**`@mcp.resource("config://app")`** — регистрация ресурса с фиксированным URI. Клиент может запросить `resources/read` с URI `config://app` и получить содержимое.

**`@mcp.prompt()`** — регистрация промпта. Аргументы функции становятся параметрами промпта. Клиент вызывает `prompts/get` с аргументами, получает массив сообщений.

**`mcp.run(transport="stdio")`** — запуск сервера. Для stdio — блокирует процесс, слушает stdin. Для SSE — запускает HTTP-сервер.

#### Параметризованные ресурсы (resource templates)

Ресурсы могут содержать переменные в URI:

```python
@mcp.resource("rubric://{name}")
def get_rubric(name: str) -> str:
    rubrics = {
        "essay": "Criteria: thesis, evidence, structure, grammar",
        "code": "Criteria: correctness, efficiency, readability, tests",
    }
    return rubrics.get(name, f"Rubric '{name}' not found")
```

Клиент запрашивает `rubric://essay` — сервер вызывает `get_rubric(name="essay")`.

#### Промпты с несколькими сообщениями

Промпт может возвращать список сообщений для мультитурного контекста:

```python
from mcp.server.fastmcp.prompts import base

@mcp.prompt()
def assess(student_work: str, rubric_name: str) -> list[base.Message]:
    return [
        base.UserMessage(
            content=f"Assess this work using the '{rubric_name}' rubric:\n\n{student_work}"
        ),
    ]
```

#### Запуск и тестирование

Запуск через CLI:

```bash
mcp dev mcp_servers/rubrics_server.py
```

Команда `mcp dev` запускает MCP Inspector — веб-интерфейс для тестирования сервера. Открывается браузер, где можно:
- Видеть список tools, resources, prompts
- Вызывать tools с произвольными аргументами
- Читать resources по URI
- Тестировать prompts

Для production:

```bash
mcp run mcp_servers/rubrics_server.py
```

### 6. Написание MCP-клиента на Python

Клиент подключается к серверу, получает список доступных примитивов и вызывает их.

#### Низкоуровневый клиент через stdio

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_servers/rubrics_server.py"],
)


async def main():
    async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} — {tool.description}")

            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"Resource: {resource.uri} — {resource.name}")

            result = await session.call_tool("search_rubrics", {"query": "essay"})
            print(f"Result: {result.content[0].text}")

            resource_content = await session.read_resource("rubric://essay")
            print(f"Resource: {resource_content.contents[0].text}")


asyncio.run(main())
```

Разберём lifecycle:

1. **`stdio_client(SERVER_PARAMS)`** — запускает subprocess с MCP-сервером, возвращает потоки чтения/записи. Это async context manager: при выходе subprocess завершается.

2. **`ClientSession(read, write)`** — создаёт сессию поверх потоков. Управляет JSON-RPC коммуникацией.

3. **`session.initialize()`** — handshake: отправляет `initialize`, получает capabilities, отправляет `initialized`.

4. **`session.list_tools()`** / **`session.list_resources()`** — discovery: узнаём что сервер предоставляет.

5. **`session.call_tool(name, args)`** — вызов tool с аргументами. Возвращает `CallToolResult` с массивом `content`.

6. **`session.read_resource(uri)`** — чтение ресурса по URI.

#### Подключение через SSE

```python
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8080/sse") as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await session.list_tools()
```

Единственное отличие — используется `sse_client` вместо `stdio_client`, и вместо `StdioServerParameters` передаётся URL.

### 7. Интеграция MCP + LangChain / LangGraph

Библиотека `langchain-mcp-adapters` — мост между MCP и LangChain. Она конвертирует MCP tools в LangChain-совместимые tools, которые можно использовать в LCEL chains и LangGraph агентах.

Установка:

```bash
pip install langchain-mcp-adapters
```

#### Загрузка tools из одного сервера

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

model = ChatAnthropic(model="claude-sonnet-4-20250514")

server_params = StdioServerParameters(
    command="python",
    args=["mcp_servers/rubrics_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        tools = await load_mcp_tools(session)

        agent = create_react_agent(model, tools)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Find rubrics about essays"}]}
        )
```

**`load_mcp_tools(session)`** — ключевая функция. Она:
1. Вызывает `session.list_tools()` на MCP-сервере
2. Для каждого MCP tool создаёт `langchain_core.tools.BaseTool`
3. Конвертирует `inputSchema` в Pydantic-совместимые аргументы
4. Оборачивает `session.call_tool()` в `_run()` / `_arun()` методы

Результат — список обычных LangChain tools, которые можно передать в `bind_tools()`, `create_react_agent()` или любой другой компонент LangChain.

#### MultiServerMCPClient — несколько серверов

Реальное приложение обычно использует несколько MCP-серверов: один для рубрик, другой для оценок, третий для аналитики. `MultiServerMCPClient` управляет подключениями к нескольким серверам.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-20250514")

async with MultiServerMCPClient(
    {
        "rubrics": {
            "command": "python",
            "args": ["mcp_servers/rubrics_server.py"],
            "transport": "stdio",
        },
        "assessment": {
            "command": "python",
            "args": ["mcp_servers/assessment_server.py"],
            "transport": "stdio",
        },
    }
) as client:
    tools = client.get_tools()

    agent = create_react_agent(model, tools)
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find the 'essay' rubric and assess this work: "
                    "'Climate change is caused by greenhouse gases...'",
                }
            ]
        }
    )
```

`MultiServerMCPClient` — async context manager, который:
1. Создаёт `ClientSession` для каждого сервера
2. Инициализирует все соединения
3. Собирает tools со всех серверов в единый список
4. При выходе корректно закрывает все соединения

Метод `get_tools()` возвращает объединённый список LangChain tools со всех серверов. Имена tools уникальны — если у двух серверов есть tool с одинаковым именем, нужно переименовать.

#### Использование в LCEL chain

MCP tools можно использовать не только в агентах, но и в обычных LCEL chains:

```python
from langchain_core.prompts import ChatPromptTemplate

tools = await load_mcp_tools(session)
llm_with_tools = model.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assessment assistant. Use tools to find rubrics and assess work."),
    ("human", "{input}"),
])

chain = prompt | llm_with_tools
result = await chain.ainvoke({"input": "What rubrics are available?"})
```

В этом случае LLM вернёт `AIMessage` с `tool_calls`, но вызывать tools придётся самостоятельно. Для автоматического вызова используй `create_react_agent`.

### 8. MCP в production

#### Безопасность

MCP-серверы имеют доступ к данным и могут выполнять действия. Безопасность — критически важна:

**Валидация входных данных:**

```python
@mcp.tool()
def get_rubric(name: str) -> str:
    if not name.isalnum() or len(name) > 50:
        return "Error: invalid rubric name"
    if name not in RUBRICS:
        return f"Error: rubric '{name}' not found"
    return RUBRICS[name]
```

**Ограничение доступа к ресурсам** — сервер не должен предоставлять доступ к произвольным файлам или данным. URI ресурсов должны быть ограничены:

```python
ALLOWED_RUBRICS = {"essay", "code", "math", "science"}

@mcp.resource("rubric://{name}")
def get_rubric_resource(name: str) -> str:
    if name not in ALLOWED_RUBRICS:
        raise ValueError(f"Access denied: rubric '{name}' not available")
    return load_rubric(name)
```

#### Error handling

MCP-сервер может упасть, зависнуть или вернуть ошибку. Клиент должен быть готов:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def safe_mcp_session(server_params):
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception as e:
        raise RuntimeError(f"MCP connection failed: {e}")
```

#### Тестирование

MCP предоставляет инструменты для тестирования:

1. **`mcp dev`** — запускает MCP Inspector, веб-интерфейс для ручного тестирования
2. **`mcp run`** — запускает сервер для программного тестирования
3. **Unit-тесты** — тестируйте функции сервера отдельно, без MCP-протокола
4. **Integration-тесты** — подключайтесь через `ClientSession` и вызывайте tools

```bash
mcp dev mcp_servers/rubrics_server.py
```

MCP Inspector откроется в браузере и покажет все tools, resources и prompts сервера. Можно вызвать любой tool с произвольными аргументами и увидеть результат.

#### Deployment

Для production MCP-серверы упаковываются в Docker:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY mcp_servers/ ./mcp_servers/
CMD ["python", "mcp_servers/rubrics_server.py"]
```

При использовании HTTP-транспорта сервер запускается как обычный HTTP-сервис. При stdio — host запускает subprocess.

---

## Справочник API

### FastMCP

**Описание:** Высокоуровневый класс для создания MCP-серверов. Вдохновлён дизайном FastAPI: декораторы для регистрации tools, resources, prompts. Управляет жизненным циклом сервера и JSON-RPC коммуникацией.

```python
from mcp.server.fastmcp import FastMCP

FastMCP(
    name: str,                           # имя сервера (видно клиенту при инициализации)
    instructions: str | None = None,     # инструкции для LLM, использующей этот сервер
    **settings: Any,                     # дополнительные настройки (host, port для SSE)
)
```

**Основные методы:**

| Метод | Описание |
|---|---|
| `tool(name=None, description=None)` | Декоратор: регистрирует функцию как MCP tool |
| `resource(uri, name=None, description=None, mime_type=None)` | Декоратор: регистрирует функцию как MCP resource |
| `prompt(name=None, description=None)` | Декоратор: регистрирует функцию как MCP prompt |
| `run(transport="stdio")` | Запуск сервера (`"stdio"` или `"sse"`) |

**Пример:**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("assessment-server", instructions="Server for student work assessment")

@mcp.tool(description="Search rubrics by keyword")
def search_rubrics(query: str) -> str:
    results = [r for r in RUBRICS if query.lower() in r.lower()]
    return "\n".join(results) if results else "No rubrics found"

mcp.run(transport="stdio")
```

---

### @mcp.tool()

**Описание:** Декоратор для регистрации функции как MCP tool. Автоматически извлекает схему аргументов из type hints, имя из имени функции (или параметра `name`), описание из docstring (или параметра `description`).

```python
@mcp.tool(
    name: str | None = None,        # имя tool (default: имя функции)
    description: str | None = None,  # описание для LLM (default: docstring)
)
```

**Правила конвертации:**

| Python type hint | JSON Schema type | Пример |
|---|---|---|
| `str` | `string` | `query: str` |
| `int` | `integer` | `limit: int` |
| `float` | `number` | `threshold: float` |
| `bool` | `boolean` | `verbose: bool` |
| `list[str]` | `array` (items: string) | `tags: list[str]` |
| `dict[str, Any]` | `object` | `metadata: dict` |

**Пример с типизированными аргументами:**

```python
@mcp.tool(description="Assess a student's work against a rubric")
def assess_work(text: str, rubric: str, strict: bool = False) -> str:
    mode = "strict" if strict else "standard"
    return f"Assessment ({mode}): {text[:50]}... using rubric '{rubric}'"
```

---

### @mcp.resource()

**Описание:** Декоратор для регистрации функции как MCP resource. Ресурс доступен по URI и возвращает данные (текст, JSON). Поддерживает шаблоны URI с переменными.

```python
@mcp.resource(
    uri: str,                          # URI ресурса, напр. "rubric://essay" или "rubric://{name}"
    name: str | None = None,           # человекочитаемое имя (default: имя функции)
    description: str | None = None,    # описание ресурса
    mime_type: str | None = None,      # MIME-тип: "text/plain", "application/json"
)
```

**Пример статического ресурса:**

```python
@mcp.resource("config://assessment", mime_type="application/json")
def get_assessment_config() -> str:
    import json
    return json.dumps({
        "max_score": 100,
        "passing_score": 60,
        "rubrics_available": ["essay", "code", "math"],
    })
```

**Пример параметризованного ресурса:**

```python
@mcp.resource("rubric://{name}", mime_type="text/plain")
def get_rubric(name: str) -> str:
    rubrics = {
        "essay": "Thesis: 25pts, Evidence: 25pts, Structure: 25pts, Grammar: 25pts",
        "code": "Correctness: 30pts, Efficiency: 25pts, Readability: 25pts, Tests: 20pts",
    }
    return rubrics.get(name, f"Rubric '{name}' not found")
```

---

### @mcp.prompt()

**Описание:** Декоратор для регистрации функции как MCP prompt. Промпт — параметризованный шаблон, возвращающий массив сообщений для LLM. Аргументы функции становятся параметрами промпта.

```python
@mcp.prompt(
    name: str | None = None,           # имя промпта (default: имя функции)
    description: str | None = None,    # описание для каталога
)
```

**Пример:**

```python
from mcp.server.fastmcp.prompts import base

@mcp.prompt(description="Generate assessment prompt for student work")
def assess(student_work: str, rubric_name: str, strictness: str = "standard") -> list[base.Message]:
    return [
        base.UserMessage(
            content=f"Assess this work using '{rubric_name}' rubric.\n"
            f"Strictness: {strictness}\n\n"
            f"Student work:\n{student_work}"
        )
    ]
```

---

### ClientSession

**Описание:** Низкоуровневый MCP-клиент, управляющий соединением с одним MCP-сервером. Используется внутри async context manager, работает поверх потоков чтения/записи.

```python
from mcp import ClientSession

ClientSession(
    read_stream: ReadStream,       # поток чтения (от транспорта)
    write_stream: WriteStream,     # поток записи (от транспорта)
)
```

**Основные методы:**

| Метод | Возвращает | Описание |
|---|---|---|
| `initialize()` | `InitializeResult` | Handshake: обмен capabilities |
| `list_tools()` | `ListToolsResult` | Список доступных tools |
| `call_tool(name, arguments)` | `CallToolResult` | Вызов tool с аргументами |
| `list_resources()` | `ListResourcesResult` | Список доступных resources |
| `read_resource(uri)` | `ReadResourceResult` | Чтение resource по URI |
| `list_prompts()` | `ListPromptsResult` | Список доступных prompts |
| `get_prompt(name, arguments)` | `GetPromptResult` | Получение промпта с аргументами |

**Пример:**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("search_rubrics", {"query": "essay"})
        print(result.content[0].text)
```

---

### StdioServerParameters

**Описание:** Конфигурация для запуска MCP-сервера как subprocess через stdio. Определяет команду, аргументы, переменные окружения.

```python
from mcp import StdioServerParameters

StdioServerParameters(
    command: str,                          # исполняемый файл ("python", "node", "uv")
    args: list[str] | None = None,        # аргументы командной строки
    env: dict[str, str] | None = None,    # переменные окружения
    cwd: str | None = None,               # рабочая директория
)
```

**Пример:**

```python
params = StdioServerParameters(
    command="python",
    args=["mcp_servers/rubrics_server.py"],
    env={"PYTHONPATH": "."},
)
```

---

### load_mcp_tools

**Описание:** Функция из `langchain-mcp-adapters`, которая загружает MCP tools из сессии и конвертирует их в LangChain-совместимые `BaseTool`. Позволяет использовать MCP tools в LCEL chains и LangGraph агентах.

```python
from langchain_mcp_adapters.tools import load_mcp_tools

await load_mcp_tools(
    session: ClientSession,     # инициализированная MCP-сессия
) -> list[BaseTool]             # список LangChain tools
```

**Пример:**

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

tools = await load_mcp_tools(session)
agent = create_react_agent(model, tools)
```

---

### MultiServerMCPClient

**Описание:** Async context manager из `langchain-mcp-adapters`, управляющий подключениями к нескольким MCP-серверам одновременно. Объединяет tools со всех серверов в единый список.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

MultiServerMCPClient(
    connections: dict[str, dict],    # имя сервера → параметры подключения
)
```

**Формат параметров подключения:**

```python
{
    "server_name": {
        "command": "python",              # для stdio
        "args": ["path/to/server.py"],    # для stdio
        "transport": "stdio",             # "stdio" или "sse"
        "url": "http://...",              # для SSE
    }
}
```

**Основные методы:**

| Метод | Возвращает | Описание |
|---|---|---|
| `get_tools()` | `list[BaseTool]` | Объединённый список LangChain tools со всех серверов |

**Пример:**

```python
async with MultiServerMCPClient(
    {
        "rubrics": {
            "command": "python",
            "args": ["mcp_servers/rubrics_server.py"],
            "transport": "stdio",
        },
        "assessment": {
            "command": "python",
            "args": ["mcp_servers/assessment_server.py"],
            "transport": "stdio",
        },
    }
) as client:
    all_tools = client.get_tools()
    agent = create_react_agent(model, all_tools)
```

---

## Практика: MCP-серверы + роутер `/api/v1/mcp`

В этой практике мы создадим два MCP-сервера (рубрики и оценки), клиент для подключения к ним и FastAPI роутер, который использует MCP tools через LangChain агента.

### Шаг 1. Схемы — `app/schemas/mcp.py`

Определим Pydantic-модели для запросов и ответов MCP-эндпоинтов.

```python
from pydantic import BaseModel, Field


class MCPToolInfo(BaseModel):
    name: str = Field(description="Tool name")
    description: str | None = Field(default=None, description="Tool description")
    input_schema: dict = Field(default_factory=dict, description="JSON Schema for tool inputs")


class MCPToolsResponse(BaseModel):
    server: str = Field(description="MCP server name")
    tools: list[MCPToolInfo] = Field(default_factory=list)


class MCPAssessRequest(BaseModel):
    student_work: str = Field(description="Student work to assess")
    rubric_name: str = Field(default="essay", description="Rubric name to use")


class MCPAssessResponse(BaseModel):
    result: str = Field(description="Assessment result from LLM agent")
    tools_used: list[str] = Field(default_factory=list, description="Names of MCP tools invoked")


class MCPResourceRequest(BaseModel):
    uri: str = Field(description="MCP resource URI, e.g. rubric://essay")


class MCPResourceResponse(BaseModel):
    uri: str = Field(description="Requested URI")
    content: str = Field(description="Resource content")
    mime_type: str | None = Field(default=None, description="MIME type of the resource")


class MCPMultiServerRequest(BaseModel):
    query: str = Field(description="User query for the multi-server agent")


class MCPMultiServerResponse(BaseModel):
    result: str = Field(description="Agent response")
    tools_used: list[str] = Field(default_factory=list)
    servers_used: list[str] = Field(default_factory=list, description="MCP servers that provided tools")
```

### Шаг 2. MCP Server: Рубрики — `mcp_servers/rubrics_server.py`

Сервер предоставляет tools для поиска и получения рубрик, resources для прямого доступа к данным рубрик, и prompt для формирования оценочных запросов.

```python
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP(
    "rubrics-server",
    instructions="MCP server for managing assessment rubrics. "
    "Provides tools to search and retrieve rubrics, "
    "resources for direct rubric access, "
    "and prompts for assessment generation.",
)

RUBRICS = {
    "essay": {
        "name": "Essay Assessment Rubric",
        "criteria": [
            {"name": "Thesis & Argument", "max_score": 25, "weight": 0.25,
             "description": "Clear thesis statement, logical argument development"},
            {"name": "Evidence & Support", "max_score": 25, "weight": 0.25,
             "description": "Use of relevant evidence, proper citations"},
            {"name": "Structure & Organization", "max_score": 25, "weight": 0.25,
             "description": "Clear introduction, body, conclusion; logical flow"},
            {"name": "Grammar & Style", "max_score": 25, "weight": 0.25,
             "description": "Correct grammar, appropriate academic style"},
        ],
    },
    "code": {
        "name": "Code Review Rubric",
        "criteria": [
            {"name": "Correctness", "max_score": 30, "weight": 0.30,
             "description": "Code produces correct output for all test cases"},
            {"name": "Efficiency", "max_score": 25, "weight": 0.25,
             "description": "Optimal time and space complexity"},
            {"name": "Readability", "max_score": 25, "weight": 0.25,
             "description": "Clear naming, consistent formatting, documentation"},
            {"name": "Testing", "max_score": 20, "weight": 0.20,
             "description": "Comprehensive unit tests, edge cases covered"},
        ],
    },
    "math": {
        "name": "Math Problem Rubric",
        "criteria": [
            {"name": "Understanding", "max_score": 25, "weight": 0.25,
             "description": "Demonstrates understanding of the problem"},
            {"name": "Method", "max_score": 30, "weight": 0.30,
             "description": "Appropriate method chosen and correctly applied"},
            {"name": "Calculation", "max_score": 25, "weight": 0.25,
             "description": "Accurate calculations, correct final answer"},
            {"name": "Presentation", "max_score": 20, "weight": 0.20,
             "description": "Clear steps shown, proper mathematical notation"},
        ],
    },
    "science": {
        "name": "Science Lab Report Rubric",
        "criteria": [
            {"name": "Hypothesis", "max_score": 20, "weight": 0.20,
             "description": "Clear, testable hypothesis based on background research"},
            {"name": "Methodology", "max_score": 25, "weight": 0.25,
             "description": "Detailed procedure, controlled variables, reproducibility"},
            {"name": "Data Analysis", "max_score": 30, "weight": 0.30,
             "description": "Accurate data collection, appropriate statistical analysis"},
            {"name": "Conclusion", "max_score": 25, "weight": 0.25,
             "description": "Evidence-based conclusion, addresses hypothesis"},
        ],
    },
}


@mcp.tool(description="Search rubrics by keyword. Returns matching rubric names and descriptions.")
def search_rubrics(query: str) -> str:
    query_lower = query.lower()
    results = []
    for rubric_id, rubric_data in RUBRICS.items():
        rubric_text = json.dumps(rubric_data, ensure_ascii=False).lower()
        if query_lower in rubric_id or query_lower in rubric_text:
            criteria_names = [c["name"] for c in rubric_data["criteria"]]
            max_total = sum(c["max_score"] for c in rubric_data["criteria"])
            results.append(
                f"- {rubric_id}: {rubric_data['name']} "
                f"(max score: {max_total}, "
                f"criteria: {', '.join(criteria_names)})"
            )
    if not results:
        return f"No rubrics found matching '{query}'. Available: {', '.join(RUBRICS.keys())}"
    return f"Found {len(results)} rubric(s):\n" + "\n".join(results)


@mcp.tool(description="Get a specific rubric by name. Returns full rubric with all criteria details.")
def get_rubric(name: str) -> str:
    if name not in RUBRICS:
        return f"Rubric '{name}' not found. Available rubrics: {', '.join(RUBRICS.keys())}"
    rubric = RUBRICS[name]
    lines = [f"Rubric: {rubric['name']}", f"ID: {name}", ""]
    for criterion in rubric["criteria"]:
        lines.append(
            f"  - {criterion['name']}: {criterion['description']} "
            f"(max: {criterion['max_score']}, weight: {criterion['weight']})"
        )
    max_total = sum(c["max_score"] for c in rubric["criteria"])
    lines.append(f"\nTotal maximum score: {max_total}")
    return "\n".join(lines)


@mcp.resource("rubric://{name}", description="Get rubric data as JSON resource", mime_type="application/json")
def get_rubric_resource(name: str) -> str:
    if name not in RUBRICS:
        return json.dumps({"error": f"Rubric '{name}' not found"})
    return json.dumps(RUBRICS[name], indent=2, ensure_ascii=False)


@mcp.resource("rubrics://list", description="List all available rubrics", mime_type="application/json")
def list_rubrics_resource() -> str:
    summary = {}
    for rubric_id, rubric_data in RUBRICS.items():
        summary[rubric_id] = {
            "name": rubric_data["name"],
            "criteria_count": len(rubric_data["criteria"]),
            "max_total_score": sum(c["max_score"] for c in rubric_data["criteria"]),
        }
    return json.dumps(summary, indent=2, ensure_ascii=False)


@mcp.prompt(description="Generate an assessment prompt for evaluating student work against a rubric")
def assess(student_work: str, rubric_name: str, strictness: str = "standard") -> list[base.Message]:
    if rubric_name not in RUBRICS:
        return [
            base.UserMessage(
                content=f"Error: Rubric '{rubric_name}' not found. "
                f"Available: {', '.join(RUBRICS.keys())}"
            )
        ]

    rubric = RUBRICS[rubric_name]
    criteria_text = "\n".join(
        f"- {c['name']} (max {c['max_score']} pts): {c['description']}"
        for c in rubric["criteria"]
    )

    strictness_instruction = {
        "lenient": "Be generous with scoring. Give benefit of the doubt.",
        "standard": "Score fairly and objectively based on evidence in the work.",
        "strict": "Apply rigorous standards. Deduct points for any shortcoming.",
    }.get(strictness, "Score fairly and objectively based on evidence in the work.")

    return [
        base.UserMessage(
            content=f"Assess the following student work using the '{rubric['name']}' rubric.\n\n"
            f"Rubric criteria:\n{criteria_text}\n\n"
            f"Scoring approach: {strictness_instruction}\n\n"
            f"For each criterion, provide:\n"
            f"1. Score (0 to max)\n"
            f"2. Specific feedback with references to the student's work\n\n"
            f"Student work:\n---\n{student_work}\n---"
        )
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Сервер предоставляет:

| Примитив | Имя | Описание |
|---|---|---|
| Tool | `search_rubrics` | Поиск рубрик по ключевому слову |
| Tool | `get_rubric` | Получение полной рубрики по имени |
| Resource | `rubric://{name}` | Данные рубрики в формате JSON |
| Resource | `rubrics://list` | Список всех доступных рубрик |
| Prompt | `assess` | Промпт для оценки работы с аргументами |

Проверка через MCP Inspector:

```bash
mcp dev mcp_servers/rubrics_server.py
```

### Шаг 3. MCP Server: Оценки — `mcp_servers/assessment_server.py`

Второй сервер отвечает за оценку студенческих работ и историю оценок.

```python
import json
import hashlib
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "assessment-server",
    instructions="MCP server for student work assessment. "
    "Provides tools to assess work, track assessment history, "
    "and compare assessments.",
)

ASSESSMENT_STORE: dict[str, dict] = {}

RUBRICS = {
    "essay": {
        "name": "Essay Assessment Rubric",
        "criteria": [
            {"name": "Thesis & Argument", "max_score": 25},
            {"name": "Evidence & Support", "max_score": 25},
            {"name": "Structure & Organization", "max_score": 25},
            {"name": "Grammar & Style", "max_score": 25},
        ],
    },
    "code": {
        "name": "Code Review Rubric",
        "criteria": [
            {"name": "Correctness", "max_score": 30},
            {"name": "Efficiency", "max_score": 25},
            {"name": "Readability", "max_score": 25},
            {"name": "Testing", "max_score": 20},
        ],
    },
}


def _generate_assessment_id(text: str, rubric: str) -> str:
    hash_input = f"{text[:100]}:{rubric}:{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def _simple_assess(text: str, rubric_name: str) -> dict:
    if rubric_name not in RUBRICS:
        return {"error": f"Rubric '{rubric_name}' not found"}

    rubric = RUBRICS[rubric_name]
    word_count = len(text.split())

    scores = []
    for criterion in rubric["criteria"]:
        ratio = min(word_count / 200, 1.0)
        score = int(criterion["max_score"] * ratio * 0.7)
        scores.append({
            "criterion": criterion["name"],
            "score": score,
            "max_score": criterion["max_score"],
            "feedback": f"Score based on work length and complexity analysis for '{criterion['name']}'",
        })

    total = sum(s["score"] for s in scores)
    max_total = sum(s["max_score"] for s in scores)

    assessment_id = _generate_assessment_id(text, rubric_name)
    assessment = {
        "id": assessment_id,
        "rubric": rubric_name,
        "overall_score": total,
        "max_score": max_total,
        "percentage": round(total / max_total * 100, 1),
        "criteria_scores": scores,
        "word_count": word_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_preview": text[:200],
    }

    ASSESSMENT_STORE[assessment_id] = assessment
    return assessment


@mcp.tool(description="Assess a student's work against a named rubric. Returns detailed scores per criterion.")
def assess_work(text: str, rubric: str, student_id: str = "anonymous") -> str:
    if not text.strip():
        return "Error: empty student work provided"
    if len(text) > 50000:
        return "Error: student work exceeds 50000 character limit"

    assessment = _simple_assess(text, rubric)
    if "error" in assessment:
        return f"Error: {assessment['error']}"

    assessment["student_id"] = student_id
    ASSESSMENT_STORE[assessment["id"]] = assessment

    lines = [
        f"Assessment ID: {assessment['id']}",
        f"Rubric: {assessment['rubric']}",
        f"Overall Score: {assessment['overall_score']}/{assessment['max_score']} "
        f"({assessment['percentage']}%)",
        f"Word Count: {assessment['word_count']}",
        "",
        "Criterion Scores:",
    ]
    for cs in assessment["criteria_scores"]:
        lines.append(f"  - {cs['criterion']}: {cs['score']}/{cs['max_score']} — {cs['feedback']}")

    return "\n".join(lines)


@mcp.tool(description="Get assessment history for a student. Returns list of past assessments.")
def get_assessment_history(student_id: str) -> str:
    history = [
        a for a in ASSESSMENT_STORE.values()
        if a.get("student_id") == student_id
    ]

    if not history:
        return f"No assessments found for student '{student_id}'"

    history.sort(key=lambda x: x["timestamp"], reverse=True)

    lines = [f"Assessment history for '{student_id}' ({len(history)} assessments):"]
    for a in history:
        lines.append(
            f"  - [{a['id']}] {a['rubric']}: "
            f"{a['overall_score']}/{a['max_score']} ({a['percentage']}%) "
            f"at {a['timestamp']}"
        )
    return "\n".join(lines)


@mcp.tool(description="Compare two assessments side by side. Shows score differences per criterion.")
def compare_assessments(id1: str, id2: str) -> str:
    a1 = ASSESSMENT_STORE.get(id1)
    a2 = ASSESSMENT_STORE.get(id2)

    if not a1:
        return f"Assessment '{id1}' not found"
    if not a2:
        return f"Assessment '{id2}' not found"

    lines = [
        f"Comparison: {id1} vs {id2}",
        f"Rubrics: {a1['rubric']} vs {a2['rubric']}",
        f"Overall: {a1['overall_score']}/{a1['max_score']} vs {a2['overall_score']}/{a2['max_score']}",
        f"Percentage: {a1['percentage']}% vs {a2['percentage']}%",
        "",
    ]

    if a1["rubric"] == a2["rubric"]:
        lines.append("Criterion comparison:")
        for c1, c2 in zip(a1["criteria_scores"], a2["criteria_scores"]):
            diff = c1["score"] - c2["score"]
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            lines.append(
                f"  - {c1['criterion']}: {c1['score']} vs {c2['score']} ({arrow}{abs(diff)})"
            )
    else:
        lines.append("Different rubrics — criterion-level comparison not available")

    score_diff = a1["percentage"] - a2["percentage"]
    if abs(score_diff) < 5:
        lines.append("\nVerdict: Assessments are roughly equivalent")
    elif score_diff > 0:
        lines.append(f"\nVerdict: First assessment scored higher by {score_diff:.1f}%")
    else:
        lines.append(f"\nVerdict: Second assessment scored higher by {abs(score_diff):.1f}%")

    return "\n".join(lines)


@mcp.resource("assessment://{assessment_id}", description="Get assessment details by ID", mime_type="application/json")
def get_assessment_resource(assessment_id: str) -> str:
    assessment = ASSESSMENT_STORE.get(assessment_id)
    if not assessment:
        return json.dumps({"error": f"Assessment '{assessment_id}' not found"})
    return json.dumps(assessment, indent=2, ensure_ascii=False)


@mcp.resource("assessments://recent", description="List recent assessments", mime_type="application/json")
def list_recent_assessments() -> str:
    recent = sorted(
        ASSESSMENT_STORE.values(),
        key=lambda x: x["timestamp"],
        reverse=True,
    )[:20]

    summary = [
        {
            "id": a["id"],
            "rubric": a["rubric"],
            "score": f"{a['overall_score']}/{a['max_score']}",
            "percentage": a["percentage"],
            "student_id": a.get("student_id", "anonymous"),
            "timestamp": a["timestamp"],
        }
        for a in recent
    ]
    return json.dumps(summary, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Сервер предоставляет:

| Примитив | Имя | Описание |
|---|---|---|
| Tool | `assess_work` | Оценка работы по рубрике с сохранением |
| Tool | `get_assessment_history` | История оценок студента |
| Tool | `compare_assessments` | Сравнение двух оценок |
| Resource | `assessment://{id}` | Данные оценки по ID |
| Resource | `assessments://recent` | Список недавних оценок |

Проверка:

```bash
mcp dev mcp_servers/assessment_server.py
```

### Шаг 4. MCP Client Service — `app/services/mcp_client.py`

Сервис инкапсулирует подключение к MCP-серверам и загрузку tools. Используется как async context manager.

```python
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).parent.parent.parent

RUBRICS_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_ROOT / "mcp_servers" / "rubrics_server.py")],
    env=None,
)

ASSESSMENT_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_ROOT / "mcp_servers" / "assessment_server.py")],
    env=None,
)

MULTI_SERVER_CONFIG = {
    "rubrics": {
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "mcp_servers" / "rubrics_server.py")],
        "transport": "stdio",
    },
    "assessment": {
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "mcp_servers" / "assessment_server.py")],
        "transport": "stdio",
    },
}


@asynccontextmanager
async def connect_to_server(server_params: StdioServerParameters):
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def get_mcp_tools(server_params: StdioServerParameters):
    async with connect_to_server(server_params) as session:
        tools = await load_mcp_tools(session)
        yield tools, session


@asynccontextmanager
async def get_multi_server_client():
    async with MultiServerMCPClient(MULTI_SERVER_CONFIG) as client:
        yield client
```

Ключевые решения:

- `sys.executable` — используем тот же Python-интерпретатор, в котором работает FastAPI. Это гарантирует, что MCP-сервер найдёт все установленные библиотеки.
- `PROJECT_ROOT` — абсолютный путь к корню проекта, чтобы пути к серверам работали из любой рабочей директории.
- Каждая функция — `asynccontextmanager`. Это обеспечивает корректное закрытие соединений: при выходе из `async with` subprocess завершается, ресурсы освобождаются.
- `MULTI_SERVER_CONFIG` — конфигурация для `MultiServerMCPClient`, объединяющего оба сервера.

### Шаг 5. Router — `app/api/v1/mcp_endpoints.py`

Роутер предоставляет 4 эндпоинта, каждый из которых демонстрирует разный аспект MCP.

```python
from fastapi import APIRouter, HTTPException
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.schemas.mcp import (
    MCPAssessRequest,
    MCPAssessResponse,
    MCPMultiServerRequest,
    MCPMultiServerResponse,
    MCPResourceRequest,
    MCPResourceResponse,
    MCPToolInfo,
    MCPToolsResponse,
)
from app.services.mcp_client import (
    ASSESSMENT_SERVER,
    RUBRICS_SERVER,
    connect_to_server,
    get_mcp_tools,
    get_multi_server_client,
)

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _get_model() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


@router.post("/tools")
async def list_mcp_tools(server: str = "rubrics") -> MCPToolsResponse:
    server_params = RUBRICS_SERVER if server == "rubrics" else ASSESSMENT_SERVER
    server_name = server

    try:
        async with connect_to_server(server_params) as session:
            result = await session.list_tools()
            tools = [
                MCPToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema if tool.inputSchema else {},
                )
                for tool in result.tools
            ]
            return MCPToolsResponse(server=server_name, tools=tools)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {e}")


@router.post("/assess")
async def mcp_assess(request: MCPAssessRequest) -> MCPAssessResponse:
    model = _get_model()

    try:
        async with get_mcp_tools(RUBRICS_SERVER) as (tools, session):
            agent = create_react_agent(model, tools)

            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"First, get the '{request.rubric_name}' rubric using the get_rubric tool. "
                            f"Then provide a detailed assessment of this student work based on "
                            f"the rubric criteria:\n\n{request.student_work}",
                        }
                    ]
                }
            )

            tools_used = []
            response_text = ""
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_used.extend(tc["name"] for tc in msg.tool_calls)
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    response_text = msg.content

            return MCPAssessResponse(result=response_text, tools_used=tools_used)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP assessment failed: {e}")


@router.post("/resources")
async def read_mcp_resource(request: MCPResourceRequest) -> MCPResourceResponse:
    is_assessment = request.uri.startswith("assessment://") or request.uri.startswith("assessments://")
    server_params = ASSESSMENT_SERVER if is_assessment else RUBRICS_SERVER

    try:
        async with connect_to_server(server_params) as session:
            result = await session.read_resource(request.uri)

            content = ""
            mime_type = None
            if result.contents:
                content = result.contents[0].text if hasattr(result.contents[0], "text") else str(result.contents[0])
                mime_type = result.contents[0].mimeType if hasattr(result.contents[0], "mimeType") else None

            return MCPResourceResponse(uri=request.uri, content=content, mime_type=mime_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read MCP resource: {e}")


@router.post("/multi-server")
async def multi_server_agent(request: MCPMultiServerRequest) -> MCPMultiServerResponse:
    model = _get_model()

    try:
        async with get_multi_server_client() as client:
            tools = client.get_tools()
            tool_names_by_server = {}
            for tool in tools:
                server_name = getattr(tool, "server_name", "unknown")
                tool_names_by_server.setdefault(server_name, []).append(tool.name)

            agent = create_react_agent(model, tools)

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": request.query}]}
            )

            tools_used = []
            response_text = ""
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_used.extend(tc["name"] for tc in msg.tool_calls)
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    response_text = msg.content

            servers_used = list({
                server
                for server, tool_list in tool_names_by_server.items()
                for used in tools_used
                if used in tool_list
            })

            return MCPMultiServerResponse(
                result=response_text,
                tools_used=tools_used,
                servers_used=servers_used,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-server agent failed: {e}")
```

**Как каждый эндпоинт связан с теорией:**

| Эндпоинт | Концепция из теории | Что демонстрирует |
|---|---|---|
| `POST /mcp/tools` | §2 (Architecture), §3 (Tools) | Discovery: подключение к серверу и получение списка tools |
| `POST /mcp/assess` | §5 (Server), §7 (LangChain integration) | LangGraph агент, использующий MCP tools из одного сервера |
| `POST /mcp/resources` | §3 (Resources) | Чтение MCP resources по URI из разных серверов |
| `POST /mcp/multi-server` | §7 (MultiServerMCPClient) | Агент с tools из двух серверов одновременно |

### Шаг 6. Регистрация + Тестирование

Регистрация роутера в `app/api/router.py`:

```python
from fastapi import APIRouter

from app.api.v1 import assessment, rubrics, prompts, mcp_endpoints

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
api_router.include_router(prompts.router)
api_router.include_router(mcp_endpoints.router)
```

Установка зависимостей:

```bash
pip install "mcp[cli]" langchain-mcp-adapters httpx
```

Запуск сервера:

```bash
uvicorn app.main:app --reload
```

**Тестирование MCP серверов отдельно:**

```bash
mcp dev mcp_servers/rubrics_server.py
```

```bash
mcp dev mcp_servers/assessment_server.py
```

**POST /mcp/tools** — список tools с rubrics-сервера:

```bash
curl -s -X POST "http://localhost:8000/api/v1/mcp/tools?server=rubrics" \
  -H "Content-Type: application/json" | python -m json.tool
```

Ожидаемый результат: JSON с двумя tools — `search_rubrics` и `get_rubric`, каждый с описанием и `input_schema`.

**POST /mcp/tools** — список tools с assessment-сервера:

```bash
curl -s -X POST "http://localhost:8000/api/v1/mcp/tools?server=assessment" \
  -H "Content-Type: application/json" | python -m json.tool
```

Ожидаемый результат: три tools — `assess_work`, `get_assessment_history`, `compare_assessments`.

**POST /mcp/assess** — оценка через MCP + LangGraph агента:

```bash
curl -s -X POST http://localhost:8000/api/v1/mcp/assess \
  -H "Content-Type: application/json" \
  -d '{
    "student_work": "Climate change is a significant global challenge. Rising temperatures lead to melting ice caps, which causes sea levels to rise. Governments must implement carbon taxes and invest in renewable energy sources. Scientific evidence clearly shows that human activities are the primary driver of climate change. Without immediate action, future generations will face severe environmental consequences.",
    "rubric_name": "essay"
  }' | python -m json.tool
```

Ожидаемый результат: `result` содержит развёрнутую оценку по каждому критерию рубрики. `tools_used` содержит `["get_rubric"]` — агент сначала получил рубрику, затем оценил работу.

**POST /mcp/resources** — чтение ресурса рубрики:

```bash
curl -s -X POST http://localhost:8000/api/v1/mcp/resources \
  -H "Content-Type: application/json" \
  -d '{"uri": "rubric://essay"}' | python -m json.tool
```

Ожидаемый результат: JSON с полными данными рубрики `essay` — все критерии, max_score, weight.

**POST /mcp/resources** — список всех рубрик:

```bash
curl -s -X POST http://localhost:8000/api/v1/mcp/resources \
  -H "Content-Type: application/json" \
  -d '{"uri": "rubrics://list"}' | python -m json.tool
```

**POST /mcp/multi-server** — агент с инструментами из двух серверов:

```bash
curl -s -X POST http://localhost:8000/api/v1/mcp/multi-server \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the essay rubric, then assess this student work: \"The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration. This process involves glycolysis, the Krebs cycle, and oxidative phosphorylation.\" Use the essay rubric for assessment."
  }' | python -m json.tool
```

Ожидаемый результат: агент использует `get_rubric` из rubrics-сервера и `assess_work` из assessment-сервера. `tools_used` содержит оба tool, `servers_used` — оба сервера.

### Связь с теорией

| Концепция | Где в коде | Секция теории |
|---|---|---|
| MCP Server с FastMCP | `mcp_servers/rubrics_server.py` | §5 |
| Tools, Resources, Prompts | Оба MCP-сервера | §3 |
| stdio транспорт | `app/services/mcp_client.py` → `StdioServerParameters` | §4 |
| ClientSession lifecycle | `connect_to_server()` context manager | §6 |
| load_mcp_tools → LangChain | `get_mcp_tools()` → `load_mcp_tools()` | §7 |
| MultiServerMCPClient | `get_multi_server_client()`, endpoint `/multi-server` | §7 |
| Capability discovery | `POST /mcp/tools` → `session.list_tools()` | §2 |
| Resource URI addressing | `POST /mcp/resources` → `session.read_resource(uri)` | §3 |
| Input validation | `assess_work()` — проверка длины и пустоты | §8 |
| Error handling | `try/except` в эндпоинтах, HTTPException | §8 |

---

## Чеклист самопроверки

- [ ] Объясни разницу между MCP Host, Client и Server. Приведи пример каждой роли.
- [ ] Чем MCP tool отличается от MCP resource? Кто контролирует вызов каждого?
- [ ] Когда использовать stdio транспорт, а когда HTTP+SSE?
- [ ] Что происходит при вызове `session.initialize()`? Какие данные обмениваются?
- [ ] Как `FastMCP` автоматически генерирует `inputSchema` для tool из type hints?
- [ ] Что делает `load_mcp_tools()` и как MCP tool становится LangChain tool?
- [ ] Зачем нужен `MultiServerMCPClient`? Как он объединяет tools из разных серверов?
- [ ] Почему MCP-сервер запускается как отдельный процесс, а не как функция в коде?
- [ ] Какие проблемы безопасности возникают при использовании MCP и как их решать?
- [ ] Как тестировать MCP-сервер отдельно от клиента? Какие инструменты для этого есть?

---

## Частые ошибки

### 1. Забыть await при работе с async MCP client

```python
session.initialize()
tools = session.list_tools()
```

```python
await session.initialize()
tools = await session.list_tools()
```

`ClientSession` — полностью асинхронный. Без `await` вызов вернёт coroutine вместо результата. Python не выбросит ошибку сразу — вы получите `coroutine object` вместо данных и потратите время на отладку.

### 2. Не закрыть MCP session (resource leak)

```python
read, write = await stdio_client(server_params).__aenter__()
session = await ClientSession(read, write).__aenter__()
await session.initialize()
tools = await session.list_tools()
```

```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
```

При использовании stdio каждый MCP-сервер — это subprocess. Если не закрыть session через context manager, subprocess останется жить. Накопление незавершённых процессов приводит к утечке памяти и исчерпанию file descriptors.

### 3. MCP server без error handling

```python
@mcp.tool()
def get_rubric(name: str) -> str:
    return RUBRICS[name]["content"]
```

```python
@mcp.tool()
def get_rubric(name: str) -> str:
    if name not in RUBRICS:
        return f"Error: rubric '{name}' not found. Available: {', '.join(RUBRICS.keys())}"
    return RUBRICS[name]["content"]
```

Если LLM передаёт невалидное имя (а это будет происходить), `KeyError` убьёт MCP-сервер. Исключения внутри tool обрывают соединение. Всегда возвращайте информативную ошибку как строку — LLM прочитает сообщение и скорректирует запрос.

### 4. Смешивание stdio и SSE транспорта

```python
params = StdioServerParameters(command="python", args=["server.py"])
async with sse_client("http://localhost:8080") as (read, write):
    ...
```

```python
params = StdioServerParameters(command="python", args=["server.py"])
async with stdio_client(params) as (read, write):
    ...
```

`StdioServerParameters` предназначен только для `stdio_client`. Для SSE используется URL-строка с `sse_client`. Смешивание приведёт к ошибке подключения или зависанию.

### 5. Слишком широкие resource URIs

```python
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    return open(path).read()
```

```python
ALLOWED_DIRS = {"/data/rubrics", "/data/assessments"}

@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    from pathlib import Path
    resolved = Path(path).resolve()
    if not any(str(resolved).startswith(d) for d in ALLOWED_DIRS):
        return "Error: access denied"
    return resolved.read_text()
```

Неограниченный resource URI позволяет читать произвольные файлы системы. MCP-сервер может стать вектором атаки, если LLM сформирует вредоносный URI (path traversal: `file://../../etc/passwd`). Всегда ограничивайте доступные пути.

### 6. Не тестировать MCP server отдельно от клиента

Если тестировать MCP server только через полный pipeline (FastAPI → MCP Client → MCP Server → LLM → ответ), отладка ошибок становится крайне сложной. Невозможно понять, что сломалось: сервер, клиент, транспорт или LLM.

Всегда тестируйте MCP-сервер отдельно:

```bash
mcp dev mcp_servers/rubrics_server.py
```

MCP Inspector позволяет вызвать каждый tool, прочитать каждый resource и получить каждый prompt вручную — без клиента, без LLM, без FastAPI. Убедитесь, что сервер работает корректно, прежде чем интегрировать с остальной системой.

---

## Что читать дальше

- [MCP Specification](https://spec.modelcontextprotocol.io/) — полная спецификация протокола
- [MCP Documentation](https://modelcontextprotocol.io/introduction) — официальная документация Anthropic
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python SDK с примерами
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) — интеграция MCP с LangChain
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers) — коллекция reference MCP-серверов
- [Building MCP with FastMCP](https://modelcontextprotocol.io/tutorials/building-mcp-with-fastmcp) — туториал по FastMCP
- [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) — инструмент тестирования MCP-серверов

**Следующая тема:** [Тема 16](topic_16_a2a.md) — A2A (Agent-to-Agent) протокол: взаимодействие между AI-агентами.

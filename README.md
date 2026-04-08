# AI Engineering Bootcamp

Учебный курс по AI Engineering: LangChain, LangGraph, RAG, Agents, Evaluation и инструменты.

## Структура

```
docs/
  CURRICULUM.md          — учебный план (27 тем)
  GENAI_STACK.md         — справочник GenAI библиотек
  LEARNING_LOG.md        — заметки по сессиям
  build_book.py          — скрипт сборки PDF-книги
  ai-bootcamp-book.pdf   — собранная книга
  lessons/
    topic_01_prompt_engineering.md
    topic_02_langchain_lcel.md
    ...
    topic_22_prompt_optimization.md
PROGRESS.md              — прогресс обучения
```

## Формат

Каждый урок — текстовый документ с теорией и примерами кода в стиле Jupyter notebook. Код можно запускать в Jupyter, IPython или как standalone Python-скрипт.

## Установка зависимостей

```bash
uv sync
cp .env.example .env  # добавить ANTHROPIC_API_KEY, OPENAI_API_KEY
```

## Сборка PDF-книги

```bash
uv run python docs/build_book.py
```

## Темы

| # | Тема |
|---|------|
| 1 | Промпт-инжиниринг |
| 2 | LangChain Core + LCEL |
| 3 | Structured Output |
| 4 | Streaming |
| 5 | RAG |
| 6 | LangGraph + Agents |
| 7 | Conversational AI |
| 8 | Observability |
| 9 | Evaluation |
| 10 | Production-паттерны |
| 11 | Tool Use / Function Calling |
| 12 | Multimodal AI |
| 13 | Advanced Agentic Patterns |
| 14 | Multi-Agent Systems |
| 15 | MCP |
| 16 | Langfuse Deep Dive |
| 17 | LangSmith |
| 18 | Ollama и локальные LLM |
| 19 | Vector Databases |
| 20 | Deployment |
| 21 | AI Testing & QA |
| 22 | Prompt Optimization (GEPA, TensorZero) |
| 23 | Databricks (MLflow, Model Serving, Vector Search) |
| 24 | PydanticAI |
| 25 | DSPy |
| 26 | LiteLLM |
| 27 | CrewAI |

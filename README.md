# AI Student Assessment System

Учебный GenAI проект: LangChain → LangGraph → RAG → Agents → Evaluation

## Установка
```bash
uv sync
cp .env.example .env  # добавить ANTHROPIC_API_KEY
```

## Запуск
```bash
uv run uvicorn app.main:app --reload
```

## Линтинг и форматирование
```bash
uv run ruff check . --fix
uv run ruff format .
```

## Добавить зависимости следующих фаз
```bash
uv add --optional rag    # Фаза 2
uv add --optional agents # Фаза 3
uv add --optional eval   # Фаза 4-5
```

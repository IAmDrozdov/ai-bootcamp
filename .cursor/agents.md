# Cursor Agent Configuration

## Проект
AI Engineering Bootcamp — учебный курс по AI Engineering (LangChain, LangGraph, RAG, Agents, Evaluation).
Формат: текстовые уроки с примерами кода в стиле Jupyter notebook.

## Документация
- `PROGRESS.md` — компактный мониторинг прогресса (таблица тем, текущий фокус, история сессий)
- `docs/CURRICULUM.md` — полный учебный план (22 темы, концепции, практические задания)
- `docs/LEARNING_LOG.md` — подробные заметки по сессиям
- `docs/GENAI_STACK.md` — справочник GenAI библиотек
- `docs/lessons/` — уроки (topic_01 — topic_22)

## Философия
**Меньше кода — больше знаний.**
- Это УЧЕБНЫЙ курс. Цель — изучить GenAI стек через текстовые уроки с примерами.
- Примеры кода — самостоятельные сниппеты в стиле Jupyter notebook (не приложение).
- Пользователь НЕ пишет код руками — весь контент генерирует Agent.

## Обновление прогресса — ОБЯЗАТЕЛЬНО
**В конце КАЖДОГО чата** (перед завершением) обнови:

1. **`PROGRESS.md`**:
   - Таблицу тем: статус (⬜ не начата / 🔄 в процессе / ✅ завершена), счётчик заданий
   - "Текущий фокус" — над чем работали
   - "Следующий шаг" — что делать в следующем чате
   - "История сессий" — добавить строку: дата, тема, что сделано, ключевой инсайт

2. **`docs/LEARNING_LOG.md`**:
   - Добавить запись в секцию соответствующей темы: что сделали, какие GenAI концепции освоены, инсайты

## Tooling
- **uv** — единственный package manager. НЕ pip, НЕ poetry, НЕ pipenv.
  - Установка пакетов: `uv add <package>`
  - Lock file: `uv.lock` (коммитим)
- **ruff** — единственный linter/formatter. НЕ black, НЕ flake8, НЕ isort.
  - Format: `uv run ruff format .`
  - Lint: `uv run ruff check . --fix`
- Python 3.14

## Стек
- LangChain (langchain-core, langchain-anthropic, langchain-openai, langchain-chroma)
- LangGraph
- ChromaDB
- Langfuse / LangSmith
- GEPA / TensorZero
- DSPy
- Anthropic Claude как основная модель

## При генерации контента
- Примеры кода — self-contained, без `from app.*` импортов
- Type hints всегда
- Async где поддерживается (LangChain async API)
- Pydantic v2 синтаксис
- Импорты из langchain_core (underscore, не dash) где возможно

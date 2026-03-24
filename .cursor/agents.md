# Cursor Agent Configuration

## Проект
AI Student Assessment System — учебный проект для освоения AI Engineering (LangChain, LangGraph, RAG, Agents).

## Документация
- `PROGRESS.md` — компактный мониторинг прогресса (таблица тем, текущий фокус, история сессий)
- `docs/CURRICULUM.md` — полный учебный план (10 тем, концепции, практические задания)
- `docs/LEARNING_LOG.md` — подробные заметки по сессиям
- `docs/GENAI_STACK.md` — справочник GenAI библиотек
- `docs/API.md` — документация эндпоинтов

## Философия
**Меньше кода — больше знаний.**
- Это УЧЕБНЫЙ проект. Цель — изучить GenAI стек, не строить production систему.
- Никаких тестов. Совсем. Не предлагай их, не создавай.
- Минимум абстракций и boilerplate. Если можно написать проще — пиши проще.
- Комментарии в коде объясняют GenAI/LangChain концепции, не очевидный Python.
- Пользователь НЕ пишет код руками — весь код генерирует Agent.

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
  - Запуск: `uv run uvicorn app.main:app --reload`
  - Lock file: `uv.lock` (коммитим)
- **ruff** — единственный linter/formatter. НЕ black, НЕ flake8, НЕ isort.
  - Format: `uv run ruff format .`
  - Lint: `uv run ruff check . --fix`
- Python 3.14

## Стек
- FastAPI + Pydantic v2
- LangChain (langchain-core, langchain-anthropic, langchain-openai, langchain-chroma)
- LangGraph (с Темы 6)
- ChromaDB (с Темы 5)
- Langfuse (с Темы 8)
- Anthropic Claude как основная модель

## Архитектурные решения
- LCEL chains (prompt | model | parser) — основной паттерн
- Pydantic v2 для structured output
- FastAPI dependency injection для LLM и сервисов
- Промпты хранятся в отдельных файлах, не захардкожены в сервисах

## Текущая тема
Тема 1 — Промпт-инжиниринг. См. `docs/CURRICULUM.md`.

## При генерации кода
- Type hints всегда
- Async где поддерживается (LangChain async API)
- Docstrings на английском, минимальные
- Pydantic v2 синтаксис
- Импорты из langchain_core (underscore, не dash) где возможно

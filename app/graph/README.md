# LangGraph — Фаза 3

Этот модуль будет содержать LangGraph-агентов для системы оценки.

## Планируемые компоненты
- `StateGraph` с типизированным состоянием для пайплайна оценки
- Ноды: analyze_work, retrieve_rubric, check_plagiarism, compare_previous, assess
- Условные переходы (conditional edges) для маршрутизации по сложности работы
- Human-in-the-loop для подтверждения преподавателем

# Документация API

Базовый URL: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

## Проверка здоровья
`GET /health` → `{ "status": "ok" }`

## Оценка

### Оценка студенческой работы
`POST /api/v1/assess`
```json
{
  "student_work": "текст работы...",
  "rubric_id": "essay_default",
  "rubric": null
}
```
Ответ: `AssessmentResponse` с полями overall_score, criterion_scores, summary, strengths, improvements.

### Оценка (потоковая)
`POST /api/v1/assess/stream` — SSE поток, тело запроса то же самое.

## Рубрики
- `POST /api/v1/rubrics` — создать рубрику
- `GET /api/v1/rubrics` — список рубрик
- `GET /api/v1/rubrics/{rubric_id}` — получить рубрику по ID

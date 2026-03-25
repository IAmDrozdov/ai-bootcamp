# Тема 9: Evaluation

> **Пререквизиты:** [Тема 1-4](topic_01_prompt_engineering.md), рекомендуется [Тема 5 (RAG)](topic_05_rag.md), [Тема 8 (Observability)](topic_08_observability.md)
> **Где в проекте:** `data/golden/`, eval-скрипты
> **Зависимости:** `langfuse`, `ragas` (группа `eval`)

---

## Теория

### 1. Зачем evaluation

Обычный софт: написал тест → зелёный = работает, красный = сломано.

LLM-софт: тесты на **корректность** недостаточны. Модель может вернуть валидный JSON с бессмысленными оценками. Нужны метрики **качества**:
- Насколько оценки совпадают с эталоном?
- Стабильны ли оценки при повторных запусках?
- Новый промпт лучше старого или хуже?

Evaluation — это **CI/CD для промптов**: прежде чем менять промпт в production, прогони его на golden dataset и убедись, что не стало хуже.

### 2. Golden dataset

Golden dataset — набор данных с **эталонными ответами**, размеченными вручную:

```json
{
  "input": {
    "student_work": "Essay text...",
    "rubric_id": "essay_default"
  },
  "expected_output": {
    "overall_score": 72,
    "criterion_scores": [
      {"name": "Thesis & Argument", "score": 18, "max_score": 25},
      ...
    ]
  }
}
```

Требования к golden dataset:
- **Размер** — минимум 20 примеров, идеально 50-100
- **Разнообразие** — все уровни качества (отлично, хорошо, средне, плохо)
- **Edge cases** — очень короткие, очень длинные, off-topic, с инъекциями
- **Эталон** — оценки поставлены экспертом, не LLM

### 3. LLM-as-judge

Вместо ручной проверки каждого ответа — используем **другую LLM** для оценки:

```
Input: student essay + rubric
System A output: overall_score=72, feedback="..."
Expected: overall_score=75, feedback="..."

Judge LLM: "System A's assessment is 85% aligned with the expected output.
  Scores match within 5 points on 4/5 criteria. Feedback captures main strengths
  but misses the weak transition in paragraph 3."
```

Паттерны LLM-as-judge:
- **Pointwise** — оцени один ответ по шкале 1-5
- **Pairwise** — какой из двух ответов лучше?
- **Reference-based** — насколько ответ совпадает с эталоном?

### 4. Метрики для оценочных систем

| Метрика | Формула | Что измеряет |
|---------|---------|-------------|
| MAE (Mean Absolute Error) | Σ\|predicted - actual\| / N | Средняя ошибка в баллах |
| Exact Match | count(predicted == actual) / N | Процент точных совпадений |
| Within-N | count(\|pred - actual\| ≤ N) / N | Процент оценок в пределах N баллов |
| Pearson correlation | ρ(predicted, actual) | Корреляция (порядок сохранён?) |

Для нашего проекта: MAE ≤ 5 баллов по критерию — хороший результат. Exact match на всех 5 критериях — маловероятен и не обязателен.

### 5. RAGAS метрики (для RAG)

Если используешь RAG (Тема 5), нужны дополнительные метрики:

| Метрика | Что измеряет |
|---------|-------------|
| **Faithfulness** | Ответ основан на найденных документах (нет "галлюцинаций") |
| **Answer Relevancy** | Ответ отвечает на вопрос (не off-topic) |
| **Context Precision** | Найденные документы релевантны запросу |
| **Context Recall** | Все нужные документы найдены |

### 6. A/B testing промптов

Два промпта → один golden dataset → какой даёт оценки ближе к эталону:

```
Prompt A (текущий):       MAE=4.2, Within-5=78%
Prompt B (с CoT):         MAE=3.1, Within-5=86%
Prompt C (с few-shot x3): MAE=3.5, Within-5=82%

Вывод: Prompt B лучше. Деплоим.
```

### 7. Regression testing

При каждом изменении промпта — прогон на golden dataset. Если метрики ухудшились — не деплоить:

```
Current production: MAE=3.1
New prompt candidate: MAE=3.8  ← REGRESSION! Don't deploy.
```

---

## Практические задания

### Задание 1: Golden dataset

**Цель:** создать набор эталонных данных для evaluation.

**Файлы:** `data/golden/`, скрипт создания

**Критерии успеха:**
- 20 эссе с эталонными оценками
- Разнообразие: 5 сильных, 5 средних, 5 слабых, 5 edge cases
- Формат: JSON с input (текст + рубрика) и expected output (оценки)

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create a golden dataset for assessment evaluation.

1. Create data/golden/ directory

2. Create data/golden/golden_dataset.json with 20 entries:
   Each entry has:
   {
     "id": "golden_001",
     "input": {
       "student_work": "Essay text...",
       "rubric_id": "essay_default"
     },
     "expected_output": {
       "overall_score": 75,
       "criterion_scores": [
         {"criterion_name": "Thesis & Argument", "score": 18, "max_score": 25},
         {"criterion_name": "Evidence & Support", "score": 19, "max_score": 25},
         {"criterion_name": "Structure & Organization", "score": 15, "max_score": 20},
         {"criterion_name": "Critical Thinking", "score": 14, "max_score": 20},
         {"criterion_name": "Language & Style", "score": 9, "max_score": 10}
       ]
     },
     "metadata": {
       "quality_level": "good",
       "description": "Well-structured essay on AI with good citations"
     }
   }

   Distribution:
   - 5 strong essays (75-95): clear thesis, citations, good structure
   - 5 medium essays (50-74): decent but lacking depth or citations
   - 5 weak essays (20-49): vague thesis, no citations, poor structure
   - 5 edge cases: very short (<100 words), off-topic, with injection attempts, mixed language

   Each essay should be 100-300 words, realistic student writing.
   Scores should be carefully calibrated against the essay_rubric.json criteria.

3. Create data/golden/schema.py with Pydantic models for the golden dataset format

4. Create experiments/t9_load_golden.py to load and validate the dataset:
   - Load JSON, validate against schema
   - Print distribution: quality levels, score ranges
   - Verify all scores are within rubric bounds

Run with: python -m experiments.t9_load_golden
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review data/golden/golden_dataset.json:

1. QUANTITY: Are there 20 entries?
2. DIVERSITY: 5 strong, 5 medium, 5 weak, 5 edge cases?
3. REALISTIC: Do the essays read like actual student writing?
4. CALIBRATION: Do scores match the quality? (Strong essays should score 75+, weak <50)
5. COMPLETENESS: All 5 criteria scored for each entry?
6. EDGE CASES: Short texts, off-topic, injection attempts included?
7. SCHEMA: Is the JSON structure consistent and validated?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 2: Eval pipeline

**Цель:** создать скрипт, который прогоняет golden dataset через chain и измеряет качество.

**Файлы:** `experiments/t9_eval_pipeline.py`

**Критерии успеха:**
- Все 20 эссе прогоняются через chain
- Метрики: MAE, exact match, within-5 по каждому критерию
- Сводная таблица с результатами

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t9_eval_pipeline.py — evaluation pipeline for the assessment chain.

1. Load golden dataset from data/golden/golden_dataset.json

2. Run each entry through the assessment chain:
   - Use temperature=0 for determinism
   - Collect predicted scores for all criteria

3. Calculate metrics per criterion:
   - MAE (Mean Absolute Error): average |predicted - expected|
   - Exact match rate: predicted == expected
   - Within-3: |predicted - expected| <= 3
   - Within-5: |predicted - expected| <= 5

4. Calculate overall metrics:
   - Overall score MAE
   - Overall score correlation (Pearson)
   - Average MAE across all criteria

5. Print detailed report:
   Criterion              MAE    Exact%   Within-3   Within-5
   Thesis & Argument      3.2    15%      55%        80%
   Evidence & Support     4.1    10%      40%        65%
   ...
   OVERALL SCORE          5.8    5%       35%        60%

6. Print worst predictions (top 5 largest errors) with details

7. Save results to data/golden/eval_results.json

Run with: python -m experiments.t9_eval_pipeline
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t9_eval_pipeline.py:

1. PIPELINE: Does it load golden data, run chain, compare results?
2. METRICS: MAE, exact match, within-N all calculated correctly?
3. PER-CRITERION: Are metrics broken down by criterion?
4. REPORT: Clear, readable output table?
5. WORST CASES: Are the worst predictions highlighted for analysis?
6. DETERMINISM: Is temperature=0 used for reproducible results?
7. PERSISTENCE: Are results saved for later comparison?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 3: LLM-as-judge

**Цель:** создать chain, где LLM оценивает качество оценок другой LLM.

**Файлы:** `experiments/t9_llm_judge.py`

**Критерии успеха:**
- Judge chain получает: student work, expected output, system output
- Выдаёт: alignment score (0-100), подробный анализ расхождений
- Judge корректно находит ошибки в оценках

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t9_llm_judge.py — LLM-as-judge for assessment quality.

1. Define JudgeResponse schema:
   - alignment_score: int (0-100, how well system output matches expected)
   - criterion_analysis: list[CriterionJudgment] with fields:
     criterion_name, expected_score, actual_score, difference, judgment (str)
   - overall_judgment: str (detailed analysis)
   - major_discrepancies: list[str] (criteria where |diff| > 5)

2. Create judge prompt:
   System: "You are an expert meta-assessor. Compare a student work assessment 
   against an expert reference assessment. Evaluate alignment on each criterion."
   Human: includes student_work, expected scores, system scores

3. Build judge chain: prompt | llm.with_structured_output(JudgeResponse)

4. Run judge on 10 golden dataset entries:
   - First run assessment chain to get system output
   - Then run judge to compare system vs expected
   - Print: alignment scores, major discrepancies, overall judgment

5. Aggregate: average alignment score, most common discrepancy patterns

Run with: python -m experiments.t9_llm_judge
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t9_llm_judge.py:

1. JUDGE SCHEMA: Is JudgeResponse well-structured with per-criterion analysis?
2. JUDGE PROMPT: Does it clearly instruct the judge on what to evaluate?
3. THREE INPUTS: Does judge receive student work, expected, AND actual scores?
4. DISCREPANCIES: Are major discrepancies identified and highlighted?
5. AGGREGATION: Is there a summary across all judged entries?

Common mistakes:
- Judge only sees scores, not the actual feedback (misses qualitative issues)
- Using the same model for judge and assessment (bias)
- Not including student work in judge context

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

### Задание 4: A/B тестирование промптов

**Цель:** сравнить два промпта на golden dataset и определить лучший.

**Файлы:** `experiments/t9_ab_test.py`

**Критерии успеха:**
- Два промпта: текущий и кандидат (другая формулировка)
- Оба прогоняются на golden dataset
- Таблица: prompt A vs prompt B по всем метрикам
- Чёткий вывод: какой промпт лучше

<details>
<summary>Промпт для реализации (Cursor AI)</summary>

```
Create experiments/t9_ab_test.py — A/B testing for assessment prompts.

1. Define two prompt variants:
   - Prompt A: current ASSESSMENT_SYSTEM_PROMPT from templates.py
   - Prompt B: modified version — e.g., more explicit scoring instructions, different CoT structure, stricter calibration instructions

2. For each prompt:
   - Build assessment chain
   - Run on full golden dataset (20 entries)
   - Calculate metrics: MAE, within-3, within-5, correlation

3. Print side-by-side comparison:
   Metric               Prompt A    Prompt B    Winner
   MAE (overall)        5.8         4.2         B
   Within-3 (avg)       40%         55%         B
   Within-5 (avg)       65%         78%         B
   Correlation          0.72        0.85        B

4. Per-criterion comparison: which prompt is better for each criterion

5. Statistical significance: is the difference meaningful or noise?
   (Simple: if MAE difference > 1.0, consider significant)

6. Recommendation: which prompt to deploy

Run with: python -m experiments.t9_ab_test
```

</details>

<details>
<summary>Промпт для оценки решения</summary>

```
Review experiments/t9_ab_test.py:

1. TWO PROMPTS: Are two meaningfully different prompts compared?
2. SAME DATASET: Both evaluated on the same golden dataset?
3. METRICS: Comprehensive comparison with multiple metrics?
4. PER-CRITERION: Breakdown by criterion?
5. RECOMMENDATION: Clear winner identified?
6. SIGNIFICANCE: Some notion of whether difference is meaningful?

Rate: PASS / NEEDS IMPROVEMENT / FAIL
```

</details>

---

## Чеклист самопроверки

- [ ] Зачем golden dataset? Почему LLM-сгенерированные эталоны не подходят?
- [ ] Объясни MAE и within-N. Какой MAE считается "хорошим" для оценок 0-25?
- [ ] Как работает LLM-as-judge? Какие у него ограничения?
- [ ] Зачем A/B тестирование промптов? Почему нельзя просто "улучшить" промпт и деплоить?
- [ ] Что такое regression testing для промптов?

---

## Частые ошибки

### 1. Golden dataset, размеченный LLM

```python
# Плохо: LLM оценивает → LLM проверяет = circular evaluation
expected = await llm.ainvoke("Score this essay")
# Тогда eval всегда покажет высокий alignment — модель согласна сама с собой

# Хорошо: эксперт размечает вручную
expected = human_expert_scores["essay_001"]
```

### 2. A/B тест на 3 примерах

```python
# Плохо: слишком маленькая выборка
dataset = golden_dataset[:3]  # 3 примера ничего не докажут

# Хорошо: минимум 20 примеров
dataset = golden_dataset  # все 20+
```

### 3. Один и тот же LLM как judge и subject

```python
# Спорно: Claude судит Claude
subject = ChatAnthropic(model="claude-sonnet-4-20250514")
judge = ChatAnthropic(model="claude-sonnet-4-20250514")  # тот же!

# Лучше: разные модели или разные температуры
judge = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)  # хотя бы temp=0
# Или: judge = ChatOpenAI(model="gpt-4o")  # другой провайдер
```

---

## Что читать дальше

- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — eval фреймворк LangChain
- [Langfuse Evaluation](https://langfuse.com/docs/scores/overview) — scoring в Langfuse
- [RAGAS Documentation](https://docs.ragas.io/) — метрики для RAG
- Paper: "Judging LLM-as-a-Judge" (Zheng et al., 2023) — надёжность LLM-as-judge

**Следующая тема:** [Тема 10: Production-паттерны](topic_10_production_patterns.md) — как оптимизировать и защищать.

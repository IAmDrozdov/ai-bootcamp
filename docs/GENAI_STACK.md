# GenAI Stack — библиотеки и инструменты

## Core Framework
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| langchain-core | Базовые абстракции: messages, prompts, output parsers, runnables (LCEL) | 1 |
| langchain | Chains, retrievers, document loaders, text splitters | 1-2 |
| langchain-anthropic | ChatAnthropic — интеграция с Claude API | 1 |
| langchain-openai | ChatOpenAI — fallback провайдер, embeddings | 2 |
| langgraph | Stateful agent graphs, циклы, условные переходы, human-in-the-loop | 3 |
| langsmith | Трейсинг, debugging, evaluation, dataset management (SaaS от LangChain) | 5 |

## Vector Stores & Embeddings
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| chromadb | Локальный vector store, работает на CPU/MPS | 2 |
| langchain-chroma | LangChain интеграция с ChromaDB | 2 |
| sentence-transformers | Локальные embedding модели (all-MiniLM-L6-v2 и др.) | 2 |
| tiktoken | Токенизатор OpenAI — подсчёт токенов, chunk sizing | 2 |

## Document Processing
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| unstructured | Парсинг PDF, DOCX, PPTX, HTML → текст | 2 |
| pypdf | Лёгкий PDF reader | 2 |
| python-docx | Чтение .docx файлов студенческих работ | 2 |

## Observability & Evaluation
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| langfuse | Open-source трейсинг, мониторинг LLM вызовов, prompt management | 4-5 |
| ragas | Evaluation framework для RAG pipeline (faithfulness, relevance) | 5 |
| deepeval | Unit-тесты для LLM (альтернатива ragas) | 5 |
| promptfoo | CLI для A/B тестирования промптов | 5 |

## Guardrails & Safety
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| guardrails-ai | Валидация LLM output, retry logic, structured guarantees | 5 |

## Caching & Optimization
| Библиотека | Назначение | Фаза |
|-----------|-----------|------|
| redis | Кэширование LLM ответов, semantic cache | 5 |
| langchain-redis | LangChain Redis integration | 5 |

## Альтернативы (знать, не обязательно использовать)
| Библиотека | Назначение | Зачем знать |
|-----------|-----------|-------------|
| instructor | Structured output через function calling (проще LangChain) | Популярная альтернатива |
| litellm | Единый интерфейс к 100+ LLM, proxy server | Multi-provider |
| llamaindex | Альтернатива LangChain, сильнее в RAG | Сравнить подходы |
| dspy | Programmatic prompt optimization | Cutting-edge |
| crewai | Multi-agent фреймворк | Multi-agent сценарии |
| autogen | Microsoft multi-agent framework | Альтернатива CrewAI |
| haystack | Pipeline-based RAG от deepset | Альтернатива для RAG |
| semantic-kernel | Microsoft AI orchestration SDK | Enterprise |
| pydantic-ai | Agent framework от создателей Pydantic | Набирает популярность |

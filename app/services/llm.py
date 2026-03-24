# LLM service — creating and configuring the chat model.
#
# LangChain distinguishes two abstractions:
# - LLM (BaseLLM): text-in → text-out, legacy completion models
# - ChatModel (BaseChatModel): messages-in → message-out, modern chat APIs
#
# ChatAnthropic implements BaseChatModel. It wraps the Anthropic API and exposes
# the unified Runnable interface (invoke/ainvoke/stream/astream/batch).
# This means you can plug it into any LCEL chain with the | operator.
#
# Why wrap creation in a function?
# - Centralized config (model name, temperature, etc.)
# - Easy to swap providers later (ChatOpenAI, ChatOllama)
# - Testable: can create with different configs

from langchain_anthropic import ChatAnthropic

from app.config import Settings


def create_llm(config: Settings) -> ChatAnthropic:
    """Create a configured ChatAnthropic instance."""
    return ChatAnthropic(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.anthropic_api_key,
    )

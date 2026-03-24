# Assessment service — orchestrates the chain invocation.
#
# invoke vs stream in LangChain:
# - invoke/ainvoke: waits for the full response, returns complete object.
#   Best for structured output where you need the whole Pydantic model.
# - stream/astream: yields chunks as they arrive from the LLM.
#   Great for UX (show progress) but chunks are partial — you get token strings,
#   not a complete Pydantic object.
# - astream_events: yields detailed events (on_chat_model_stream, on_chain_end, etc.)
#   with metadata — most flexible for complex streaming scenarios.
#
# For structured output (with_structured_output), use ainvoke — the parser needs
# the complete JSON to construct the Pydantic model.
# For streaming, we use the raw LLM (without structured output) and stream text.

from collections.abc import AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse
from app.schemas.rubric import Rubric


async def assess(
    student_work: str,
    rubric: Rubric,
    chain: Runnable,
) -> AssessmentResponse:
    """Run assessment chain and return structured result."""
    rubric_text = _format_rubric(rubric)
    return await chain.ainvoke({"student_work": student_work, "rubric": rubric_text})


async def assess_stream(
    student_work: str,
    rubric: Rubric,
    llm: ChatAnthropic,
) -> AsyncIterator[str]:
    """Stream raw LLM assessment response token by token.

    Streaming and structured output don't mix well — with_structured_output
    needs the full response to parse JSON. So for streaming we use the raw LLM
    with a text-based prompt and yield chunks as they arrive.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSESSMENT_SYSTEM_PROMPT),
            ("human", "Please assess the following student work:\n\n{student_work}"),
        ]
    ).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )

    chain = prompt | llm
    rubric_text = _format_rubric(rubric)

    async for chunk in chain.astream({"student_work": student_work, "rubric": rubric_text}):
        if chunk.content:
            yield chunk.content


def _format_rubric(rubric: Rubric) -> str:
    lines = [f"Rubric: {rubric.name}\n"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (max {c.max_score}, weight {c.weight}): {c.description}")
    return "\n".join(lines)

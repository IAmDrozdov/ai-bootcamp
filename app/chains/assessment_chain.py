# LCEL chain for student work assessment.
#
# LCEL (LangChain Expression Language) lets you compose chains with the | operator:
#   chain = prompt | model | parser
#
# Every component implements the Runnable protocol:
#   .invoke(input)    — single call, returns result
#   .ainvoke(input)   — async version
#   .stream(input)    — yields output chunks
#   .astream(input)   — async streaming
#   .batch(inputs)    — parallel execution
#
# The pipe operator connects them: output of left becomes input of right.
# ChatPromptTemplate.invoke({"rubric": ...}) → ChatMessages
# ChatAnthropic.invoke(messages) → AIMessage
# with_structured_output wraps the model to parse JSON → Pydantic object
#
# with_structured_output() vs PydanticOutputParser:
# - with_structured_output(): uses tool calling / JSON mode at the API level,
#   more reliable, the model is constrained by the API itself
# - PydanticOutputParser: injects format instructions into the prompt,
#   then parses the text output — can fail if LLM produces malformed JSON

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.prompts.templates import (
    ASSESSMENT_SYSTEM_PROMPT,
    FEW_SHOT_BAD_EXAMPLE,
    FEW_SHOT_GOOD_EXAMPLE,
)
from app.schemas.assessment import AssessmentResponse


def build_assessment_chain(llm: ChatAnthropic) -> Runnable:
    """Build LCEL chain: prompt | llm.with_structured_output(AssessmentResponse).

    Input keys: student_work, rubric
    Output: AssessmentResponse
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ASSESSMENT_SYSTEM_PROMPT,
            ),
            (
                "human",
                "Please assess the following student work:\n\n{student_work}",
            ),
        ]
    ).partial(
        few_shot_good=FEW_SHOT_GOOD_EXAMPLE,
        few_shot_bad=FEW_SHOT_BAD_EXAMPLE,
    )

    structured_llm = llm.with_structured_output(AssessmentResponse)

    return prompt | structured_llm

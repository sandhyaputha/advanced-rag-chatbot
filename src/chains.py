

from langchain_core.output_parsers import StrOutputParser

from src.config import llm

from src.prompts import (
    intent_prompt,
    contextualize_prompt,
    answer_prompt,
    general_prompt
)

# 1. Intent classification chain
intent_chain = (
    intent_prompt
    | llm
    | StrOutputParser()
)

# 2. Follow-up question rewriting chain
contextualize_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)

# 3. Document-based answer chain
document_answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)

# 4. General conversation chain
general_chain = (
    general_prompt
    | llm
    | StrOutputParser()
)


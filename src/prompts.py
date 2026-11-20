
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# 1. Intent classification prompt
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an intent classification assistant.

        Classify the latest user question into exactly one category.

        DOCUMENT:
        Questions requiring information from PDF documents.

        HISTORY:
        Questions about previous user questions,
        AI responses, or conversation details.

        GENERAL:
        Greetings, small talk, and general conversation.

        Use conversation history to understand follow-up questions.

        Examples:
        What is the leave policy? -> DOCUMENT
        What about other leave types? -> DOCUMENT
        What was my last question? -> HISTORY
        How are you? -> GENERAL
        Hello -> GENERAL

        Return ONLY:
        DOCUMENT, HISTORY, or GENERAL.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# 2. Follow-up question rewriting prompt
contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Rewrite the latest user question into a standalone question
        using conversation history.

        Rules:
        1. Do not answer the question.
        2. Do not explain anything.
        3. Do not invert information.
        4. Preserve the original meaning.
        5. If the question is already clear, return it.
        6. Return only the rewritten question.

        Example:

        Conversation:
        User: What is the leave policy?

        Latest Question:
        What about others?

        Rewritten Question:
        What other leave types are mentioned in the company policy?
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# 3. Document-based answer prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a document-based AI assistant.

        Answer using ONLY the provided document context.

        Rules:
        1. Do not use outside knowledge.
        2. Do not assume or invent information.
        3. If information is unavailable, say:
           I don't know based on the provided documents.
        4. Present the answer clearly.

        Context:
        {context}
        """
    ),
    ("human", "{input}")
])


# 4. General conversation prompt
general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a friendly AI assistant.

        Respond naturally to greetings.

        Dont answer for any technical related Queries ,

        For technical questions answer like I Dont Know.

        Keep your responses simple, polite, and conversational.

        Do not retrieve documents for these questions.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

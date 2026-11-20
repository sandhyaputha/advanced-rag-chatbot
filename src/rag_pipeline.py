


from src.config import embeddings
from src.supabase_client import supabase

from src.chains import (
    intent_chain,
    contextualize_chain,
    document_answer_chain,
    general_chain
)

def get_rag_response(user_input, chat_history):

    # Step 1: Detect user intent
    intent = intent_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    }).strip().upper()

    # Step 2: Handle general conversation
    if intent == "GENERAL":

        response = general_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        return response, intent, []

    # Step 3: Handle conversation history
    elif intent == "HISTORY":

        if not chat_history:
            return (
                "There is no previous conversation yet.",
                intent,
                []
            )

        history_text = "\n".join([
            f"{message.type}: {message.content}"
            for message in chat_history
        ])

        return (
            "Here is our previous conversation:\n\n" + history_text,
            intent,
            []
        )

    # Step 4: Handle document-based questions
    elif intent == "DOCUMENT":

        # Rewrite follow-up question
        standalone_question = contextualize_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        # Generate embedding for the question
        query_embedding = embeddings.embed_query(
            standalone_question
        )

        # Search document chunks in Supabase
        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": 3
            }
        ).execute()

        documents = result.data or []

        # Check whether matching documents were found
        if not documents:
            return (
                "I don't know based on the provided documents.",
                intent,
                []
            )

        # Extract retrieved text
        context = "\n\n".join([
            doc["content"] for doc in documents
        ])

        # Generate answer using retrieved context
        response = document_answer_chain.invoke({
            "input": standalone_question,
            "context": context
        })

        # Extract document sources
        sources = [
            doc.get("metadata") or {}
            for doc in documents
        ]

        return response, intent, sources

    # Step 5: Handle unexpected intent
    else:
        return (
            "Sorry, I could not identify your question type.",
            intent,
            []
        )


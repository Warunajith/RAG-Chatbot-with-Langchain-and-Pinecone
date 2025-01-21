from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage


def get_prompt():

    # Define the system prompt
    system_prompt = (
        "You are an intelligent chatbot. Use the following context to answer the question. If you don't know the answer, just say that you don't know."
        "\n\n"
        "{context}"
    )

    prompt=ChatPromptTemplate.from_messages(

        [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
        ]
    )

    return prompt




def get_contextualize_prompt():

    # Define the contextualize system prompt
    contextualize_system_prompt = (
        "using chat history and the latest user question, just reformulate question if needed and otherwise return it as it is"
    )

    # Create the contextualize prompt template
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_prompt

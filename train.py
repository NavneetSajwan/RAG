import os
from dotenv import load_dotenv

import pinecone
from langchain.chat_models import  ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

INDEX_NAME = 'llama-2-rag'

chat = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model = "gpt-3.5-turbo"
)

#get pinecone index
index = pinecone.Index(INDEX_NAME)

embed_model  = OpenAIEmbeddings(
    model = "text-embedding-ada-002"
)

text_field="text"
vectorstore = Pinecone(
    index, embed_model.embed_query,
    text_field
    )


def augment_prompt(query: str):
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augment_prompt = f""" Using the context below, answer the query.

context: {source_knowledge}

query: {query}
"""
    return augment_prompt

messages = [
    SystemMessage(content="You are a helpful assistant. Your name is Friday"),
    HumanMessage(content= "Hi Friday, how are you today?"),
    SystemMessage(content="Hi, I'm great thank you. How can I help you?")
]

def answer_me(query):
    prompt = augment_prompt(query)
    messages.append(HumanMessage(content=prompt))
    return chat(messages).content
    
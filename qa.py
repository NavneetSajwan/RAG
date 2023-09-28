
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from load_index import chat, vectorstore
import warnings
warnings.filterwarnings("ignore")

def augment_prompt(query: str):
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augment_prompt = f""" Using the context below, answer the query.

context: {source_knowledge}

query: {query}
"""
    return augment_prompt

def answer_me(query, messages):
    prompt = augment_prompt(query)
    messages.append(HumanMessage(content=prompt))
    return chat(messages).content

if __name__ == "__main__":
    query = "how is llama 2 better than other llms?"
    messages = [
        SystemMessage(content="You are a helpful assistant. Your name is Friday"),
        HumanMessage(content= "Hi Friday, how are you doing today?"),
        SystemMessage(content="Hi, I'm great thank you. How can I help you?")
    ]
    out = answer_me(query, messages)
    print("Processing query...")
    print(out)
    
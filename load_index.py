import os
from dotenv import load_dotenv

import pinecone
from langchain.chat_models import  ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

INDEX_NAME = 'llama-2-rag'

print(os.environ.get("OPENAI_API_KEY"))
print("===oooøøøøøø===================≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠")
chat = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model = "gpt-3.5-turbo"
)

#init pinecone object
pinecone.init(
    api_key=os.environ.get("PINCONE_API_KEY"),
    environment="gcp-starter"
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






    
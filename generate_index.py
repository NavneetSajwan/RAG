from datasets import load_dataset
import pinecone
import time
import os

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar

#declare variables
DATAPATH = "jamescalam/llama-2-arxiv-papers-chunked"
INDEX_NAME = 'llama-2-rag'

#initialize pinecone index
pinecone.init(
    api_key=os.environ.get("PINCONE_API_KEY"),
    environment="gcp-starter"
)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        INDEX_NAME,
        dimension=1536,
        metric='cosine'
    )

while not pinecone.describe_index(INDEX_NAME).status['ready']:
    time.sleep(1)

# get pinecode index
index = pinecone.Index(INDEX_NAME)

embed_model  = OpenAIEmbeddings(
    model = "text-embedding-ada-002"
)

#load tabular data into memory
dataset = load_dataset(
    DATAPATH,
    split="train"
)

data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100

#create index of the dataset
for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['chunk'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

print("Index for the Dataset create succesully: ", INDEX_NAME)
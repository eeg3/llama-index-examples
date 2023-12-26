from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os

# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.llms import Gemini, OpenAI
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    set_global_service_context,
)
from llama_index.embeddings import GeminiEmbedding, GooglePaLMEmbedding

# LLM and Embedding Model Settings
google_api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
embed_model = GooglePaLMEmbedding(model_name="models/embedding-gecko-001", api_key=google_api_key)
# embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=google_api_key)
# llm = Gemini(api_key=google_api_key)
llm = OpenAI(model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    lyft_docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data() # Load Data
    index = VectorStoreIndex.from_documents(lyft_docs)
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    wiki_docs = loader.load_data(pages=['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago'])
    for d in wiki_docs:
        index.insert(document = d)
    index.storage_context.persist(persist_dir="./storage")

index_engine = index.as_chat_engine(chat_mode="react", verbose=True)

baseline = " If the tool can't find the answer or do something then don't use the tool."
question = "What is Berlin the capital of?"
response = index_engine.chat(question + baseline)
print(str(response))
question = "What country is directly south of it?"
response = index_engine.chat(question + baseline)
print(str(response))
question = "What is 2+2"
response = index_engine.chat(question + baseline)
print(str(response))
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os

# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.llms import Gemini, OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool, ToolMetadata
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

google_api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
embed_model = GooglePaLMEmbedding(model_name="models/embedding-gecko-001", api_key=google_api_key)
# embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=google_api_key)
llm = Gemini(api_key=google_api_key)
# llm = OpenAI(model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage/lyft")
    lyft_index = load_index_from_storage(storage_context)
    storage_context = StorageContext.from_defaults(persist_dir="./storage/wiki")
    wiki_index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False

print(f"Index Loaded: {index_loaded}")

if not index_loaded:
    lyft_docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data() # Load Data
    lyft_index = VectorStoreIndex.from_documents(lyft_docs) # Build Index
    lyft_index.storage_context.persist(persist_dir="./storage/lyft") # Persist Index
    
    WikipediaReader = download_loader("WikipediaReader") # https://llamahub.ai/l/wikipedia
    loader = WikipediaReader()
    wiki_docs = loader.load_data(pages=['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago'])
    wiki_index = VectorStoreIndex.from_documents(wiki_docs)
    wiki_index.storage_context.persist(persist_dir="./storage/wiki")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
wiki_engine = wiki_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=("Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=wiki_engine,
        metadata=ToolMetadata(
            name="wiki_cities",
            description=("Provides information about cities. "
            "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
baseline = " If the tool can't find the answer or do something then don't use the tool."
questions = ["What is Berlin the capital of?", "What country is directly south of it?",
             "Who was George Washington?", "What was Lyft's revenue growth in 2021?"]

for q in questions:
    response = agent.chat(q + baseline)
    print(str(response))
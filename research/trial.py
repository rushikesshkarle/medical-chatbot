from cmd import PROMPT
from glob import glob
from importlib import import_module
from json import load
from lib2to3.pgen2.token import OP
from pydoc import doc
import re
from unittest import loader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter  import RecursiveCharacterTextSplitter

def load_pdf_files(data):
    loader=DirectoryLoader(data,
    glob="*.pdf",
    loader_cls=PyPDFLoader
    )
    
    documents=loader.load()
    return documents

extract_data=load_pdf_files("data")
print(f"Total number of documents: {len(extract_data)}")

from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
   minimal_docs: List[Document] = []
   for doc in docs:
       src= doc.metadata.get("source")
       minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))

   return minimal_docs

minimal_docs=filter_to_minimal_docs(extract_data)
print(f"Total number of minimal documents: {len(minimal_docs)}")


## Smaller chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(minimal_docs):
    # create the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )
    # split the documents
    texts_chunks = splitter.split_documents(minimal_docs)
    return texts_chunks

# Example usage
texts_chunks = split_text(minimal_docs)
print(f"Total number of text chunks: {len(texts_chunks)}")


from langchain_huggingface import HuggingFaceEmbeddings


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = download_embeddings()

vector=embeddings.embed_query("Hello world")

print(f"Vector length: {(vector)}")



from dotenv import load_dotenv
import os
load_dotenv()

PINECODE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECODE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

from pinecone import Pinecone

pc = Pinecone(api_key="PINECODE_API_KEY")
index = pc.Index("quickstart")

from pinecone import Pinecone
pinecone_api_key=PINECODE_API_KEY

pc=Pinecone(api_key=pinecone_api_key)


from pinecone import ServerlessSpec

index_name="medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-1")
    )

    index=pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embeddings=embeddings,  
    index_name="medical-chatbot"
)


# Load Exisiting index

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    embeddings=embeddings,  
    index_name="medical-chatbot"
)


retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})

retriever_docs=retriever.invoke("What is diabetes?")

print(retriever_docs)



from langchain_openai import ChatOpenAI

chatModel=ChatOpenAI(model_name="gpt-3.5-turbo")



from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate





system_prompt=(" You are a helpful AI assistant. Use the following context to answer the users question. ""{context}")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)


question_answer_chain=create_stuff_documents_chain(chatModel,prompt)

rag_chain=create_retrieval_chain(retriever,question_answer_chain)
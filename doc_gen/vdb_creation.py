from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS


# Loading embeddings from Huggingface
embeddings = HuggingFaceEmbeddings()

# Document Loader
loader = TextLoader('../data/jd_data.txt')
documents = loader.load()


# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Creating and saving embeddings into VDB
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

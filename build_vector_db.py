from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformer")

# 向量数据库持久化路径
persist_directory = "sd"

# 加载数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

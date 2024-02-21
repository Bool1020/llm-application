import os
from tqdm import tqdm
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from ...config.model_config import model_config

embedding_model = model_config.embedding_model
embeddings = HuggingFaceEmbeddings(model_name=r'../pretrained_models/{embedding_model}'.format(embedding_model=embedding_model))


def load_docs(path):
    if path.endswith('.txt'):
        loader = TextLoader(path)
    elif path.endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif path.endswith('.docx') or path.endswith('.doc'):
        loader = Docx2txtLoader(path)
    return loader.load()


def create_knowledge(dir_path, kg_name, split_strategy={'mode': 'auto'}):
    if split_strategy.get('mode') == 'auto':
        splitter = RecursiveCharacterTextSplitter(
            separators=['。', '!', '\n'],
            keep_separator=False,
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
    elif split_strategy.get('mode') == 'user-defined':
        splitter = RecursiveCharacterTextSplitter(
            separators=split_strategy.get('separators'),
            keep_separator=False,
            chunk_size=split_strategy.get('chunk_size'),
            chunk_overlap=split_strategy.get('chunk_overlap'),
            length_function=len,
        )
    else:
        return {'code': 0, 'message': '处理失败'}
    document_split = []
    dir_path = dir_path + '/{}'.format(kg_name)
    for path in tqdm(os.listdir(dir_path)):
        document = load_docs(dir_path+'/'+path)
        document_split += splitter.split_documents(document)
    faiss_index = FAISS.from_documents(document_split, embeddings)
    faiss_index.save_local('knowledge_base/vector_db/{}_index'.format(kg_name))
    return {'code': 1, 'message': '处理成功'}


def add_knowledge(path, kg_name, split_strategy={'mode': 'auto'}):
    if split_strategy.get('mode') == 'auto':
        splitter = RecursiveCharacterTextSplitter(
            separators=['。', '!', '\n'],
            keep_separator=False,
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
    elif split_strategy.get('mode') == 'user-defined':
        splitter = RecursiveCharacterTextSplitter(
            separators=split_strategy.get('separators'),
            chunk_size=split_strategy.get('chunk_size'),
            chunk_overlap=split_strategy.get('chunk_overlap'),
            length_function=len,
        )
    else:
        return {'code': 0, 'message': '处理失败'}
    document = load_docs(path)
    document_split = splitter.split_documents(document)
    faiss_index = FAISS.from_documents(document_split, embeddings)
    try:
        faiss_last = FAISS.load_local("knowledge_base/vector_db/{}_index".format(kg_name), embeddings)
        faiss_index.merge_from(faiss_last)
    except:
        pass
    faiss_index.save_local('knowledge_base/vector_db/{}_index'.format(kg_name))
    return {'code': 1, 'message': '处理成功'}


def load_knowledge(kg_name):
    return FAISS.load_local("knowledge_base/vector_db/{}_index".format(kg_name), embeddings)

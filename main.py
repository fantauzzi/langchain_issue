import logging
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import Chroma
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning
error = _logger.error

config_path = Path('.')
load_dotenv(config_path / '.env')

if not os.environ.get('OPENAI_API_KEY'):
    error('OPENAI_API_KEY is not set. It should be set in the shell or it can be set in the .env file')
    exit(-1)

data_path = Path('.')

# Load the questions and answers for model validation from movies_qa.yaml
movies_qa_path = data_path / 'movies_qa.yaml'
movies_qa = OmegaConf.load(movies_qa_path)
movies_qa = [{'Question': item.Question.rstrip('\n )'), 'Answer': item.Answer.rstrip('\n ')} for item in movies_qa]

# Instantiate the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')  # This is right for GPT-3.5


def tokenized_length(s: str) -> int:
    """
    Returns the length in tokens in a given string after tokenization.
    :param s: the given string.
    :return: the count of tokens in the tokenized string.
    """
    tokenized = tokenizer.encode(s)
    return len(tokenized)


embeddings_path = data_path / 'chroma'
embedding = OpenAIEmbeddings()

# Load the embeddings...
info(f'Reloading embeddings from {embeddings_path}')
vectordb = Chroma(persist_directory=str(embeddings_path), embedding_function=embedding)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
qa_chain_no_context = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
template = """Use also the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use four sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)

# Description of the metadata that go with the embeddings
metadata_field_info = [
    AttributeInfo(name='title',
                  description='The movie title',
                  type='string'),
    AttributeInfo(name='year',
                  description='The movie release year',
                  type='integer'),
    AttributeInfo(name='id',
                  description='The movide unique ID within Wikipedia',
                  type='integer'),
    AttributeInfo(name='revision_id',
                  description='The movie unique revision ID within Wikipedia',
                  type='integer')
]
document_content_description = 'The movie plot or synopsis'

# Get ready to retrieve contex from the embeddings store based also on metadata
retriever = SelfQueryRetriever.from_llm(llm,
                                        vectordb,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True)

qa_chain_with_context = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)


def run_query_with_context(query: dict) -> dict:
    try:
        res = qa_chain_with_context(query)
    except Exception as ex:
        info(f'Query with context failed with exception {ex}')
        return {'query': query, 'result': 'ERROR!'}
    info(f'Query with context completed')
    return res


# Send all the queries to gpt-3.5
queries = [{'query': qa['Question']} for qa in movies_qa]

results_with_context = []
for query in tqdm(queries):
    res = run_query_with_context(query)
    results_with_context.append(res)

# Gather the received answers and save them in processed_qa.yaml
movies_qa_processed = []
for qa, answer_with_context in zip(movies_qa, results_with_context):
    source_chunks = [doc.page_content for doc in answer_with_context['source_documents']] if answer_with_context.get(
        'source_documents') else None

    movies_qa_processed.append({'Question': qa['Question'],
                                'Answer': qa['Answer'],
                                'Answer_from_llm': answer_with_context['result'],
                                'Context': source_chunks
                                })

processed_qa_file = data_path / 'processed_qa.yaml'
OmegaConf.save(movies_qa_processed, processed_qa_file)

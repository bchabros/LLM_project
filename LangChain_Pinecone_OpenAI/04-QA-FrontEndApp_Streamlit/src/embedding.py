from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import tiktoken


def create_embeddings(chunks_embeddings):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    return Chroma.from_documents(chunks_embeddings, embeddings)


def print_embedding_costs(texts):

    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

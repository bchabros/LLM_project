import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def ask_and_get_answer(vector_store, q, k=3):

    llm = ChatOpenAI(model='gpt-3.5-turbo',
                     temperature=0.5)

    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever)
    answer = chain.invoke(q)
    return answer['result']


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

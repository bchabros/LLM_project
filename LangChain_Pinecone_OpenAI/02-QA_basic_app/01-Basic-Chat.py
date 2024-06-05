from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv(find_dotenv(), override=True)

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessage(content='You respond only in Polish.'),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = prompt | llm

while True:
    content = input('Your prompt: ')
    if content.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break
    
    response = chain.invoke({'content': content})
    print(response)
    print('-' * 50)
    
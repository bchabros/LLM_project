from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain

# 1. Imports
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

# 2. Create memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# 3. add  MessagesPlaceholder(variable_name='messages') to the prompt
prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        SystemMessage(content="You are a chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# 4. Add the memory to the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

while True:
    content = input('Your prompt: ')
    if content.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break

    response = chain.invoke({'content': content})
    print(response)
    print('-' * 50)

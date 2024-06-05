from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False) 

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain

# 1. Import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

# 2. Add a keyword argument to the ConversationBufferMemory() constructor
history = FileChatMessageHistory('chat_history.json')
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        SystemMessage(content="You are a chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"), 
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

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
    
# The messages property contains the list of messages in order.
print(history.messages)

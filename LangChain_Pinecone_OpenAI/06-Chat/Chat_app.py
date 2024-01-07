# import libraries
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
import streamlit as st
from streamlit_chat import message

# loading API key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title='You Custom Assistnat',
    page_icon = '🤖'
)
st.subheader('Your Custom ChatGPT 🤖')

chat = ChatOpenAI(model_name='gpt-4',
                  temperature=0.5)

# creating the messages (chat history) in the streamlit session state 
if 'messages' not in st.session_state:
    st.session_state.messages = []

# creating sidebar
with st.sidebar:
    # streamlit text input widget for the system message (role)
    system_message = st.text_input(label='System role')
    # streamlit text input widget for the user message
    user_prompt = st.text_input(label='Send a message')
    
    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(content=system_message)
                )  

        # for debugging
        # st.write(st.session_state.messages)
            
    # if the user entered a question
    if user_prompt: 
        st.session_state.messages.append(
            HumanMessage(content=user_prompt)
        )

        with st.spinner('Working on you request ...'):
            # creating the ChatGPT response
            response = chat(st.session_state.messages)

        # adding the response content to the session state
        st.session_state.messages.append(AIMessage(content=response.content))

#for debugging
# st.session_state.messages
# message('this is chatgpt', is_user=False)
# message('this is the user', is_user=True)

# adding a default SystemMessage if the user didn't entered one
if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))      

# displaying the messages (chat history)
for i, msg in enumerate(st.session_state.messages[1:]): # index zero is the system message
    if i % 2 == 0: #user message
        message(msg.content, is_user=True, key=f'{i} + 🙂')
    else:
        message(msg.content, is_user=False, key=f'{i} + 🤖')


# to run app
# run Chat_app.py using comand streamlit run Chat_app.py
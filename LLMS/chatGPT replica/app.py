import streamlit as st
import llm


st.set_page_config(page_title="ChatGPT Replica", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .user-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
    }
    .bot-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .chat-container {
        max-width: 700px;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

with st.container():
    st.title("ChatGPT Replica 🤖")
    
    chat_container = st.container()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["is_user"]:
                st.markdown(f'<div class="user-message"><b>You:</b> {chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message"><b>Bot:</b> {chat["message"]}</div>', unsafe_allow_html=True)
    
    user_input = st.text_input("Type your message here...")
    
    if user_input:
        st.session_state.chat_history.append({"message": user_input, "is_user": True})
        
        with st.spinner("Generating response..."):
            bot_response = llm.gen_response(user_input)
        
        st.session_state.chat_history.append({"message": bot_response, "is_user": False})

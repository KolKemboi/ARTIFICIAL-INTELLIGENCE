import streamlit as st

# Streamlit UI Code
st.set_page_config(page_title="ChatGPT Replica", page_icon="🤖", layout="wide")

# Custom CSS to style the chat interface
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

# Create a chat container in the middle of the page
with st.container():
    st.title("ChatGPT Replica 🤖")
    
    # Scrollable chat history
    chat_container = st.container()
    
    # Initialize the session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["is_user"]:
                st.markdown(f'<div class="user-message"><b>You:</b> {chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message"><b>Bot:</b> {chat["message"]}</div>', unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input("Type your message here...")
    
    # If user sends a message
    if user_input:
        # Add user's message to chat history
        st.session_state.chat_history.append({"message": user_input, "is_user": True})
        
        # Generate response from model
        with st.spinner("Generating response..."):
            bot_response = "dfhvujiarhuigvjnaijfoavj"
        
        # Add bot's response to chat history
        st.session_state.chat_history.append({"message": bot_response, "is_user": False})
        
import streamlit as st
from educational_chatbot_backend import get_ai_response, set_student_info

st.set_page_config(layout="wide")

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "info"
if "student_info" not in st.session_state:
    st.session_state.student_info = {}
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def submit_info():
    st.session_state.page = "chat"
    set_student_info(st.session_state.student_info)

def change_agent(agent):
    st.session_state.selected_agent = agent
    st.session_state.messages = []

# Student Information Page
if st.session_state.page == "info":
    st.title("Student Information")
    
    st.session_state.student_info["name"] = st.text_input("Name")
    st.session_state.student_info["age"] = st.number_input("Age", min_value=10, max_value=25)
    st.session_state.student_info["class"] = st.selectbox("Class", options=[f"{i}th" for i in range(8, 13)] + ["College"])
    st.session_state.student_info["stream"] = st.selectbox("Stream", options=["Science", "Commerce", "Arts", "Not Applicable"])
    
    if st.button("Submit"):
        submit_info()

# Chat Page
elif st.session_state.page == "chat":
    st.title("AI Career Guidance Assistant")

    # Sidebar for agent selection
    st.sidebar.title("Select an Agent")
    if st.sidebar.button("Career Counselor"):
        change_agent("Career Counselor")
    if st.sidebar.button("Skill Analyst"):
        change_agent("Skill Analyst")
    if st.sidebar.button("Industry Expert"):
        change_agent("Industry Expert")

    # Display current agent
    if st.session_state.selected_agent:
        st.sidebar.write(f"Current Agent: {st.session_state.selected_agent}")
    else:
        st.sidebar.write("Please select an agent to start the conversation.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if st.session_state.selected_agent:
        prompt = st.chat_input("What would you like to know?")
        if prompt:
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get AI response
            full_prompt = f"!{st.session_state.selected_agent.lower().replace(' ', '_')} {prompt}"
            response = get_ai_response(full_prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Instructions
    st.sidebar.title("How to use")
    st.sidebar.markdown("""
    1. Select an agent from the sidebar.
    2. Ask questions related to career guidance.
    3. Change the agent at any time to get different perspectives.
    """)
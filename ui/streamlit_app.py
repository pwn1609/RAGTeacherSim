import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "expert_chat_history" not in st.session_state:
    st.session_state.expert_chat_history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

# Load scenario data
@st.cache_data
def load_scenarios():
    try:
        with open("data/scenario_menu.json", "r") as f:
            menu_data = json.load(f)
        with open("data/scenarios.json", "r") as f:
            scenarios_data = json.load(f)
        return menu_data["scenario_menu"], {s["scenario_id"]: s for s in scenarios_data}
    except Exception as e:
        st.error(f"Error loading scenarios: {e}")
        return [], {}

scenario_menu, scenarios_dict = load_scenarios()

# Create the main layout
st.title("Teacher-Student Chat Simulator")

# Define a callback for expert chat submission
def on_expert_advice_submit():
    if st.session_state.expert_question and st.session_state.current_scenario:
        # Add the question to the chat history
        st.session_state.expert_chat_history.append({"role": "user", "content": st.session_state.expert_question})
        
        # Get expert advice from backend
        try:
            expert_response = requests.post(f"{API_URL}/expert-advice", json={
                "question": st.session_state.expert_question,
                "conversation_history": st.session_state.chat_history,
                "scenario_id": st.session_state.current_scenario["scenario_id"]
            }).json()["response"]
            
            # Add the response to the expert chat history
            st.session_state.expert_chat_history.append({"role": "assistant", "content": expert_response})
            
            # Clear the input field
            st.session_state.expert_question = ""
        except Exception as e:
            st.error(f"Error getting expert advice: {e}")

# Create a sidebar for expert teacher
with st.sidebar:
    st.header("Expert Teacher Advisor")
    st.write("Ask for advice on how to handle this teaching scenario")
    
    # Expert chat input
    if st.session_state.current_scenario:
        # Initialize expert_question in session state if it doesn't exist
        if "expert_question" not in st.session_state:
            st.session_state.expert_question = ""
            
        # Use a form to handle submission properly
        with st.form(key="expert_form", clear_on_submit=True):
            st.text_input("Ask for teaching advice:", key="expert_question")
            submit_button = st.form_submit_button("Ask", on_click=on_expert_advice_submit)
    else:
        st.info("Select a scenario to get expert advice")
    
    # Display expert chat history
    if st.session_state.expert_chat_history:
        st.subheader("Advice History")
        for msg in st.session_state.expert_chat_history:
            if msg["role"] == "user":
                st.write("**You:** " + msg["content"])
            else:
                st.write("**Expert:** " + msg["content"])
                st.divider()

# Main content area
# Scenario selection dropdown
scenario_options = ["Select a scenario..."] + [f"{s['title']}" for s in scenario_menu]
selected_scenario = st.selectbox("Choose a scenario:", scenario_options, index=0)

# Process scenario selection
if selected_scenario != "Select a scenario..." and (st.session_state.current_scenario is None or 
                                                 selected_scenario != st.session_state.current_scenario["title"]):
    # Find the selected scenario
    for menu_item in scenario_menu:
        if menu_item["title"] == selected_scenario:
            scenario_id = menu_item["scenario_id"]
            if scenario_id in scenarios_dict:
                st.session_state.current_scenario = scenarios_dict[scenario_id]
                # Clear chat histories when changing scenarios
                st.session_state.chat_history = []
                st.session_state.expert_chat_history = []
                st.rerun()

# Display current scenario information
if st.session_state.current_scenario:
    with st.expander("Current Scenario", expanded=True):
        st.write(f"**{st.session_state.current_scenario['title']}**")
        st.write(st.session_state.current_scenario['description'])
        
        # Display additional information based on scenario type
        if "type" in st.session_state.current_scenario:
            scenario_type = st.session_state.current_scenario["type"]
            if scenario_type == "Student Emotion" and "emotion" in st.session_state.current_scenario:
                st.write(f"**Student Emotion:** {st.session_state.current_scenario['emotion']}")
            elif scenario_type == "Teaching Concept":
                st.write("**Teaching a Concept**")
            elif scenario_type == "Classroom Management":
                st.write("**Classroom Management Situation**")

# Main student chat area
st.header("Student Conversation")

# Display the chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Define a callback for student chat message submission
def on_student_chat_submit():
    if st.session_state.current_scenario and st.session_state.student_chat_input:
        user_input = st.session_state.student_chat_input
        
        # Add the user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": f"Teacher: {user_input}"})
        
        try:
            # Get response from backend with scenario context
            response = requests.post(f"{API_URL}/student-response", json={
                "user_input": user_input,
                "chat_history": st.session_state.chat_history,
                "scenario_id": st.session_state.current_scenario["scenario_id"]
            }).json()["response"]
            
            # Add the response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error getting student response: {e}")

# Initialize student input key if needed
if "student_chat_input" not in st.session_state:
    st.session_state.student_chat_input = ""

# Chat input
if st.session_state.current_scenario:
    prompt = st.chat_input("Message the student...", key="student_chat_input", on_submit=on_student_chat_submit)
else:
    st.info("Please select a scenario to begin chatting with the student.")
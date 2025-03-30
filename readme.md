### BackEnd API Contract Documentation

#### POST `/student-response`
- **Description**: Generates a simulated student response based on teacher input and prior chat history.
- **Request Body**:
```json
{
  "user_input": "string",
  "chat_history": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```
- **Response**:
```json
{
  "response": "string"
}
```

---

#### POST `/expert-advice`
- **Description**: Returns advice from an expert teacher based on a question and the current conversation history.
- **Request Body**:
```json
{
  "question": "string",
  "conversation_history": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```
- **Response**:
```json
{
  "response": "string"
}
```

---

### 🚀 Backend Startup Instructions

1. **Install dependencies**:
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu ollama
```

2. **Start the backend server**:
```bash
uvicorn backend.main:app --reload
```
- Runs at: http://localhost:8000
- Docs available at: http://localhost:8000/docs

---

### 🎨 Frontend Startup Instructions

1. **Install dependencies**:
```bash
pip install streamlit requests
```

2. **Run the Streamlit frontend**:
```bash
streamlit run ui/streamlit_app.py
```

---

### User Interface Instructions
- **Start the chatbot**: When you start the app you will see this screen. Click the **Select a scenario...** dropdown and select a desired scenario to begin interacting with the second-grader chatbot.  
![Image of the app after it's started.](/Images/start.png)
- **Scenario Description**: After selecting your desired scenario a description of the scenario will appear. Read through the description and decide how you would like to interact with the chatbot.
![Image of the scenario description.](/Images/scenario%20description.png)
- **Interact with the chatbot**: You can now interact with the chatbot by clicking into the Message the student... box and typing out your message. To send the message, press enter on your keyboard or click the send message arrow on the right side of the box.
![Image of using the message function.](/Images/message%20chatbot.png)
- **Open Expert Teacher Advisor Chat**: If you would like advice on the student interaction, click the arrow on the top left of the screen to open the Expert Teacher Chatbot. Ask any questions you have, and it will provide adviced based on various teaching textbooks.
![Image of button to click for expert teacher sidebar.](/Images/expert%20sidebar.png)
- **Ask the Expert a Question**: With the Expert Teacher Advisor open you can ask a question by clicking into the **Ask for teaching advice:** box, typing your message and pressing enter on your keyboard or the **Ask** button.
![Image of the expert teacher sidebar.](/Images/expert%20chat.png)
- **Access Settings**: To access appearance settings, click the 3 dots in the top right.
![Image showing where the three dots are.](/Images/settings.png)
- **Access Settings**: Click on the settings button.
![Image showing where the settings button is.](/Images/settings2.png)
- **Appearance Settings**: You may now put the screen in wide mode, change to a preset theme, or create your own custom theme. 
![Image showing the settings window.](/Images/settings3.png)
- **Custom Theme**: You can change the colors of the screen and use a different font family. 
![Image showing the custom theme page.](/Images/settings4.png)
- **Preset Theme**: You can select the light theme, dark theme, or have the theme reflect your system settings.
![Image showing the use of the theme dropdown.](/Images/settings5.png)
- **Example of light theme and wide mode**
![Image of light theme and wide mode.](/Images/settings6.png)
- **Continue Chatting**: Continue using our chatbots as you would like!
# MemoryCompanion - A tool to help people remember things
MemoryCompanion is a tool for people suffering with Mild Cognitive Impairment (MCI), a condition in which users face memory or thinking related problems but not to the point that it affects their daily activities. These people tend to miss social events, forget appointments and have issues with remembering names or conversations with other people. MemoryCompanion helps these people keep track of conversations, names, tasks to do and much more!

## Features

* **Two modes**: 
1) Add - In this mode the user is able to store information that he feels is important to his life. For example "Chris told me he is getting married on 24th November 2024".
2) Query - In this mode the user can talk with an interactive llm to know things like - "Do I have anything important on 24th november?" Answer - "Chris is getting married on 24th Nov"
   
   Another example is:
   - Add - "John told me to work on my programming skills to improve efficiency and that he will visit me later"
   - Query - "What did John tell me, I don't remember properly"
   - Answer - "John told you work on your programming skills"


 * **Conversational Chatbot**:
   Llama70B is used for conversational capabilities which gives the users a realistic touch with the technology. It is used through (GROQ API)[https://groq.com].

 * **To-Do List**:
   Innovative To-Do list that gets updated from the same conversation chatbot's input field, to make it easier for the user.

 * **Calendar**:
   A simple calendar for users to easily give date to their information addition so that they can give the chatbot more context.

## Installation
1. Clone the Repository and change directory
   ```python
   git clone https://github.com/mani2001/MemoryCompanion.git
   cd MemoryCompanion
   ```
2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
     # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
   ```
3. Installation of required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a .env file in the root directory:
   ```bash
   GROQ_API_KEY=<Insert your API key here>
   ```
5. Run the application:
   ```bash
   python app.py
   ```
6. Open the localhost:
   ```bash
    http://127.0.0.1:8081 (use the localhost that shows up in your terminal in place of this
   ```
   The initialization may take upto 20 second the first time.
## Usage
**Select Modes** - Either select 'Add' if you want to add an information to the database and select 'Question' if you want to query the LLM.

**Chatbot** - Enter your query or information in the input chatbox.

**To-do** - Scroll through your To-Do list to check time-stamped reminders. You can click on 'X' mark to cancel a reminder.

**Calendar** - Use the calendar to look at dates and days and give commands to the Chat assistant like "I need to buy groceries on 20th november".

## Architecture
**Frontend** : HTML, CSS, JS
**Backend** : Flask
**LLM**: Llama 70B with GROQ API

## Components
**Web Interface** - built using vanilla js, html and css, contains of a three panels, a To-Do List, A chatbox and a calendar. Simple and easy to use.
**LLM** - Using Llama3 70B by Meta with GROQ platform. Llama3 70B is a large language model with 70 billion parameters and is capable of answering complex questions.

## Screenshot

<img width="1409" alt="Screenshot 2024-11-24 at 10 09 04 PM" src="https://github.com/user-attachments/assets/2597ff0b-d81d-47a2-abf0-82add8cc09bc">

## Future scope
- Adding speech to text and text to speech for blind or elderly users who find it hard to use the web interface.
- Add options to delete information from the database (for now, once it is stored you cannot remove it through the interface)
- Better website with simpler components

## Considerations
- The chroma vectorstore used has a limit and a lot of data cannot be stored so user may need to delete the database.txt file or reduce it from time to time (which can be improved with persistent storage and so on)
- Need to first choose 'Add' or 'Question' mode to send inputs

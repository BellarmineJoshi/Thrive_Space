import streamlit as st
import numpy as np
import openai
import random
from textblob import TextBlob
import sqlite3
import pickle
import json
import difflib
from datetime import datetime

# Set up OpenAI client with API key
client = openai.OpenAI(api_key="my_api_key")  # Replace with your actual API key

# Q-table storage file (for RL learning)
Q_TABLE_FILE = "q_table.pkl"

#####################################
# Database Functions for Employee Profiles and Conversation History
#####################################

def init_db():
    """Initialize the SQLite database and create necessary tables if they don't exist."""
    conn = sqlite3.connect("employee_profiles.db")
    cur = conn.cursor()
    # Create employees table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            name TEXT PRIMARY KEY,
            department TEXT,
            role TEXT
        )
    ''')
    # Create conversation_history table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_name TEXT,
            conversation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_employee_profile(name):
    """Retrieve the employee profile by name."""
    conn = sqlite3.connect("employee_profiles.db")
    cur = conn.cursor()
    cur.execute("SELECT name, department, role FROM employees WHERE name = ?", (name,))
    result = cur.fetchone()
    conn.close()
    return result  # Returns (name, department, role) or None

def insert_employee_profile(name, department, role):
    """Insert a new employee profile."""
    conn = sqlite3.connect("employee_profiles.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO employees (name, department, role) VALUES (?, ?, ?)", (name, department, role))
    conn.commit()
    conn.close()

def get_employee_history(name, limit=5):
    """Retrieve the last few conversation entries for the employee."""
    conn = sqlite3.connect("employee_profiles.db")
    cur = conn.cursor()
    cur.execute("SELECT conversation FROM conversation_history WHERE employee_name = ? ORDER BY timestamp DESC LIMIT ?", (name, limit))
    rows = cur.fetchall()
    conn.close()
    return "\n".join(reversed([row[0] for row in rows]))

def store_conversation(name, conversation_text):
    """Store a new conversation entry for the employee."""
    conn = sqlite3.connect("employee_profiles.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO conversation_history (employee_name, conversation) VALUES (?, ?)", (name, conversation_text))
    conn.commit()
    conn.close()

#####################################
# Q-table Functions
#####################################

def initialize_q_table():
    states = ["very_low", "low", "medium", "high", "very_high"]
    actions = ["breathing", "stretching", "time_management", "meditation", "talking_to_friend", "engage_in_hobby"]
    expected_shape = (len(states), len(actions))
    
    try:
        with open(Q_TABLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        if q_table.shape != expected_shape:
            st.warning("Loaded Q-table shape does not match expected shape. Reinitializing Q-table.")
            q_table = np.zeros(expected_shape)
    except FileNotFoundError:
        q_table = np.zeros(expected_shape)
    
    return states, actions, q_table

def save_q_table(q_table):
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(q_table, f)

def get_state_from_sentiment(sentiment):
    if sentiment < -0.6:
        return 4
    elif sentiment < -0.3:
        return 3
    elif sentiment < 0.3:
        return 2
    elif sentiment < 0.6:
        return 1
    else:
        return 0

def map_rl_response_to_action(rl_response, actions):
    best_match = None
    highest_ratio = 0
    for act in actions:
        ratio = difflib.SequenceMatcher(None, rl_response.lower(), act.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = act
    if highest_ratio > 0.2:
        return best_match
    return None

def update_q_table(state, action_idx, reward, next_state, q_table):
    gamma = 0.9
    alpha = 0.1
    q_table[state, action_idx] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action_idx])
    save_q_table(q_table)
    return q_table

#####################################
# Chatbot Functions (Personalized for Each Employee)
#####################################

def choose_action(state, user_input, q_table, actions, employee_name, department, role, history=""):
    """
    Generate an RL response for the given query and return a tuple:
    (RL response, debug_info dictionary)
    """
    # Extract additional context: noun phrases from the user input.
    blob = TextBlob(user_input)
    noun_phrases = blob.noun_phrases
    context_info = f"🔎 Noun Phrases: {', '.join(noun_phrases)}" if noun_phrases else "🔎 No noun phrases detected."
    
    # Lower temperature for deterministic output.
    temperature_rl = 0.2
    
    # Define simple greetings, boredom keywords, and check for prior attempt words.
    greetings = ["hello", "hi", "hey", "greetings"]
    boredom_keywords = ["bored", "boring", "uninterested", "dull", "tedious"]
    prior_attempt_keywords = ["tried", "already", "attempted"]
    lower_input = user_input.lower().strip()
    
    if lower_input in greetings:
        debug_info = {
            "💬 User Input": user_input,
            "🔎 Extracted Noun Phrases": noun_phrases,
            "🗂 Context Info": context_info,
            "🕰 History Text": history,
            "📊 Computed State": state,
            "📝 Detailed Instruction": "Greeting detected – returning welcome message.",
            "🖥 Full Prompt": ""
        }
        return f"Hello {employee_name}! How can I help you today?", debug_info

    few_shot = (
        "Few-shot examples:\n"
        "1️⃣ Input: 'I'm overwhelmed by deadlines at work.'\n"
        "   Response: 'I'm sorry you're feeling overwhelmed. Try breaking your tasks into smaller parts and schedule a 10‑minute break at 3pm to recharge.'\n\n"
        "2️⃣ Input: 'I feel isolated in the office.'\n"
        "   Response: 'I understand how isolation can affect you. Consider arranging a virtual coffee chat with a colleague to reconnect.'\n\n"
        "3️⃣ Input: 'I'm bored at work and need a change.'\n"
        "   Response: 'It seems you need a change. Perhaps ask your manager for new responsibilities or schedule a 10‑minute mindfulness break at 2pm to refresh your mind.'\n\n"
    )
    
    history_text = f"🗂 Conversation History: {history}\n" if history else ""
    
    base_instructions = (
        f"You are a highly experienced mental health assistant for company employees. Your client is {employee_name}, who works in the {department} department as a {role}. "
        "Analyze the user's message and any provided conversation history to identify key emotional cues and work-related context. "
        "Then produce a response in two sentences: the first sentence should empathetically acknowledge and validate the user's feelings, "
        "and the second sentence must offer one specific, measurable, evidence-based actionable recommendation tailored for a corporate environment. "
        "Reference best practices such as mindfulness, effective task management, or structured communication. "
        "Include at least one concrete measurable step (e.g., 'schedule a 10‑minute break at 3pm'). "
        "Return only your final combined response in a clear, friendly tone."
    )
    
    additional_instruction = (
        "Ensure your recommendation is detailed and actionable, including a specific time or measurable step. "
        "If the user indicates that previous strategies have been tried (using words like 'tried', 'already', 'attempted'), suggest an alternative approach."
    )
    
    if any(keyword in lower_input for keyword in boredom_keywords):
        detailed_instruction = (
            f"User said: \"{user_input}\"\n"
            f"{history_text}{context_info}\n"
            f"📊 Computed State is {state}\n"
            "Provide a response that first empathetically acknowledges the user's boredom, then offers a specific, measurable recommendation to boost engagement at work."
        )
    elif any(keyword in lower_input for keyword in prior_attempt_keywords):
        detailed_instruction = (
            f"User said: \"{user_input}\"\n"
            f"{history_text}{context_info}\n"
            f"📊 Computed State is {state}\n"
            "Since you mentioned trying previous methods, provide an alternative response that validates your continued frustration, "
            "and then offers a new specific, measurable recommendation (e.g., prepare a report of your achievements and schedule a follow-up meeting with your manager)."
        )
    else:
        detailed_instruction = (
            f"User said: \"{user_input}\"\n"
            f"{history_text}{context_info}\n"
            f"📊 Computed State is {state}\n"
            "Provide a response that first validates the user's feelings, then offers one specific, measurable recommendation to help alleviate work-related stress."
        )
    
    prompt = few_shot + base_instructions + "\n" + additional_instruction + "\n\n" + detailed_instruction

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an RL agent specialized in corporate mental health support."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        temperature=temperature_rl,
    )
    
    suggested_response = response.choices[0].message.content.strip()
    for prefix in ["action:", "action", ":", "'", "\""]:
        suggested_response = suggested_response.replace(prefix, "")
    suggested_response = suggested_response.strip()
    
    debug_info = {
        "💬 User Input": user_input,
        "🔎 Extracted Noun Phrases": noun_phrases,
        "🗂 Context Info": context_info,
        "🕰 History Text": history_text,
        "📊 Computed State": state,
        "📝 Detailed Instruction": detailed_instruction,
        "🖥 Full Prompt": prompt
    }
    
    return suggested_response, debug_info

def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and empathetic mental health assistant specialized in supporting company employees."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

def auto_evaluate_responses(user_input, openai_response, rl_response):
    eval_prompt = (
        f"Evaluate the following responses for a corporate mental health support chatbot.\n\n"
        f"User Input: {user_input}\n\n"
        f"OpenAI Response: {openai_response}\n\n"
        f"RL Response: {rl_response}\n\n"
        "For each response, provide a score out of 10 for the following criteria:\n"
        "1. Contextual Relevance\n"
        "2. Empathy\n"
        "3. Actionable Advice\n"
        "Then compute a composite score (average of the three scores) for each response. "
        "Return the result in JSON format as:\n"
        "{\"openai\": {\"context\": <value>, \"empathy\": <value>, \"action\": <value>, \"composite\": <value>},\n"
        " \"rl\": {\"context\": <value>, \"empathy\": <value>, \"action\": <value>, \"composite\": <value>}}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an unbiased evaluator for mental health chatbot responses."},
            {"role": "user", "content": eval_prompt}
        ],
        max_tokens=150,
    )
    eval_text = response.choices[0].message.content.strip()
    try:
        evaluation = json.loads(eval_text)
    except Exception as e:
        evaluation = None
    return evaluation

#####################################
# Helper: Display RL Debug Info in a Visually Pleasing Format
#####################################

def display_rl_debug_info(debug_info):
    st.markdown("#### 🔍 RL Debug Information")
    st.markdown(f"💬 User Input:** {debug_info.get('💬 User Input', '')}")
    st.markdown(f"🔎 Extracted Noun Phrases:** {debug_info.get('🔎 Extracted Noun Phrases', '')}")
    st.markdown(f"🗂 Context Info:** {debug_info.get('🗂 Context Info', '')}")
    st.markdown(f"🕰 History Text:** {debug_info.get('🕰 History Text', '')}")
    st.markdown(f"📊 Computed State:** {debug_info.get('📊 Computed State', '')}")
    st.markdown(f"🎯 Action Taken:** {debug_info.get('Action Taken', 'N/A')}")
    st.markdown(f"💰 Reward:** {debug_info.get('Reward', 'N/A')}")
    st.markdown(f"🚀 Goal State (Next State):** {debug_info.get('Goal State (Next State)', 'N/A')}")
    st.markdown("📝 Detailed Instruction:")
    st.code(debug_info.get("📝 Detailed Instruction", ""), language="text")
    st.markdown("🖥 Full Prompt Sent to RL Model:")
    st.code(debug_info.get("🖥 Full Prompt", ""), language="python")

#####################################
# Main Streamlit App
#####################################

def main():
    st.title("💬THRIVE SPACE")
    st.write("Welcome to Thrive Space. I'm here to help you manage work-related stress and enhance your well-being.")
    
    init_db()  # Initialize the database
    
    if "employee_name" not in st.session_state or st.session_state.employee_name == "":
        st.session_state.employee_name = st.text_input("Please enter your name:")
        if not st.session_state.employee_name:
            st.warning("Please enter your name to continue.")
            st.stop()
    
    profile = get_employee_profile(st.session_state.employee_name)
    if profile is None:
        st.write("We don't have your profile yet. Please provide your department and role.")
        department = st.text_input("Department:")
        role = st.text_input("Role:")
        if department and role:
            insert_employee_profile(st.session_state.employee_name, department, role)
            st.success("Profile saved!")
            profile = (st.session_state.employee_name, department, role)
        else:
            st.warning("Please provide both department and role to continue.")
            st.stop()
    else:
        st.write(f"Welcome back, {st.session_state.employee_name}!")
    
    employee_name, department, role = profile
    past_history = get_employee_history(employee_name, limit=5)
    
    states, actions, q_table = initialize_q_table()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_responses" not in st.session_state:
        st.session_state.total_responses = 0
    if "current_state" not in st.session_state:
        st.session_state.current_state = 2  # Default to medium state
    if "openai_total_rating" not in st.session_state:
        st.session_state.openai_total_rating = 0.0
    if "rl_total_rating" not in st.session_state:
        st.session_state.rl_total_rating = 0.0
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = {}
    
    conversation_history = "\n".join([f"User: {ch[0]} | OpenAI: {ch[1]} | RL: {ch[2]}" for ch in st.session_state.chat_history])
    full_history = past_history + "\n" + conversation_history if past_history else conversation_history
    
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        sentiment = TextBlob(user_input).sentiment.polarity
        st.session_state.current_state = get_state_from_sentiment(sentiment)
        
        openai_response = generate_response(user_input)
        rl_response, rl_debug = choose_action(st.session_state.current_state, user_input, q_table, actions, employee_name, department, role, history=full_history)
        
        st.session_state.chat_history.append((user_input, openai_response, rl_response, rl_debug))
        st.session_state.total_responses += 1
        
        conversation_entry = f"User: {user_input}\nOpenAI: {openai_response}\nRL: {rl_response}"
        store_conversation(employee_name, conversation_entry)
        
        evaluation = auto_evaluate_responses(user_input, openai_response, rl_response)
        st.session_state.last_evaluation = evaluation
        
        mapped_action = map_rl_response_to_action(rl_response, actions)
        if mapped_action is None:
            mapped_action = "time_management"
            st.write("No close match found in RL response; defaulting to 'time_management' for Q-table update.")
        action_idx = actions.index(mapped_action)
        
        # Compute reward and next state; update RL debug info with RL details.
        if evaluation and "rl" in evaluation:
            composite = evaluation["rl"]["composite"]
            reward_rl = (2 * ((composite - 1) / 9)) - 1
            next_state = max(0, st.session_state.current_state - 1) if reward_rl > 0 else min(len(states) - 1, st.session_state.current_state + 1)
            # Update the latest debug info with RL state transition details
            rl_debug.update({
                "Current State": st.session_state.current_state,
                "Action Taken": mapped_action,
                "Reward": reward_rl,
                "Goal State (Next State)": next_state
            })
            st.session_state.chat_history[-1] = (st.session_state.chat_history[-1][0],
                                                   st.session_state.chat_history[-1][1],
                                                   st.session_state.chat_history[-1][2],
                                                   rl_debug)
            q_table = update_q_table(st.session_state.current_state, action_idx, reward_rl, next_state, q_table)
        else:
            st.write("RL response did not map to a predefined action; Q-table update skipped.")
    
    st.write("### Chat History")
    for user_msg, openai_msg, rl_msg, _ in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"You: {user_msg}")
        with st.chat_message("assistant"):
            st.markdown(f"WellbeingBot (OpenAI): {openai_msg}")
        with st.chat_message("assistant"):
            st.markdown(f"WellbeingBot (RL): {rl_msg}")
    
    if st.session_state.last_evaluation:
        st.write("### Auto Evaluation of Responses")
        st.write("OpenAI Response Evaluation:")
        st.write(st.session_state.last_evaluation.get("openai", {}))
        st.write("RL Response Evaluation:")
        st.write(st.session_state.last_evaluation.get("rl", {}))
    
    if st.session_state.total_responses > 0:
        openai_avg = st.session_state.openai_total_rating / st.session_state.total_responses
        rl_avg = st.session_state.rl_total_rating / st.session_state.total_responses
        st.write("### Average Composite Rewards (Auto Evaluation)")
        st.write(f"✅ OpenAI Average Reward: {openai_avg:.2f} (range: -1 to +1)")
        st.write(f"✅ RL Average Reward: {rl_avg:.2f} (range: -1 to +1)")
    
    # ---------------------------
    # RL Debug Information Section (Enhanced with Icons & RL State Details)
    # ---------------------------
    st.markdown("---")
    st.header("🔍 RL Debug Information for the Latest Query")
    if st.session_state.chat_history:
        latest_debug = st.session_state.chat_history[-1][3]
        with st.expander("Show Detailed RL Debug Info"):
            display_rl_debug_info(latest_debug)

if __name__ == "__main__":
    init_db()  # Initialize the database and tables
    main()
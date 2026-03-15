import streamlit as st
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime, timedelta
import csv
import json
import tempfile
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# ----------------------------
# Groq Client
# ----------------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

RESULT_FILE = "student_results.csv"

# ----------------------------
# Session Memory
# ----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Helper Functions
# ----------------------------

def load_results():

    if os.path.exists(RESULT_FILE):
        return pd.read_csv(RESULT_FILE)

    return pd.DataFrame(columns=["Date","Topic","Marks","Total"])


def save_result(topic, marks, total):

    date = datetime.today().date()

    file_exists = os.path.isfile(RESULT_FILE)

    with open(RESULT_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Date","Topic","Marks","Total"])

        writer.writerow([date,topic,marks,total])


# ----------------------------
# Voice Functions
# ----------------------------

def speech_to_text(audio_bytes):

    r = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        filename = f.name

    with sr.AudioFile(filename) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return ""


def speak(text):

    tts = gTTS(text)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(tmp.name)

    audio_file = open(tmp.name, "rb")
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3")


# ----------------------------
# Sidebar Navigation
# ----------------------------

st.sidebar.title("📘 Learning Panel")

page = st.sidebar.radio(
    "Navigate",
    ["Home Dashboard","AI Tutor","Weekly Test"]
)

results = load_results()

# ----------------------------
# HOME DASHBOARD
# ----------------------------

if page == "Home Dashboard":

    st.title("🏠 Student Dashboard")

    if not results.empty:
        total_points = (results["Marks"] >= 0.9 * results["Total"]).sum() * 10
    else:
        total_points = 0

    col1, col2 = st.columns(2)

    col1.metric("⭐ Points Earned", total_points)
    col2.metric("📝 Tests Taken", len(results))

    st.markdown("---")

    st.subheader("📅 Test History")

    if results.empty:
        st.info("No tests taken yet.")
    else:
        st.dataframe(results.sort_values("Date", ascending=False))


# ----------------------------
# AI TUTOR
# ----------------------------

elif page == "AI Tutor":

    st.title("📚 AI Tutor")

    st.write("Ask anything and continue the conversation with your tutor.")

    question = st.text_input("Ask your tutor")

    # ----------------------
    # Voice Input
    # ----------------------

    st.subheader("🎤 Speak Instead")

    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True
    )

    if audio:

        voice_text = speech_to_text(audio["bytes"])

        st.write("You said:", voice_text)

        question = voice_text

    # ----------------------
    # Ask Tutor
    # ----------------------

    if st.button("Ask Tutor"):

        st.session_state.chat_history.append({"role":"user","content":question})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"You are a friendly tutor for an 11 year old student."}
            ] + st.session_state.chat_history
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append({"role":"assistant","content":answer})

    # ----------------------
    # Display Chat
    # ----------------------

    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            st.write("🧑 **Student:**", msg["content"])

        else:
            st.write("👩‍🏫 **Tutor:**", msg["content"])
            speak(msg["content"])


# ----------------------------
# WEEKLY TEST
# ----------------------------

elif page == "Weekly Test":

    st.title("📝 Weekly Test")

    if not results.empty:

        last_test = pd.to_datetime(results["Date"]).max()

        next_allowed = last_test + timedelta(days=7)

        if datetime.today() < next_allowed:

            st.warning(f"Next test available on {next_allowed.date()}")
            st.stop()

    topic = st.text_input("Enter Test Topic")

    if st.button("Generate Test"):

        prompt = f"""
Create a 20 mark class 5 test on {topic}.

10 questions.
Mix of MCQ, fill in blanks and word problems.

Return ONLY JSON:

{{
 "questions":[
  {{
   "type":"mcq",
   "question":"",
   "options":["A","B","C","D"],
   "answer":""
  }},
  {{
   "type":"fill_blank",
   "question":"",
   "answer":""
  }},
  {{
   "type":"word_problem",
   "question":"",
   "answer":""
  }}
 ]
}}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )

        data = response.choices[0].message.content

        try:
            test = json.loads(data)
            st.session_state.test = test["questions"]
        except:
            st.error("Test generation failed.")
            st.stop()

    if "test" in st.session_state:

        answers = {}

        for i,q in enumerate(st.session_state.test):

            st.write(f"Q{i+1}. {q['question']}")

            if q["type"] == "mcq":
                answers[i] = st.radio(
                    "Choose",
                    q["options"],
                    key=f"q{i}"
                )
            else:
                answers[i] = st.text_input(
                    "Answer",
                    key=f"q{i}"
                )

        if st.button("Submit Test"):

            score = 0
            wrong = []

            for i,q in enumerate(st.session_state.test):

                student = answers[i]
                correct = q["answer"]

                if str(student).lower().strip() == str(correct).lower().strip():
                    score += 2
                else:
                    wrong.append((q["question"], student, correct))

            save_result(topic, score, 20)

            st.success(f"Final Score: {score}/20")

            if wrong:

                st.subheader("Review Mistakes")

                for q,s,c in wrong:
                    st.write("Question:", q)
                    st.write("Your answer:", s)
                    st.write("Correct answer:", c)
                    st.write("---")

            else:
                st.success("Perfect Score! 🎉")

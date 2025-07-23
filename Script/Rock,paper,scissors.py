import streamlit as st
import random
import pandas as pd
import numpy as np

# ---  Custom CSS Styling for UI Components ---
st.markdown("""
<style>
/* Main title banner */
.main-title {
    background-color: #BE5014;
    color: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

/* Subtitle under the main title */
.subtitle {
    background-color: #F7CEB7;
    color: #374151;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 30px;
}

/* General buttons (e.g., move, reset) */
.stButton > button {
    background-color: #BE5014;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 16px;
    transition: 0.3s;
}
.stButton > button:hover {
    background-color: #a9440f;
}
.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 2px #be50144d;
}

/* Section headers like STATS or GAME HISTORY */
.move-title {
    background-color: white;
    color: #BE5014;
    font-size: 20px;
    text-align: center;
    margin-bottom: 30px;
    font-weight: bold;
    border: 2px solid #BE5014;
    border-radius: 10px;
    max-width: 100%;
    width: 100%;
    margin: 0 auto;
}

/* Centered layout helper */
.centered {
    display: flex;
    justify-content: center;
}

/* Stat cards for displaying metrics */
.stat-card {
    background-color: #F7CEB7;
    padding: 10px;
    border-radius: 15px;
    text-align: center;
    font-weight: bold;
    font-size: 14px;
    color: #374151;
    border: 2px solid #BE5014;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
.stat-value {
    font-size: 15px;
    color: #BE5014;
    margin-top: 10px;
}

/* Table styling */
.styled-table {
    margin-left: auto;
    margin-right: auto;
    width: 80%;
    border-collapse: collapse;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    font-family: sans-serif;
    font-size: 14px;
    text-align: center;
}
.styled-table thead tr {
    background-color: #BE5014;
    color: white;
    font-weight: bold;
    text-align: center;
}
.styled-table th, .styled-table td {
    padding: 12px 15px;
    border-bottom: 1px solid #ddd;
    text-align: center;
}
.styled-table tbody tr:nth-child(even) {
    background-color: #F7CEB7;
}
.styled-table tbody tr:nth-child(odd) {
    background-color: #fff4ec;
}

/* Input styling */
input[type="text"], textarea {
    border: 2px solid #BE5014;
    border-radius: 8px;
    padding: 10px;
    font-size: 16px;
}

/* Sidebar customization */
section[data-testid="stSidebar"] {
    background-color: #f5e1d6;
    color: #374151;
}

/* Alert box style */
.stAlert {
    border-radius: 10px;
    padding: 15px;
    font-weight: bold;
    font-size: 16px;
}

/* Custom light-orange alert box */
.custom-alert {
    background-color: #FFF6E5;
    color: #BE5014;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 15px;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
}
    <style>
/* Aumentar el ancho del sidebar */
    [data-testid="stSidebar"] {
        width: 400px !important;
        max-width: 400px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---  Game Title ---
st.markdown('<div class="main-title">✊ ROCK, PAPER, SCISSORS </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> PREDICTIVE AI GAME USING MARKOV CHAINS </div>', unsafe_allow_html=True)
st.markdown('<div class="centered"><div class="move-title">CHOOSE YOUR MOVE</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---  AI Class with Markov Chains of Orders 1, 2, 3 ---
class RockPaperScissorsAI:
    def __init__(self):
        self.history = []
        self.transitions = [{} for _ in range(3)]  # One dictionary per order

    def update_history(self, user_choice):
        for order in range(3):
            seq = self._get_sequence(order + 1)
            if seq:
                if seq not in self.transitions[order]:
                    self.transitions[order][seq] = {"rock": 0, "paper": 0, "scissors": 0}
                self.transitions[order][seq][user_choice] += 1
        self.history.append(user_choice)

    def _get_sequence(self, order):
        if len(self.history) < order:
            return None
        return "-".join(self.history[-order:])

    def predict_next(self):
        for order in reversed(range(3)):
            seq = self._get_sequence(order + 1)
            if seq and seq in self.transitions[order]:
                prediction = self.transitions[order][seq]
                return max(prediction, key=prediction.get)
        return random.choice(["rock", "paper", "scissors"])

    def get_ai_choice(self):
        predicted_user_move = self.predict_next()
        st.session_state.last_prediction = predicted_user_move
        counter = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        return counter[predicted_user_move]

# ---  User Profile Classification ---
def classify_user(history, window=10):
    if len(history) < window:
        return "Undetermined"
    recent = history[-window:]
    changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
    if changes >= window * 0.8:
        return "Impulsive"
    elif changes <= window * 0.3:
        return "Repetitive"
    else:
        return "Strategic"

# ---  Session Initialization ---
if 'ai' not in st.session_state:
    st.session_state.ai = RockPaperScissorsAI()
    st.session_state.results = []
    st.session_state.score = {"User": 0, "AI": 0, "Ties": 0}
    st.session_state.last_prediction = None
    st.session_state.hits = []

# ---  Sidebar Instructions + Reset Button ---
with st.sidebar:
    st.markdown("""
    <div style='
        background-color: #FDEAE2;
        border-left: 6px solid #BE5014;
        padding: 15px 20px;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.6;
        color: #374151;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 700px;
        margin: 0 auto;
    '>
        <h4 style='margin-top: 0; display: flex; align-items: center; color: #A1440E;'>
            📋&nbsp;Instructions
        </h4>
        <ul style='padding-left: 20px; margin: 0;'>
            <li>Choose your move.</li>
            <li>The AI will try to predict your next move.</li>
            <li>Your play style will be analyzed.</li>
            <li>You'll see a chart of the AI's prediction accuracy.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔁 Reset game"):
        st.session_state.ai = RockPaperScissorsAI()
        st.session_state.results = []
        st.session_state.score = {"User": 0, "AI": 0, "Ties": 0}
        st.session_state.last_prediction = None
        st.session_state.hits = []
        st.rerun()

# ---  Move Selection Buttons (with images) ---
col_empty1, col1, col2, col3 = st.columns([0.6, 1.5, 1.5, 1.5])
user_move = None

with col1:
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("ROCK"):
            user_move = "rock"
        st.image("Script/rock.png", width=100)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("PAPER"):
            user_move = "paper"
        st.image("Script/paper.png", width=100)
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("SCISSORS"):
            user_move = "scissors"
        st.image("Script/scissors.png", width=100)
        st.markdown("</div>", unsafe_allow_html=True)

# ---  Game Logic ---
def determine_winner(user, ai):
    if user == ai:
        return "Tie"
    if (user == "rock" and ai == "scissors") or \
       (user == "scissors" and ai == "paper") or \
       (user == "paper" and ai == "rock"):
        return "User"
    return "AI"

if user_move:
    ai_move = st.session_state.ai.get_ai_choice()
    winner = determine_winner(user_move, ai_move)
    st.session_state.ai.update_history(user_move)

    st.markdown(f"""
    <div class="custom-alert">
    🎮 AI chose: <strong>{ai_move.upper()}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="custom-alert">
    🏆 Who won?: <strong>{winner}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.results.append((user_move, ai_move, winner))
    if winner == "User":
        st.session_state.score["User"] += 1
    elif winner == "AI":
        st.session_state.score["AI"] += 1
    else:
        st.session_state.score["Ties"] += 1

    if st.session_state.last_prediction:
        hit = user_move == st.session_state.last_prediction
        st.session_state.hits.append(1 if hit else 0)

# ---  Stats and Visualizations ---
if st.session_state.results:
    profile = classify_user(st.session_state.ai.history)
    total_games = len(st.session_state.results)
    ai_wins = st.session_state.score["AI"]
    winrate_ai = ai_wins / total_games if total_games > 0 else 0
    accuracy = np.mean(st.session_state.hits) if st.session_state.hits else 0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="centered"><div class="move-title">STATS</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            🎯 Prediction Accuracy
            <div class="stat-value">{accuracy:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            🧠 Player Profile
            <div class="stat-value">{profile}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="centered"><div class="move-title">GAME HISTORY</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.results, columns=["User", "AI", "Winner"])
    st.markdown(
        df.to_html(index=False, classes="styled-table", escape=False, border=0),
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="centered"><div class="move-title">CUMULATIVE PREDICTION ACCURACY</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if len(st.session_state.hits) > 1:
        import altair as alt
        df_acc = pd.DataFrame({
            "x": range(len(st.session_state.hits)),
            "y": pd.Series(st.session_state.hits).expanding().mean()
        })

        st.altair_chart(
            alt.Chart(df_acc).mark_line(color="#BE5014").encode(x="x", y="y"),
            use_container_width=True
        )

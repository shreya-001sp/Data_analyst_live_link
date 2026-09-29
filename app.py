import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Learning System",
    page_icon="🎓",
    layout="wide"
)

DB_NAME = "learning.db"


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def create_tables():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            level TEXT DEFAULT 'Beginner',
            interests TEXT DEFAULT ''
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            description TEXT NOT NULL,
            keywords TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER,
            question TEXT NOT NULL,
            option_a TEXT NOT NULL,
            option_b TEXT NOT NULL,
            option_c TEXT NOT NULL,
            option_d TEXT NOT NULL,
            answer TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            FOREIGN KEY(topic_id) REFERENCES topics(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            topic_id INTEGER,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            difficulty TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(topic_id) REFERENCES topics(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            topic_id INTEGER,
            activity TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(topic_id) REFERENCES topics(id)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# INSERT INITIAL DATA
# ============================================================

def insert_initial_data():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM topics")
    topic_count = cursor.fetchone()[0]

    if topic_count == 0:

        topics = [

            (
                "Python Basics",
                "Programming",
                "Beginner",
                "Introduction to Python programming, variables, data types and basic syntax.",
                "python programming variables syntax beginner"
            ),

            (
                "Object Oriented Programming",
                "Programming",
                "Intermediate",
                "Classes, objects, inheritance, polymorphism and encapsulation.",
                "python oop classes objects inheritance polymorphism"
            ),

            (
                "Data Structures",
                "Computer Science",
                "Intermediate",
                "Arrays, linked lists, stacks, queues, trees and graphs.",
                "data structures arrays linked lists stacks queues trees graphs"
            ),

            (
                "SQL",
                "Database",
                "Beginner",
                "Learn databases, SQL queries, SELECT, WHERE, JOIN and aggregation.",
                "sql database select where join group by"
            ),

            (
                "Machine Learning",
                "Artificial Intelligence",
                "Intermediate",
                "Introduction to machine learning, supervised and unsupervised learning.",
                "machine learning ai supervised unsupervised classification regression"
            ),

            (
                "Deep Learning",
                "Artificial Intelligence",
                "Advanced",
                "Neural networks, CNNs, RNNs and deep learning concepts.",
                "deep learning neural networks cnn rnn tensorflow"
            ),

            (
                "Computer Networks",
                "Networking",
                "Intermediate",
                "OSI model, TCP/IP, IP addressing, routing and transport protocols.",
                "computer networks osi tcp ip routing networking"
            ),

            (
                "Cyber Security",
                "Security",
                "Intermediate",
                "Basic concepts of cybersecurity, threats, encryption and authentication.",
                "cyber security encryption authentication threats"
            ),

            (
                "Web Development",
                "Web",
                "Beginner",
                "HTML, CSS, JavaScript and basic web development.",
                "html css javascript web development frontend"
            ),

            (
                "Statistics",
                "Data Science",
                "Beginner",
                "Mean, median, probability, variance and basic statistics.",
                "statistics mean median probability variance data"
            )
        ]

        cursor.executemany("""
            INSERT INTO topics
            (name, category, difficulty, description, keywords)
            VALUES (?, ?, ?, ?, ?)
        """, topics)

    conn.commit()

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    cursor.execute("SELECT COUNT(*) FROM questions")
    question_count = cursor.fetchone()[0]

    if question_count == 0:

        cursor.execute("SELECT id, name FROM topics")
        topic_rows = cursor.fetchall()

        topic_ids = {name: topic_id for topic_id, name in topic_rows}

        questions = [

            # Python
            (
                topic_ids["Python Basics"],
                "Which keyword is used to define a function in Python?",
                "function",
                "def",
                "func",
                "define",
                "B",
                "Beginner"
            ),

            (
                topic_ids["Python Basics"],
                "Which of the following is a Python list?",
                "(1,2,3)",
                "[1,2,3]",
                "{1,2,3}",
                "<1,2,3>",
                "B",
                "Beginner"
            ),

            (
                topic_ids["Python Basics"],
                "Which symbol is used for comments in Python?",
                "//",
                "/*",
                "#",
                "--",
                "C",
                "Beginner"
            ),

            (
                topic_ids["Python Basics"],
                "What is the output of len([1,2,3])?",
                "2",
                "3",
                "4",
                "1",
                "B",
                "Beginner"
            ),

            # OOP
            (
                topic_ids["Object Oriented Programming"],
                "Which concept allows a class to acquire properties of another class?",
                "Encapsulation",
                "Inheritance",
                "Abstraction",
                "Compilation",
                "B",
                "Intermediate"
            ),

            (
                topic_ids["Object Oriented Programming"],
                "Which concept hides implementation details?",
                "Inheritance",
                "Polymorphism",
                "Abstraction",
                "Looping",
                "C",
                "Intermediate"
            ),

            (
                topic_ids["Object Oriented Programming"],
                "What is an object?",
                "A database",
                "An instance of a class",
                "A function",
                "A loop",
                "B",
                "Intermediate"
            ),

            # Data Structures
            (
                topic_ids["Data Structures"],
                "Which data structure follows LIFO?",
                "Queue",
                "Stack",
                "Array",
                "Graph",
                "B",
                "Intermediate"
            ),

            (
                topic_ids["Data Structures"],
                "Which data structure follows FIFO?",
                "Stack",
                "Queue",
                "Tree",
                "Graph",
                "B",
                "Beginner"
            ),

            (
                topic_ids["Data Structures"],
                "Which structure consists of nodes connected by edges?",
                "Array",
                "Graph",
                "Stack",
                "Variable",
                "B",
                "Intermediate"
            ),

            # SQL
            (
                topic_ids["SQL"],
                "Which command is used to retrieve data?",
                "GET",
                "SELECT",
                "FETCH",
                "OPEN",
                "B",
                "Beginner"
            ),

            (
                topic_ids["SQL"],
                "Which clause filters rows?",
                "ORDER BY",
                "GROUP BY",
                "WHERE",
                "SELECT",
                "C",
                "Beginner"
            ),

            (
                topic_ids["SQL"],
                "Which keyword combines rows from two tables?",
                "JOIN",
                "MERGE",
                "CONNECT",
                "LINK",
                "A",
                "Intermediate"
            ),

            (
                topic_ids["SQL"],
                "Which command removes a table?",
                "REMOVE",
                "DELETE TABLE",
                "DROP",
                "CLEAR",
                "C",
                "Intermediate"
            ),

            # Machine Learning
            (
                topic_ids["Machine Learning"],
                "Which is an example of supervised learning?",
                "Clustering",
                "Classification",
                "Dimensionality reduction",
                "Association",
                "B",
                "Beginner"
            ),

            (
                topic_ids["Machine Learning"],
                "Which algorithm is commonly used for clustering?",
                "Linear Regression",
                "K-Means",
                "Decision Tree",
                "Naive Bayes",
                "B",
                "Intermediate"
            ),

            (
                topic_ids["Machine Learning"],
                "What does ML stand for?",
                "Machine Learning",
                "Multiple Logic",
                "Machine Language",
                "Model Learning",
                "A",
                "Beginner"
            ),

            (
                topic_ids["Machine Learning"],
                "Which algorithm can be used for predicting continuous values?",
                "Linear Regression",
                "K-Means",
                "Apriori",
                "PCA",
                "A",
                "Intermediate"
            ),

            # Deep Learning
            (
                topic_ids["Deep Learning"],
                "What is the basic computational unit of a neural network?",
                "Neuron",
                "Router",
                "Database",
                "Compiler",
                "A",
                "Beginner"
            ),

            (
                topic_ids["Deep Learning"],
                "CNN is commonly used for?",
                "Images",
                "Databases",
                "Networking",
                "SQL",
                "A",
                "Intermediate"
            ),

            # Networking
            (
                topic_ids["Computer Networks"],
                "How many layers are present in the OSI model?",
                "5",
                "6",
                "7",
                "8",
                "C",
                "Beginner"
            ),

            (
                topic_ids["Computer Networks"],
                "Which protocol is connection-oriented?",
                "UDP",
                "TCP",
                "IP",
                "ARP",
                "B",
                "Intermediate"
            ),

            (
                topic_ids["Computer Networks"],
                "What does IP stand for?",
                "Internet Protocol",
                "Internet Process",
                "Internal Protocol",
                "Internet Program",
                "A",
                "Beginner"
            ),

            # Cyber Security
            (
                topic_ids["Cyber Security"],
                "What is encryption used for?",
                "Data protection",
                "Increasing RAM",
                "Compiling programs",
                "Creating databases",
                "A",
                "Beginner"
            ),

            (
                topic_ids["Cyber Security"],
                "Which is a common authentication factor?",
                "Password",
                "Compiler",
                "Router",
                "Database",
                "A",
                "Beginner"
            ),

            # Web
            (
                topic_ids["Web Development"],
                "HTML is primarily used for?",
                "Structure of web pages",
                "Database management",
                "Machine learning",
                "Networking",
                "A",
                "Beginner"
            ),

            (
                topic_ids["Web Development"],
                "CSS is primarily used for?",
                "Styling",
                "Database queries",
                "Server security",
                "Machine learning",
                "A",
                "Beginner"
            ),

            # Statistics
            (
                topic_ids["Statistics"],
                "What is the average of 2, 4 and 6?",
                "2",
                "3",
                "4",
                "6",
                "C",
                "Beginner"
            ),

            (
                topic_ids["Statistics"],
                "Which measure represents the middle value?",
                "Mean",
                "Median",
                "Variance",
                "Range",
                "B",
                "Beginner"
            )
        ]

        cursor.executemany("""
            INSERT INTO questions
            (
                topic_id,
                question,
                option_a,
                option_b,
                option_c,
                option_d,
                answer,
                difficulty
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, questions)

    conn.commit()
    conn.close()


# ============================================================
# USER FUNCTIONS
# ============================================================

def register_user(name, username, password, level, interests):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO students
            (name, username, password, level, interests)
            VALUES (?, ?, ?, ?, ?)
        """, (
            name,
            username,
            hash_password(password),
            level,
            interests
        ))

        conn.commit()

        return True, "Registration successful."

    except sqlite3.IntegrityError:

        return False, "Username already exists."

    finally:

        conn.close()


def login_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, username, level, interests
        FROM students
        WHERE username = ? AND password = ?
    """, (
        username,
        hash_password(password)
    ))

    user = cursor.fetchone()

    conn.close()

    return user


def get_student(student_id):

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM students
        WHERE id = ?
    """, conn, params=(student_id,))

    conn.close()

    if df.empty:
        return None

    return df.iloc[0]


# ============================================================
# TOPIC FUNCTIONS
# ============================================================

def get_topics():

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM topics
    """, conn)

    conn.close()

    return df


def get_topic_by_name(name):

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM topics
        WHERE name = ?
    """, conn, params=(name,))

    conn.close()

    if df.empty:
        return None

    return df.iloc[0]


# ============================================================
# QUIZ FUNCTIONS
# ============================================================

def get_questions(topic_id, difficulty=None):

    conn = get_connection()

    if difficulty:

        df = pd.read_sql_query("""
            SELECT *
            FROM questions
            WHERE topic_id = ?
            AND difficulty = ?
            ORDER BY RANDOM()
            LIMIT 5
        """, conn, params=(topic_id, difficulty))

    else:

        df = pd.read_sql_query("""
            SELECT *
            FROM questions
            WHERE topic_id = ?
            ORDER BY RANDOM()
            LIMIT 5
        """, conn, params=(topic_id,))

    conn.close()

    return df


def save_quiz_result(
    student_id,
    topic_id,
    score,
    total,
    percentage,
    difficulty
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_results
        (
            student_id,
            topic_id,
            score,
            total,
            percentage,
            difficulty
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        topic_id,
        score,
        total,
        percentage,
        difficulty
    ))

    cursor.execute("""
        INSERT INTO learning_history
        (
            student_id,
            topic_id,
            activity
        )
        VALUES (?, ?, ?)
    """, (
        student_id,
        topic_id,
        "Completed Quiz"
    ))

    conn.commit()
    conn.close()


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

def get_student_performance(student_id):

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT
            t.name AS topic,
            t.category,
            COUNT(qr.id) AS attempts,
            AVG(qr.percentage) AS average_score,
            MAX(qr.percentage) AS best_score
        FROM quiz_results qr
        JOIN topics t
            ON qr.topic_id = t.id
        WHERE qr.student_id = ?
        GROUP BY qr.topic_id
        ORDER BY average_score ASC
    """, conn, params=(student_id,))

    conn.close()

    return df


def get_weak_topics(student_id):

    performance = get_student_performance(student_id)

    if performance.empty:
        return []

    weak = performance[
        performance["average_score"] < 60
    ]

    return weak["topic"].tolist()


def get_strong_topics(student_id):

    performance = get_student_performance(student_id)

    if performance.empty:
        return []

    strong = performance[
        performance["average_score"] >= 80
    ]

    return strong["topic"].tolist()


# ============================================================
# PERSONALIZED RECOMMENDATION ENGINE
# ============================================================

def recommend_topics(student_id):

    topics = get_topics()

    student = get_student(student_id)

    if student is None:
        return topics

    performance = get_student_performance(student_id)

    weak_topics = get_weak_topics(student_id)

    studied_topics = []

    if not performance.empty:
        studied_topics = performance["topic"].tolist()

    # --------------------------------------------------------
    # Create student profile
    # --------------------------------------------------------

    profile_parts = []

    profile_parts.append(str(student["interests"]))

    profile_parts.append(str(student["level"]))

    for topic in weak_topics:
        profile_parts.append(topic)

    for topic in studied_topics:
        profile_parts.append(topic)

    student_profile = " ".join(profile_parts)

    # --------------------------------------------------------
    # Create topic documents
    # --------------------------------------------------------

    documents = []

    for _, row in topics.iterrows():

        document = (
            str(row["name"])
            + " "
            + str(row["category"])
            + " "
            + str(row["description"])
            + " "
            + str(row["keywords"])
        )

        documents.append(document)

    documents.append(student_profile)

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    topics = topics.copy()

    topics["similarity"] = similarity

    # --------------------------------------------------------
    # Give extra priority to weak topics
    # --------------------------------------------------------

    topics["recommendation_score"] = topics["similarity"]

    for i, row in topics.iterrows():

        if row["name"] in weak_topics:

            topics.loc[
                i,
                "recommendation_score"
            ] += 0.50

    # Avoid recommending already mastered topics first

    for i, row in topics.iterrows():

        if row["name"] in get_strong_topics(student_id):

            topics.loc[
                i,
                "recommendation_score"
            ] -= 0.20

    topics = topics.sort_values(
        "recommendation_score",
        ascending=False
    )

    return topics


# ============================================================
# DIFFICULTY ENGINE
# ============================================================

def determine_difficulty(student_id, topic_id):

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT AVG(percentage) AS avg_score
        FROM quiz_results
        WHERE student_id = ?
        AND topic_id = ?
    """, conn, params=(student_id, topic_id))

    conn.close()

    if df.empty or pd.isna(df.iloc[0]["avg_score"]):

        return "Beginner"

    score = df.iloc[0]["avg_score"]

    if score >= 80:
        return "Advanced"

    elif score >= 60:
        return "Intermediate"

    else:
        return "Beginner"


# ============================================================
# STUDY HISTORY
# ============================================================

def get_history(student_id):

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT
            lh.activity,
            t.name AS topic,
            lh.date
        FROM learning_history lh
        JOIN topics t
            ON lh.topic_id = t.id
        WHERE lh.student_id = ?
        ORDER BY lh.date DESC
    """, conn, params=(student_id,))

    conn.close()

    return df


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_dashboard_stats(student_id):

    conn = get_connection()

    result = {}

    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM quiz_results
        WHERE student_id = ?
    """, (student_id,))

    result["quizzes"] = cursor.fetchone()[0]

    cursor.execute("""
        SELECT AVG(percentage)
        FROM quiz_results
        WHERE student_id = ?
    """, (student_id,))

    result["average"] = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT topic_id)
        FROM quiz_results
        WHERE student_id = ?
    """, (student_id,))

    result["topics"] = cursor.fetchone()[0]

    cursor.execute("""
        SELECT MAX(percentage)
        FROM quiz_results
        WHERE student_id = ?
    """, (student_id,))

    result["best"] = cursor.fetchone()[0]

    conn.close()

    return result


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "student_id" not in st.session_state:
        st.session_state.student_id = None

    if "student_name" not in st.session_state:
        st.session_state.student_name = None

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None

    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = None

    if "quiz_difficulty" not in st.session_state:
        st.session_state.quiz_difficulty = None


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title("🎓 Personalized Learning Recommendation System")

    st.subheader("Login")

    username = st.text_input(
        "Username", key ="login_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        key =" login_password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        user = login_user(
            username,
            password
        )

        if user:

            st.session_state.logged_in = True
            st.session_state.student_id = user[0]
            st.session_state.student_name = user[1]

            st.session_state.page = "Dashboard"

            st.success(
                "Login successful!"
            )

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )


# ============================================================
# REGISTER PAGE
# ============================================================

def register_page():

    st.title("📝 Create Student Account")

    name = st.text_input(
        "Full Name",
        key ="register_name"
    )

    username = st.text_input(
        "Username", key ="register_username"
    )

    password = st.text_input(
        "Password",
        type="password", key ="register_password"
    )

    level = st.selectbox(
        "Current Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    interests = st.multiselect(
        "Your Interests",
        [
            "Python",
            "Programming",
            "Data Structures",
            "SQL",
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Networking",
            "Cyber Security",
            "Web Development",
            "Data Science"
        ]
    )

    if st.button(
        "Create Account",
        use_container_width=True
    ):

        if not name or not username or not password:

            st.warning(
                "Please fill all required fields."
            )

            return

        interest_text = " ".join(
            interests
        )

        success, message = register_user(
            name,
            username,
            password,
            level,
            interest_text
        )

        if success:

            st.success(message)

            st.info(
                "You can now login using your username and password."
            )

        else:

            st.error(message)


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    with st.sidebar:

        st.title("🎓 LearnAI")

        st.write(
            f"Welcome, **{st.session_state.student_name}**"
        )

        st.divider()

        if st.button(
            "🏠 Dashboard",
            use_container_width=True
        ):
            st.session_state.page = "Dashboard"

        if st.button(
            "🧠 Recommendations",
            use_container_width=True
        ):
            st.session_state.page = "Recommendations"

        if st.button(
            "📝 Take Quiz",
            use_container_width=True
        ):
            st.session_state.page = "Quiz"

        if st.button(
            "📊 Performance",
            use_container_width=True
        ):
            st.session_state.page = "Performance"

        if st.button(
            "📚 Topics",
            use_container_width=True
        ):
            st.session_state.page = "Topics"

        if st.button(
            "📜 Learning History",
            use_container_width=True
        ):
            st.session_state.page = "History"

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.student_id = None
            st.session_state.student_name = None

            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():

    st.title("🏠 Student Dashboard")

    student_id = st.session_state.student_id

    stats = get_dashboard_stats(
        student_id
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Quizzes Completed",
        stats["quizzes"]
    )

    average = stats["average"]

    if average is None:
        average_display = "0%"
    else:
        average_display = f"{average:.1f}%"

    col2.metric(
        "Average Score",
        average_display
    )

    col3.metric(
        "Topics Studied",
        stats["topics"]
    )

    best = stats["best"]

    if best is None:
        best_display = "0%"
    else:
        best_display = f"{best:.1f}%"

    col4.metric(
        "Best Score",
        best_display
    )

    st.divider()

    st.subheader("🎯 Personalized Recommendations")

    recommendations = recommend_topics(
        student_id
    )

    recommendations = recommendations.head(5)

    for _, row in recommendations.iterrows():

        with st.container():

            col1, col2 = st.columns([4, 1])

            with col1:

                st.markdown(
                    f"### 📘 {row['name']}"
                )

                st.write(
                    row["description"]
                )

                st.caption(
                    f"Category: {row['category']} | "
                    f"Difficulty: {row['difficulty']}"
                )

            with col2:

                st.metric(
                    "AI Score",
                    f"{row['recommendation_score']:.2f}"
                )

            st.divider()

    weak = get_weak_topics(student_id)

    if weak:

        st.warning(
            "⚠️ Topics requiring more practice: "
            + ", ".join(weak)
        )

    else:

        st.info(
            "Take a few quizzes to allow the AI "
            "to identify your weak topics."
        )


# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

def recommendations_page():

    st.title("🧠 AI Personalized Recommendations")

    st.write(
        "The recommendation engine analyzes your interests, "
        "learning history and quiz performance."
    )

    recommendations = recommend_topics(
        st.session_state.student_id
    )

    for index, row in recommendations.head(8).iterrows():

        st.markdown(
            f"## {index + 1}. {row['name']}"
        )

        st.write(
            row["description"]
        )

        col1, col2, col3 = st.columns(3)

        col1.write(
            f"**Category:** {row['category']}"
        )

        col2.write(
            f"**Difficulty:** {row['difficulty']}"
        )

        col3.write(
            f"**AI Recommendation Score:** "
            f"{row['recommendation_score']:.2f}"
        )

        st.divider()


# ============================================================
# QUIZ PAGE
# ============================================================

def quiz_page():

    st.title("📝 Personalized Quiz")

    topics = get_topics()

    topic_name = st.selectbox(
        "Choose a topic",
        topics["name"].tolist()
    )

    topic = get_topic_by_name(
        topic_name
    )

    recommended_difficulty = determine_difficulty(
        st.session_state.student_id,
        int(topic["id"])
    )

    st.info(
        f"🤖 Based on your previous performance, "
        f"recommended difficulty: **{recommended_difficulty}**"
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        index=[
            "Beginner",
            "Intermediate",
            "Advanced"
        ].index(recommended_difficulty)
    )

    if st.button(
        "Start Quiz",
        use_container_width=True
    ):

        questions = get_questions(
            int(topic["id"]),
            difficulty
        )

        if questions.empty:

            questions = get_questions(
                int(topic["id"])
            )

        st.session_state.quiz_questions = questions
        st.session_state.quiz_topic = topic
        st.session_state.quiz_difficulty = difficulty

        st.rerun()

    questions = st.session_state.quiz_questions

    if questions is None:
        return

    if questions.empty:

        st.warning(
            "No questions are available for this topic."
        )

        return

    st.divider()

    st.subheader(
        f"Quiz: {st.session_state.quiz_topic['name']}"
    )

    answers = {}

    for index, row in questions.iterrows():

        st.markdown(
            f"### Q{index + 1}. {row['question']}"
        )

        answers[index] = st.radio(
            "Select your answer:",
            [
                f"A. {row['option_a']}",
                f"B. {row['option_b']}",
                f"C. {row['option_c']}",
                f"D. {row['option_d']}"
            ],
            key=f"question_{index}"
        )

    if st.button(
        "Submit Quiz",
        use_container_width=True
    ):

        score = 0

        for index, row in questions.iterrows():

            selected = answers.get(index)

            if selected:

                selected_letter = selected[0]

                if selected_letter == row["answer"]:

                    score += 1

        total = len(questions)

        percentage = (
            score / total
        ) * 100

        save_quiz_result(
            st.session_state.student_id,
            int(st.session_state.quiz_topic["id"]),
            score,
            total,
            percentage,
            st.session_state.quiz_difficulty
        )

        st.session_state.quiz_questions = None

        st.success(
            f"🎉 Quiz completed! "
            f"You scored **{score}/{total} "
            f"({percentage:.1f}%)**."
        )

        if percentage >= 80:

            st.balloons()

            st.success(
                "Excellent! You have a strong understanding "
                "of this topic."
            )

        elif percentage >= 60:

            st.info(
                "Good job! A little more practice will "
                "strengthen your understanding."
            )

        else:

            st.warning(
                "You should revise this topic and try "
                "another practice quiz."
            )


# ============================================================
# PERFORMANCE PAGE
# ============================================================

def performance_page():

    st.title("📊 Performance Analysis")

    performance = get_student_performance(
        st.session_state.student_id
    )

    if performance.empty:

        st.info(
            "No performance data available yet. "
            "Take a quiz first."
        )

        return

    st.subheader("Topic-wise Performance")

    display_df = performance.copy()

    display_df["average_score"] = (
        display_df["average_score"]
        .round(2)
    )

    display_df["best_score"] = (
        display_df["best_score"]
        .round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "📈 Average Score by Topic"
    )

    chart_df = performance[
        ["topic", "average_score"]
    ].set_index("topic")

    st.bar_chart(
        chart_df
    )

    st.divider()

    weak = get_weak_topics(
        st.session_state.student_id
    )

    strong = get_strong_topics(
        st.session_state.student_id
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⚠️ Weak Topics")

        if weak:

            for topic in weak:
                st.write(
                    f"🔴 {topic}"
                )

        else:

            st.success(
                "No weak topics detected."
            )

    with col2:

        st.subheader("🌟 Strong Topics")

        if strong:

            for topic in strong:
                st.write(
                    f"🟢 {topic}"
                )

        else:

            st.info(
                "Take more quizzes to identify strong topics."
            )


# ============================================================
# TOPICS PAGE
# ============================================================

def topics_page():

    st.title("📚 Available Learning Topics")

    topics = get_topics()

    category = st.selectbox(
        "Filter by category",
        ["All"] + sorted(
            topics["category"].unique().tolist()
        )
    )

    if category != "All":

        topics = topics[
            topics["category"] == category
        ]

    for _, row in topics.iterrows():

        with st.expander(
            f"📘 {row['name']}"
        ):

            st.write(
                row["description"]
            )

            st.write(
                f"**Category:** {row['category']}"
            )

            st.write(
                f"**Difficulty:** {row['difficulty']}"
            )

            st.write(
                f"**Keywords:** {row['keywords']}"
            )


# ============================================================
# HISTORY PAGE
# ============================================================

def history_page():

    st.title("📜 Learning History")

    history = get_history(
        st.session_state.student_id
    )

    if history.empty:

        st.info(
            "No learning activity recorded yet."
        )

        return

    st.dataframe(
        history,
        use_container_width=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    create_tables()

    insert_initial_data()

    initialize_session()

    # --------------------------------------------------------
    # NOT LOGGED IN
    # --------------------------------------------------------

    if not st.session_state.logged_in:

        tab1, tab2 = st.tabs(
            [
                "🔐 Login",
                "📝 Register"
            ]
        )

        with tab1:
            login_page()

        with tab2:
            register_page()

        return

    # --------------------------------------------------------
    # LOGGED IN
    # --------------------------------------------------------

    sidebar()

    page = st.session_state.page

    if page == "Dashboard":

        dashboard_page()

    elif page == "Recommendations":

        recommendations_page()

    elif page == "Quiz":

        quiz_page()

    elif page == "Performance":

        performance_page()

    elif page == "Topics":

        topics_page()

    elif page == "History":

        history_page()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
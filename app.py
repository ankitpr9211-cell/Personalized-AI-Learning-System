import os
import streamlit as st
import json
from datetime import datetime, timedelta

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(
    page_title="Ultimate AI Study Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PAGE CONFIG
# =========================

st.markdown("""
<style>

/* =============================
KEEP STREAMLIT LAYOUT NORMAL
============================= */

.block-container{
max-width:1200px;
padding-top:80px;
}

/* =============================
INSTAGRAM MOVING GRADIENT
============================= */

.stApp{

background: linear-gradient(
45deg,
#ff006e,
#ff7a00,
#ffbe0b,
#8338ec,
#3a86ff
);

background-size:400% 400%;
animation:gradientMove 16s ease infinite;

}

@keyframes gradientMove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}


/* =============================
GLASS MAIN PANELS
============================= */

[data-testid="stVerticalBlock"]{

background:rgba(255,255,255,0.08);
backdrop-filter:blur(16px);

border-radius:18px;

padding:15px;

border:1px solid rgba(255,255,255,.2);

}


/* =============================
GLASS SIDEBAR (TOOLS FIX)
============================= */

section[data-testid="stSidebar"]{

background:rgba(255,255,255,0.15);
backdrop-filter:blur(18px);

}

section[data-testid="stSidebar"] *{

color:white !important;

}


/* =============================
LIQUID COLOR BLOBS
============================= */

.blob{

position:fixed;

width:400px;
height:400px;

border-radius:50%;

filter:blur(120px);

opacity:.7;

z-index:-1;

animation:blobMove 20s infinite alternate;

}

.blob1{
background:#ff006e;
top:10%;
left:10%;
}

.blob2{
background:#8338ec;
bottom:20%;
right:10%;
}

.blob3{
background:#3a86ff;
top:40%;
right:30%;
}

@keyframes blobMove{

0%{transform:translate(0,0);}
100%{transform:translate(120px,-80px);}

}


/* =============================
CURSOR GLOW
============================= */

.cursor-light{

position:fixed;

width:280px;
height:280px;

pointer-events:none;

border-radius:50%;

background:radial-gradient(
circle,
rgba(255,255,255,.35),
transparent 70%
);

filter:blur(35px);

z-index:999;

}


/* =============================
CHATGPT THINKING LOADER
============================= */

.ai-loader{

display:flex;
gap:8px;
justify-content:center;
margin-top:20px;

}

.ai-loader span{

width:8px;
height:8px;

background:white;
border-radius:50%;

animation:think 1.4s infinite;

}

.ai-loader span:nth-child(2){animation-delay:.2s;}
.ai-loader span:nth-child(3){animation-delay:.4s;}

@keyframes think{

0%{opacity:.3;transform:translateY(0);}
50%{opacity:1;transform:translateY(-8px);}
100%{opacity:.3;transform:translateY(0);}

}

</style>

<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div class="blob blob3"></div>

<div class="cursor-light"></div>

<div class="ai-loader">
<span></span>
<span></span>
<span></span>
</div>


<script>

const light = document.querySelector(".cursor-light")

document.addEventListener("mousemove",e=>{

light.style.left = e.clientX-140 + "px"
light.style.top = e.clientY-140 + "px"

})

</script>

""", unsafe_allow_html=True)
# =========================
# GROQ API KEY
# =========================
GROQ_API_KEY = st.secrets["Your_API"]

# =========================
# CREATE FOLDERS
# =========================
if not os.path.exists("documents"):
    os.mkdir("documents")
if not os.path.exists("vector_db"):
    os.mkdir("vector_db")

# =========================
# HEADER
# =========================
st.markdown("# 🤖 Ultimate AI Study Assistant")
st.markdown("### Your Personal AI Tutor")
st.write("")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚡ Tools")
menu = st.sidebar.selectbox(
    "Choose Tool",
    ["Ask AI", "Chat with PDF", "Quiz Generator", "Study Planner"]
)
st.sidebar.markdown("---")
st.sidebar.info(
"""
Built with:

• Groq AI  
• LangChain  
• Streamlit  
"""
)

# =========================
# LOAD AI MODEL
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# =========================
# EMBEDDINGS MODEL
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =========================
# ASK AI
# =========================
if menu == "Ask AI":
    st.subheader("🤖 Ask AI Anything")
    question = st.text_input("Enter your question")

    if st.button("Ask AI"):
        if question:
            with st.spinner("🤖 AI is thinking..."):
                prompt = f"Answer the following question clearly:\n{question}"
                result = llm.invoke(prompt)
            st.success("Answer Generated")
            st.write(result.content)

# =========================
# CHAT WITH PDF
# =========================
elif menu == "Chat with PDF":
    st.subheader("📚 Chat With Your PDF")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        file_path = os.path.join("documents", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF Uploaded Successfully")

        with st.spinner("🤖 Reading PDF..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = splitter.split_documents(docs)
            vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="vector_db")
            retriever = vectordb.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        question = st.text_input("Ask question from PDF")
        if question:
            with st.spinner("🤖 Searching answer..."):
                response = qa.invoke({"query": question})
            st.write(response["result"])
            st.success("Answer Found")
            st.write(answer)

# =========================
# QUIZ GENERATOR
# =========================
elif menu == "Quiz Generator":
    st.subheader("🧠 AI Quiz Generator")
    topic = st.text_input("Enter Topic")
    if st.button("Generate Quiz"):
        if topic:
            with st.spinner("🤖 Creating quiz..."):
                prompt = f"""
                Create 5 multiple choice questions about {topic}.

                Format:

                Question
                A)
                B)
                C)
                D)

                Correct Answer:
                """
                result = llm.invoke(prompt)
            st.success("Quiz Generated")
            st.write(result.content)

# =========================
# STUDY PLANNER + STREAK + NEXT TOPIC
# =========================
elif menu == "Study Planner":

    st.subheader("📅 AI Study Planner")

    subject = st.text_input("Enter Subject")
    days = st.number_input("Number of days to prepare", min_value=1, max_value=365)

    if subject:

        PROGRESS_FILE = "study_progress.json"
        PLAN_FILE = "study_plan.json"

        # LOAD PROGRESS DATA
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                progress_data = json.load(f)
        else:
            progress_data = {}

        # LOAD PLAN DATA
        if os.path.exists(PLAN_FILE):
            with open(PLAN_FILE, "r") as f:
                plan_data = json.load(f)
        else:
            plan_data = {}

        user_key = subject.strip().lower()
        today_str = datetime.now().strftime("%Y-%m-%d")

        # INITIALIZE USER DATA
        if user_key not in progress_data:
            progress_data[user_key] = {
                "streak": 0,
                "last_date": None,
                "completed_days": 0
            }

        if user_key not in plan_data:
            plan_data[user_key] = []

        data = progress_data[user_key]
        schedule = plan_data[user_key]

        # =========================
        # GENERATE STUDY PLAN
        # =========================
        if st.button("🧠 Generate Study Plan", key="generate_plan"):

            with st.spinner("🤖 Creating AI study schedule..."):

                prompt = f"""
Create a {days}-day study plan for {subject}.

Format:
Day 1: topic
Day 2: topic
Day 3: topic
"""

                result = llm.invoke(prompt)

                schedule = [line.strip() for line in result.content.split("\n") if line.strip()]

                plan_data[user_key] = schedule

                with open(PLAN_FILE, "w") as f:
                    json.dump(plan_data, f)

                st.success("Study Plan Generated!")

        # =========================
        # SHOW STUDY PLAN
        # =========================
        if plan_data[user_key]:

            st.subheader("📚 Your Study Plan")

            for day in plan_data[user_key]:
                st.write(day)

        # =========================
        # MARK TODAY COMPLETE
        # =========================
        if st.button("✅ Mark Today's Study Complete", key="complete_study"):

            if data["last_date"] != today_str:

                data["completed_days"] += 1

                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

                if data["last_date"] == yesterday:
                    data["streak"] += 1
                else:
                    data["streak"] = 1

                data["last_date"] = today_str

                progress_data[user_key] = data

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress_data, f)

                st.success("Today's study marked complete!")

            else:
                st.warning("You already marked today as complete.")

        # =========================
        # SHOW STREAK
        # =========================
        st.write(f"🔥 **Daily Study Streak:** {data['streak']} days")

        # =========================
        # SHOW PROGRESS BAR
        # =========================
        progress_percent = min(data["completed_days"] / days, 1.0)
        st.progress(progress_percent)

        # =========================
        # NEXT TOPIC FROM PLAN
        # =========================
        if st.button("🤖 What should I study next?", key="next_topic"):

            day_index = data["completed_days"]

            if day_index < len(plan_data[user_key]):

                st.success("📖 Next Study Task")
                st.write(plan_data[user_key][day_index])

            else:
                st.info("🎉 You have completed your study plan!")
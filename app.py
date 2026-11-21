
import streamlit as st
import torch
import re
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="✨ Nova AI | Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CREATIVE CSS ----------------
st.markdown("""
<style>

/* Animated colourful background */
.stApp {
    background: linear-gradient(
        -45deg,
        #10002b,
        #240046,
        #3c096c,
        #5a189a,
        #1d3557,
        #003566
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.hero-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    padding: 15px;
}

.hero-title span {
    background: linear-gradient(
        90deg, #00f5d4, #00bbf9, #f15bb5, #fee440
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

@keyframes glow {
    from {filter: drop-shadow(0 0 5px #00f5d4);}
    to {filter: drop-shadow(0 0 18px #f15bb5);}
}

.hero-subtitle {
    text-align: center;
    color: #e0e7ff;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* Feature cards */
.feature-card {
    background: linear-gradient(
        135deg,
        rgba(0,245,212,0.18),
        rgba(241,91,181,0.18)
    );
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 125px;
    transition: 0.3s;
}

.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 25px rgba(0,245,212,0.25);
}

.feature-card h3 {
    color: #00f5d4;
}

/* Text areas */
.stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    color: #17112b !important;
    border-radius: 14px !important;
    border: 2px solid #00f5d4 !important;
    font-size: 16px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #00bbf9, #9b5de5, #f15bb5);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    padding: 12px 25px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.04);
    box-shadow: 0 5px 20px rgba(241,91,181,0.5);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10002b, #3c096c, #240046);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* General text */
p, label, .stMarkdown {
    color: #ffffff;
}

h2, h3 {
    color: #00f5d4 !important;
}

hr {
    border-color: rgba(255,255,255,0.3);
}

</style>
""", unsafe_allow_html=True)


# ---------------- AI MODELS ----------------

MODELS = {
    "⚡ DistilBART - Faster": "sshleifer/distilbart-cnn-12-6",
    "💎 BART Large - Quality": "facebook/bart-large-cnn"
}

LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250)
}

CHUNK_WORDS = 450


# ---------------- LOAD MODEL ----------------

@st.cache_resource(show_spinner=False)
def load_model(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------------- TEXT PROCESSING ----------------

def chunk_text(text):

    words = text.split()

    return [
        " ".join(words[i:i + CHUNK_WORDS])
        for i in range(0, len(words), CHUNK_WORDS)
    ]


def summarize_chunk(text, tokenizer, model, device, min_len, max_len):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        summary_ids = model.generate(
            **inputs,
            min_length=min_len,
            max_length=max_len,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def summarize(text, tokenizer, model, device, min_len, max_len):

    chunks = chunk_text(text)

    progress = st.progress(0, text="Preparing your summary...")

    summaries = []

    for i, chunk in enumerate(chunks):

        words = len(chunk.split())

        actual_min = max(5, min(min_len, words // 2))

        summary = summarize_chunk(
            chunk,
            tokenizer,
            model,
            device,
            actual_min,
            max_len
        )

        summaries.append(summary)

        progress.progress(
            (i + 1) / len(chunks),
            text=f"✨ Processing section {i + 1} of {len(chunks)}"
        )

    progress.empty()

    final_summary = " ".join(summaries)

    if len(chunks) > 1 and len(final_summary.split()) > CHUNK_WORDS:

        final_summary = summarize_chunk(
            final_summary,
            tokenizer,
            model,
            device,
            min_len,
            max_len
        )

    return final_summary


# ---------------- PDF / TXT READER ----------------

def read_file(file):

    if file.name.lower().endswith(".pdf"):

        reader = PdfReader(file)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    return file.read().decode("utf-8", errors="ignore")


# ---------------- HERO SECTION ----------------

st.markdown(
    '<div class="hero-title">📝 <span>✨ NOVA AI SUMMARIZER</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    '🚀 Transform lengthy documents into smart, simple and meaningful summaries!'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------- FEATURE CARDS ----------------

c1, c2, c3, c4 = st.columns(4)

features = [
    ("🤖", "AI Powered", "Intelligent summarization"),
    ("📄", "PDF & TXT", "Multiple file formats"),
    ("⚡", "Fast Results", "Save your valuable time"),
    ("🔒", "Local Processing", "No API key required")
]

for col, (icon, title, description) in zip(
    [c1, c2, c3, c4], features
):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <h1>{icon}</h1>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.markdown("# 🌟 NOVA AI")

st.sidebar.markdown("### ⚙️ Personalize Your Summary")

model_name = st.sidebar.selectbox(
    "🧠 Choose AI Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "📏 Summary Length",
    list(LENGTHS.keys()),
    index=1
)

output_format = st.sidebar.selectbox(
    "📝 Output Style",
    ["Paragraph", "Bullet Points"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 About Nova AI")

st.sidebar.info(
    """
    Nova AI uses advanced BART transformer models.

    ✨ Smart summarization

    📚 Document understanding

    ⚡ Fast processing

    🔐 Local AI processing
    """
)

st.sidebar.markdown("---")

st.sidebar.caption("Made with ❤️ using Python & Streamlit")


# ---------------- INPUT SECTION ----------------

st.markdown("## 📝 Start Your Summarization Journey")

source = st.radio(
    "Choose your input method",
    ["✍️ Paste Text", "📂 Upload File"],
    horizontal=True
)

text = ""

if source == "✍️ Paste Text":

    text = st.text_area(
        "Enter your content below",
        height=280,
        placeholder="✨ Paste your article, notes, research paper or document here..."
    )

else:

    uploaded_file = st.file_uploader(
        "📂 Upload your document",
        type=["pdf", "txt"],
        help="Supported formats: PDF and TXT"
    )

    if uploaded_file:

        try:
            text = read_file(uploaded_file)

            st.success(
                f"✅ Successfully loaded: {uploaded_file.name}"
            )

            with st.expander("👀 Preview your document"):

                st.write(text[:4000])

                if len(text) > 4000:
                    st.caption("Preview limited to 4000 characters.")

        except Exception as e:

            st.error(f"Unable to read the file: {e}")


# ---------------- TEXT STATISTICS ----------------

if text.strip():

    word_count = len(text.split())

    char_count = len(text)

    reading_time = max(1, round(word_count / 200))

    st.markdown("### 📊 Your Content Overview")

    m1, m2, m3 = st.columns(3)

    m1.metric("📝 Total Words", f"{word_count:,}")

    m2.metric("🔤 Characters", f"{char_count:,}")

    m3.metric("⏱️ Reading Time", f"{reading_time} min")


# ---------------- SUMMARIZE BUTTON ----------------

st.markdown("---")

left, center, right = st.columns([1, 2, 1])

with center:

    summarize_button = st.button(
        "🚀 GENERATE SMART SUMMARY",
        type="primary",
        use_container_width=True
    )


if summarize_button:

    if not text.strip():

        st.warning("⚠️ Please enter some text or upload a document.")

    elif len(text.split()) < 50:

        st.warning("⚠️ Please provide at least 50 words.")

    else:

        try:

            with st.spinner(
                "🤖 Nova AI is understanding your content..."
            ):

                tokenizer, model, device = load_model(
                    MODELS[model_name]
                )

                min_len, max_len = LENGTHS[length]

                summary = summarize(
                    text,
                    tokenizer,
                    model,
                    device,
                    min_len,
                    max_len
                )

            if not summary.strip():
                st.error("The model could not generate a summary. Try another model.")
                st.stop()

            st.session_state["summary"] = summary
            st.session_state["original"] = text

            st.success("🎉 Your summary is ready!")

        except Exception as e:

            st.error(f"Something went wrong: {e}")


# ---------------- SUMMARY RESULTS ----------------

if "summary" in st.session_state:

    summary = st.session_state["summary"]

    original = st.session_state["original"]

    summary_words = len(summary.split())

    original_words = len(original.split())

    reduction = max(
        0,
        round((1 - summary_words / original_words) * 100)
    )

    st.markdown("---")

    st.markdown("## 💎 Your AI Generated Summary")

    # Statistics

    a, b, c = st.columns(3)

    a.metric("📄 Original Words", f"{original_words:,}")

    b.metric("✨ Summary Words", f"{summary_words:,}")

    c.metric("📉 Reduction", f"{reduction}%")

    st.write("")

    # Output format

    if output_format == "Bullet Points":

        sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

        formatted_summary = "\n".join(
            f"• {sentence}" for sentence in sentences if sentence
        )

    else:

        formatted_summary = summary

    # Summary card

    st.markdown(
        """
        <div class="glass-card">
            <h2>✨ Your Smart Summary</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.text_area(
        "Generated Summary",
        value=formatted_summary,
        height=250
    )

    # Downloads

    d1, d2 = st.columns(2)

    with d1:

        st.download_button(
            "📥 Download Summary",
            data=formatted_summary,
            file_name="Nova_AI_Summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    with d2:

        st.download_button(
            "📄 Download Original Text",
            data=original,
            file_name="Original_Text.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="glass-card">
            <h3>🌟 Keep Learning with Nova AI!</h3>
            <p>
            Turn long documents into easy-to-understand information.
            Learn faster, understand better and save time with AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔄 Start New Summary"):

        del st.session_state["summary"]
        del st.session_state["original"]

        st.rerun()


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <h3>✨ NOVA AI SUMMARIZER</h3>
        <p>Intelligence meets simplicity.</p>
        <p>Made with 💜 Python | Streamlit | Hugging Face Transformers</p>
        <p>🚀 Learn More. Read Less. Achieve More.</p>
    </div>
    """,
    unsafe_allow_html=True
)

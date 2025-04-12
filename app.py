import streamlit as st
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API setup
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
QA_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
CLAUSE_CLASSIFIER_API = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Streamlit UI setup
st.set_page_config(page_title="LawMate", layout="wide")
st.title("LawMate AI")
st.markdown("**Document Summarizer, Q&A Assistant & Clause Comparator**")

# Sidebar setup
st.sidebar.title("Actions")
action = st.sidebar.radio("Choose an action", ["Summarize Document", "Ask a Question", "Classify Clauses", "Compare Documents"])

# File uploaders
uploaded_file_1 = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"], key="doc1")

# Extract text helper
def extract_text(file):
    if file.type == "application/pdf":
        import fitz  # PyMuPDF
        pdf_content = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_content += page.get_text()
        return pdf_content
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return None

# Summarization function
def summarize_text(text):
    payload = {"inputs": text[:3000]}  #long docs
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        return "API Error: " + str(response.json())

# Q&A function
def answer_question(question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context[:3000],
        }
    }
    response = requests.post(QA_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "No answer found.")
    else:
        return "API Error: " + str(response.json())

# Clause classification function
def classify_clauses(text):
    candidate_labels = [
        "Confidentiality",
        "Termination",
        "Liability",
        "Payment Terms",
        "Governing Law",
        "Dispute Resolution"
    ]
    payload = {
        "inputs": text[:3000],
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }
    response = requests.post(CLAUSE_CLASSIFIER_API, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()}

# Document comparison helper
def compare_documents(text1, text2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score * 100, 2)
    except:
        return 0

#UI logic
document_text = ""
if uploaded_file_1 is not None:
    document_text = extract_text(uploaded_file_1)

    if document_text:
        st.success("File uploaded and text extracted successfully!")

        with st.expander("View Extracted Document Text"):
            st.write(document_text)

        if action == "Summarize Document":
            if st.button("Summarize Document"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(document_text)
                st.subheader("Summary")
                st.write(summary)

        elif action == "Ask a Question":
            st.subheader("Ask a Question")
            user_question = st.text_input("Enter your legal question based on the document")
            if st.button("Get Answer") and user_question.strip():
                with st.spinner("Finding answer..."):
                    answer = answer_question(user_question, document_text)
                st.success("Answer:")
                st.write(answer)

        elif action == "Classify Clauses":
            st.subheader("Clause Classification")
            if st.button("Classify Clauses"):
                with st.spinner("Classifying clauses..."):
                    result = classify_clauses(document_text)
                if "labels" in result:
                    st.success("Classification Complete")
                    sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)
                    st.subheader("Clause Classification Results")
                    for label, score in sorted_results:
                        st.write(f"**{label}**: {score:.2f}")
                else:
                    st.error("Failed to classify clauses.")

        elif action == "Compare Documents":
            st.subheader("Document Comparison")
            uploaded_file_2 = st.file_uploader("Upload second document for comparison", type=["pdf", "txt", "docx"], key="doc2")
            if uploaded_file_2 is not None:
                document_text_2 = extract_text(uploaded_file_2)
                if st.button("Compare Documents"):
                    with st.spinner("Comparing..."):
                        similarity_score = compare_documents(document_text, document_text_2)
                    st.success("Similarity Score")
                    st.write(f"The documents are **{similarity_score}%** similar.")

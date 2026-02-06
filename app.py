from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=None,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})



@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    user_msg = request.form["msg"].strip()
    
    
    greetings = ["hi", "hii", "hello", "hey", "good morning", "good evening"]
    if user_msg in greetings:
        return "Hello! I’m your medical assistant. Ask me any medical question."

    # 1️⃣ Retrieve relevant medical context
    docs = retriever.invoke(user_msg)

    if not docs:
        return "I could not find relevant medical information."

    context = "\n\n".join(doc.page_content for doc in docs)

    # 2️⃣ Build RAG prompt
    prompt = f"""
You are a medical assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{user_msg}

Answer:
(This is for educational purposes only. Consult a doctor.)
"""

    # 3️⃣ Call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

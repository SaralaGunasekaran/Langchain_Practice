import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ===============================================================
# SETUP: read the API key from the .env file
# ===============================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env file")


# ===============================================================
# 1. DATA INGESTION
#    Read the PDF and turn each page into a LangChain Document.
# ===============================================================
PDF_PATH = "python_basics_one_page.pdf"

reader = PdfReader(PDF_PATH)
documents = []

for page_number, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    documents.append(
        Document(
            page_content=text,
            metadata={"source": PDF_PATH, "page": page_number},
        )
    )

print("1. Data ingestion  -> pages loaded:", len(documents))


# ===============================================================
# 2. SPLITTER
#    Cut the pages into small overlapping chunks.
# ===============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # max characters per chunk
    chunk_overlap=50,   # characters shared between neighbouring chunks
)

chunks = splitter.split_documents(documents)

print("2. Splitter        -> chunks created:", len(chunks))


# ===============================================================
# 3. EMBEDDINGS
#    The model that turns text into a list of numbers (a vector).
# ===============================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
)

print("3. Embeddings      -> model ready: text-embedding-3-small")


# ===============================================================
# 4. VECTOR STORE
#    Embed every chunk and store the vectors so they can be searched.
# ===============================================================
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

print("4. Vector store    -> chunks stored:", len(chunks))


# ===============================================================
# 5. RETRIEVER
#    Given a question, return the k most similar chunks.
# ===============================================================
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

print("5. Retriever       -> ready (k=2)")


# ===============================================================
# 6. LLM + PROMPT
#    The chat model that writes the final answer from the chunks.
# ===============================================================
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context.
If the answer is not in the context, say it is not available in the PDF.

Context:
{context}

Question:
{question}
""")


# ===============================================================
# 7. ASK QUESTIONS IN A LOOP                      <-- CHANGED
#    retriever -> prompt -> LLM -> answer, repeated until you quit
#    (steps 1-6 above run only once, so the PDF is not re-embedded)
# ===============================================================
print("\nAsk questions about the PDF. Type 'exit' to stop.")

while True:                                        # <-- NEW: repeat forever
    question = input("\nYour question: ").strip()

    # NEW: stop when the user types exit/quit/q
    if question.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break

    # NEW: ignore empty input and ask again
    if not question:
        continue

    # 7a. Retriever finds the most relevant chunks
    retrieved_chunks = retriever.invoke(question)

    # 7b. Join the chunk texts into one block of context
    context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)

    # 7c. Fill the prompt with the context and the question
    filled_prompt = prompt.invoke({"context": context, "question": question})

    # 7d. Send the prompt to the LLM
    response = llm.invoke(filled_prompt)

    print("\nAnswer:")
    print(response.content)
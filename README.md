# PDF Question Answering with LangChain

Ask questions about a PDF and get answers based only on its content.
The project uses Retrieval-Augmented Generation (RAG): it finds the parts of
the PDF that are relevant to your question and gives them to an OpenAI chat
model to write the answer.

## How it works

```
PDF -> Data ingestion -> Splitter -> Embeddings -> Vector store -> Retriever -> LLM -> Answer
```

| # | Stage | What it does | Tool used |
|---|-------|--------------|-----------|
| 1 | Data ingestion | Reads the PDF and turns each page into a `Document` | `pypdf`, `langchain_core.documents.Document` |
| 2 | Splitter | Cuts pages into small overlapping chunks (500 characters, 50 overlap) | `RecursiveCharacterTextSplitter` |
| 3 | Embeddings | Converts text into vectors (lists of numbers that capture meaning) | `OpenAIEmbeddings` (`text-embedding-3-small`) |
| 4 | Vector store | Stores the chunk vectors so they can be searched | `InMemoryVectorStore` |
| 5 | Retriever | Returns the 2 chunks most similar to the question | `vector_store.as_retriever(k=2)` |
| 6 | LLM + prompt | Writes the answer using only the retrieved chunks | `ChatOpenAI` (`gpt-4.1-mini`) |

Stages 1 to 6 run once at startup. After that you can ask as many questions
as you like in the same run. Each question only does one retrieval and one
LLM call.

## Project structure

```
your-project/
├── traced_app.py          # the main script
├── python_basics_one_page.pdf   # the PDF to ask questions about
├── .env                         # your OpenAI API key (never share or commit this)
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key (https://platform.openai.com/api-keys)

Python packages:

```
pypdf
python-dotenv
langchain-core
langchain-openai
langchain-text-splitters
```

## Setup

1. **Create and activate a virtual environment** (recommended)

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the packages**

   ```
   pip install pypdf python-dotenv langchain-core langchain-openai langchain-text-splitters
   ```

3. **Add your API key** by creating a file named `.env` in the project folder:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Add your PDF** to the project folder. The script looks for
   `python_basics_one_page.pdf`. To use a different file, change `PDF_PATH`
   in the script.

## Run

```
python traced_app.py
```

Example session:

```
1. Data ingestion  -> pages loaded: 1
2. Splitter        -> chunks created: 3
3. Embeddings      -> model ready: text-embedding-3-small
4. Vector store    -> chunks stored: 3
5. Retriever       -> ready (k=2)

Ask questions about the PDF. Type 'exit' to stop.

Your question: What is a variable?

Answer:
...

Your question: exit
Goodbye!
```

Type `exit`, `quit` or `q` to stop. Pressing Enter on an empty line just asks again.

## Settings you can change

| Setting | Where | Effect |
|---------|-------|--------|
| `chunk_size`, `chunk_overlap` | Stage 2 | Smaller chunks give more precise matches, larger chunks give more context |
| `k` in `search_kwargs` | Stage 5 | How many chunks are sent to the model |
| `model` in `ChatOpenAI` | Stage 6 | Which OpenAI chat model writes the answer |
| Prompt text | Stage 6 | Change how the model answers, for example its tone or language |

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| `OPENAI_API_KEY is missing from .env file` | The `.env` file is missing, in the wrong folder, or the variable name is misspelled |
| `FileNotFoundError` for the PDF | The PDF is not in the folder you run the script from, or `PDF_PATH` is wrong |
| Answers say "not available in the PDF" | The retriever may not have found the right chunk. Print `retrieved_chunks` inside the loop to check, or try a larger `k` |
| Empty or very short text from the PDF | The PDF may be a scanned image with no text layer. `pypdf` cannot read images, so you would need OCR first |
| `ModuleNotFoundError` | A package is missing. Re-run the `pip install` command above |
| `AuthenticationError` / `RateLimitError` | The API key is invalid, or your OpenAI account has no credit or hit its rate limit |

## Notes

- **No `langchain-community`.** That package is no longer maintained, so this
  project avoids it. The PDF is read with `pypdf` directly and the vector
  store comes from `langchain-core`.
- **The vector store lives in memory.** It is rebuilt each time you start the
  script, which costs a small number of embedding calls. That is fine for a
  small PDF. For large documents, use a persistent store such as Chroma
  (`pip install langchain-chroma`).
- **No chat memory.** Each question is answered on its own, so follow-ups like
  "explain that more" will not work. Write each question so it makes sense by itself.
- **Answers are limited to the PDF.** The prompt tells the model to say the
  information is not available when the context does not contain it.
- **Keep your API key private.** Add `.env` to your `.gitignore` if you use Git.

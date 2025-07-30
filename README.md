# PDF QA Bot with RAG (Watsonx.ai)


## Create a .env file with the following variables:
- WATSONX_API_KEY="YOUR_IBM_CLOUD_API_KEY"  
- WATSONX_PROJECT_ID="YOUR_WATSONX_PROJECT_ID"
- GRADIO_SERVER_PORT="7861" # optional, if port 7860 is often busy

## npm install
## run project using: 
- python qabot.py


This project implements a Question-Answering (QA) bot that leverages Retrieval-Augmented Generation (RAG) to answer questions based on the content of an uploaded PDF document. 
It uses the LangChain framework, integrated with IBM Watsonx.ai for Large Language Models (LLMs) and embeddings, and provides a user-friendly interface powered by Gradio.

## Application Process Flow

The application processes your PDF and query through a series of interconnected steps to generate accurate responses.

### 1. User Interaction (Gradio Frontend)

The process begins with the Gradio web interface, which serves as the front-end for the application.

* **Input**:
    * You upload a **PDF file** using the "Upload PDF File" drag-and-drop component. This `file_path` is then passed to the backend.
    * You enter your **query** (question) in the "Input Query" textbox. This `query` string is also sent to the backend.

These two inputs (`file_path` and `query`) are passed to the main `retriever_qa` function, which orchestrates the entire RAG pipeline.

### 2. Document Loading (`document_loader` function)

* **Input**: `file_path` (the path to your uploaded PDF).
* **Process**: The `PyPDFLoader` from `langchain_community.document_loaders` is used to read and parse the content of the specified PDF file.
* **Output**: A `List[Any]` named `loaded_document`, containing the extracted text from the PDF, typically as `Document` objects.

### 3. Text Splitting (`text_splitter` function)

* **Input**: `data` (which is the `loaded_document` from the previous step).
* **Process**: The `RecursiveCharacterTextSplitter` from `langchain.text_splitter` takes the large document and breaks it down into smaller, manageable `chunks`. 
This is crucial for efficient processing by the embedding model and LLM, preventing token limit issues. It defines `chunk_size` (1000 characters) and `chunk_overlap` (200 characters) to maintain context between chunks.
* **Output**: A `List[Any]` named `chunks`, where each item is a smaller `Document` object.

### 4. Embedding Model (`watsonx_embedding` function)

* **Purpose**: To convert the text chunks into numerical vector representations (embeddings). These embeddings capture the semantic meaning of the text, allowing for efficient similarity searches.
* **Process**: This function initializes `WatsonxEmbeddings` from `langchain_ibm`, using the `ibm/slate-125m-english-rtrvr` model hosted on Watsonx.ai. It requires your `WATSONX_PROJECT_ID` (loaded from `.env`).
* **Output**: An `embedding_model` object, capable of generating embeddings for text.

### 5. Vector Database (`vector_database` function)

* **Input**: `chunks` (the text chunks from the `text_splitter`) and the `embedding_model` (from `watsonx_embedding`).
* **Process**: The `Chroma` vector store from `langchain_community.vectorstores` is used. It takes the `chunks` and, using the `embedding_model`, generates embeddings for each chunk. These embeddings are then stored in the `Chroma` database, making them searchable.
* **Output**: A `vectordb` object, which is an in-memory vector store containing the embedded chunks.

### 6. Retriever (`retriever` function)

* **Input**: `file_path` (which implicitly triggers the `document_loader`, `text_splitter`, and `vector_database` functions internally).
* **Process**: This function builds upon the `vectordb` by converting it into a `retriever` object (`vectordb.as_retriever()`). The retriever's job is to fetch the most relevant `chunks` from the `vectordb` based on a given query.
* **Output**: A `retriever_obj` capable of finding relevant documents.

### 7. Large Language Model (LLM) Initialization (`get_llm` function)

* **Purpose**: To generate human-like text responses based on the retrieved information and the user's query.
* **Process**: This function initializes a `WatsonxLLM` instance from `langchain_ibm`, pointing to the `meta-llama/llama-2-70b-chat` model hosted on Watsonx.ai. It configures the LLM with specific generation parameters (`llm_params`) like decoding method, max tokens, temperature, etc. It also requires your `WATSONX_PROJECT_ID` (from `.env`).
* **Output**: An `llm` object, which is the initialized language model.

### 8. QA Chain (`retriever_qa` function)

* **Input**: `file_path` (to set up the `retriever`) and `query` (the user's question).
* **Process**: This is the core of the RAG pipeline.
    1.  It first retrieves the relevant `llm` and `retriever_obj`.
    2.  It then constructs a `RetrievalQA` chain using `RetrievalQA.from_chain_type`.
    3.  When `qa.invoke({"query": query})` is called:
        * The `retriever_obj` finds the most semantically similar `chunks` from the `vectordb` (based on your `query`).
        * These retrieved `chunks` are then combined with your original `query` and fed into the `llm` (the `meta-llama/llama-2-70b-chat` model).
        * The LLM uses this combined context to generate an informed answer.
* **Output**: The `response['result']` string, which is the final answer generated by the LLM based on the PDF content.

## Dependencies

* `ibm_watsonx_ai`: Core SDK for interacting with Watsonx.ai services.
* `langchain_ibm`: LangChain integration for IBM Watsonx.ai LLMs and embeddings.
* `langchain`: Core LangChain library for building RAG applications.
* `langchain_community`: Contains community-contributed LangChain components like `PyPDFLoader` and `Chroma`.
* `gradio`: For creating the interactive web user interface.
* `python-dotenv`: To load environment variables from a `.env` file for secure credential management.
* `pypdf`: Required by `PyPDFLoader` to parse PDF files.

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-project-directory>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv qa_bot_env
    # On Windows:
    .\qa_bot_env\Scripts\activate
    # On Linux/macOS:
    source qa_bot_env/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # (assuming you've created one)
    # Alternatively, install manually:
    pip install ibm-watsonx-ai langchain-ibm langchain langchain-community gradio python-dotenv pypdf
    ```
4.  **Create a `.env` file:**
    In the root directory of your project, create a file named `.env` and add your IBM Watsonx.ai credentials:
    ```dotenv
    WATSONX_API_KEY="YOUR_IBM_CLOUD_API_KEY"
    WATSONX_PROJECT_ID="YOUR_WATSONX_PROJECT_ID"
    # Optional: GRADIO_SERVER_PORT="7861" if port 7860 is often busy
    ```
    **Remember to replace `YOUR_IBM_CLOUD_API_KEY` and `YOUR_WATSONX_PROJECT_ID` with your actual credentials.**
    **Do NOT commit this file to version control (it's ignored by `.gitignore`).**
5.  **Run the application:**
    ```bash
    python qabot.py
    ```
    The application will launch in your web browser, typically at `http://127.0.0.1:7860` (or the port specified in your `.env` or `launch()` call).
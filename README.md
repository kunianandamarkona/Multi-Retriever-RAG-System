text
# **Hybrid Multi-Search RAG System**

The **Hybrid Multi-Search RAG System** is an intelligent document query system that uses **retrieval-augmented generation (RAG)**. It combines **BM25 keyword-based search**, **FAISS-based semantic similarity search**, and a **CXAI-powered LLM** to provide accurate and context-aware answers to user queries. This system processes PDF documents, splits them into retrievable chunks, and provides concise responses using a combination of retrieval strategies and generative models.

---

## **Features**

- **Hybrid Retrieval**:
  - Combines **BM25 (keyword-based search)**, **FAISS (semantic similarity search)**, and **Maximum Marginal Relevance (MMR)** for multi-strategy document retrieval.
- **Retrieval-Augmented Generation (RAG)**:
  - Uses the retrieved document context to generate expert-level answers to user questions.
- **PDF Document Processing**:
  - Efficiently processes and splits multi-page PDF documents into indexed chunks for retrieval.
- **Custom Embeddings**:
  - Leverages the **CXAI Playground API** for generating embeddings for semantic search.
- **Interactive Command-Line Interface**:
  - Allows users to interactively query the system and receive concise, context-aware answers.

---

## **System Requirements**

- **Python Version**: `3.11.12`  
  Ensure Python 3.11.12 is installed on your system. Verify your Python version using:
python --version

text

---

## **Dependencies**

The following Python libraries are required:

faiss-cpu
langchain
langchain-community
langchain-openai
langchain-pymupdf4llm
langchain-text-splitters
sentence-transformers
numpy
PyMuPDF
python-dotenv
requests
rank-bm25

text

Install all dependencies using:
pip install -r requirements.txt

text

---

## **Setup and Installation**

1. **Clone the Repository**:
git clone <repository_url>
cd <repository_folder>

text

2. **Install Dependencies**:
pip install -r requirements.txt

text

3. **Configure Environment Variables**:
Create a `.env` file in the project directory with your CXAI Playground API token:
CXAI_PLAYGROUND_ACCESS_TOKEN=<your_api_token>

text

4. **Prepare the PDF Document**:
- Place the target PDF file in the project directory.
- Update the `pdf_path` variable in the script to point to your PDF file.

---

## **How to Run**

Run the script to start the interactive question-answering system:
python script.py

text

**Example Usage**:
Enter your question: How does QoS work in ASR9K?

text

**Exit the Application**:
Type exit to close the application.

text

---

## **File Structure**

Hybrid-Multi-Search-RAG/
├── script.py # Main Python script
├── requirements.txt # List of dependencies
├── README.md # Project documentation
├── .env # Environment variables (not included in GitHub)
├── example_document.pdf # Replace with your target PDF file

text

---

## **Key Components**

- **PDF Processing**: Uses `PyMuPDF` and `langchain-pymupdf4llm` to extract text from PDFs and split them into chunks.
- **Custom Embeddings**: Generates embeddings via the CXAI Playground API for semantic search.
- **Hybrid Retrieval**: Combines BM25, FAISS, and MMR for robust retrieval.
- **Retrieval-Augmented Generation (RAG)**: Uses retrieved context to generate answers with a CXAI-powered LLM.

---

## **Future Improvements**

- **Enhanced Retrieval**: Add support for additional scoring mechanisms.
- **Multilingual Support**: Expand to multilingual datasets.
- **Web Interface**: Develop a user-friendly web interface.

---

## **Acknowledgments**

- **CXAI Playground API**: For embeddings and LLM capabilities.
- **LangChain**: For retrieval-augmented generation.
- **FAISS**: For efficient similarity search.
- **PyMuPDF**: For PDF parsing.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Additional Notes**

- Ensure your `.env` file contains a valid CXAI Playground API token.
- Ensure the target PDF file is accessible at the path specified in the script.

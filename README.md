

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
  Ensure Python 3.11.12 is installed on your system. You can verify your Python version using:
  ```bash
  python --version


Dependencies

The following Python libraries are required for this project:


faiss-cpu (for FAISS-based vector similarity search)
langchain (core framework for retrieval-augmented generation)
langchain-community (BM25Retriever and FAISS integrations)
langchain-openai (for using CXAI-powered LLMs)
langchain-pymupdf4llm (for PDF processing)
langchain-text-splitters (for splitting large texts into retrievable chunks)
sentence-transformers (for semantic embeddings)
numpy (for numerical computations)
PyMuPDF (for PDF parsing and processing)
python-dotenv (for managing environment variables)
requests (for making API calls)
rank-bm25 (for BM25-based keyword search)

Install all dependencies using:


bash
Copy Code
pip install -r requirements.txt


Setup and Installation

1. Clone the Repository

Clone the repository to your local machine:


bash
Copy Code
git clone <repository_url>
cd <repository_folder>

2. Install Dependencies

Install the required Python libraries:


bash
Copy Code
pip install -r requirements.txt

3. Configure Environment Variables

Create a .env file in the project directory to store your CXAI Playground API token:
plaintext
Copy Code
CXAI_PLAYGROUND_ACCESS_TOKEN=<your_api_token>

4. Prepare the PDF Document

Place the target PDF file in the project directory.
Update the pdf_path variable in the script to point to your PDF file.


How to Run

Run the script to start the interactive question-answering system:


bash
Copy Code
python script.py

You can then ask questions about the content of the PDF document. For example:


plaintext
Copy Code
Enter your question: How does QoS work in ASR9K?

Exit the Application:

Type exit to close the application.



File Structure

The repository includes the following files:


plaintext
Copy Code
Hybrid-Multi-Search-RAG/
├── script.py                  # Main Python script
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
├── .env                       # Environment variables \(not included in GitHub\)
├── example_document.pdf       # Replace with your target PDF file


Example Usage

Input:

plaintext
Copy Code
Enter your question: How to configure BGP?

Output:

plaintext
Copy Code
Answer: To configure BGP, follow these steps: ...

The system retrieves the most relevant content from the PDF, combines it into context, and uses the CXAI-powered LLM to generate a concise and accurate answer.



Key Components

PDF Processing:

The script uses PyMuPDF and langchain-pymupdf4llm to extract text from PDFs and split them into chunks for retrieval.
Custom Embeddings:

Uses the CXAI Playground API to generate embeddings for semantic search.
Hybrid Retrieval:

Combines BM25 keyword-based search, FAISS-based similarity search, and MMR for robust and multi-strategy retrieval.
Retrieval-Augmented Generation (RAG):

Uses the context retrieved by the hybrid retrievers to generate answers with a CXAI-powered LLM.


Future Improvements

Enhanced Retrieval: Add support for additional scoring mechanisms and retrieval strategies.
Multilingual Support: Expand capabilities to support multilingual datasets.
Web Interface: Provide a user-friendly web interface for easier interaction.


Acknowledgments

CXAI Playground API: For embeddings and LLM capabilities.
LangChain: For retrieval-augmented generation and hybrid retrieval strategies.
FAISS: For efficient similarity search.
PyMuPDF: For PDF parsing and processing.


License

This project is licensed under the MIT License. See the LICENSE file for details.



Additional Notes

Ensure your .env file contains a valid API token for the CXAI Playground.
Ensure the target PDF file is accessible at the path specified in the script.

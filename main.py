from dotenv import load_dotenv
import os
import requests
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader  # PDF Processing
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage  # For LLM queries
from langchain_core.embeddings import Embeddings  # Import correct interface

# Load environment variables
load_dotenv()

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomEmbeddings(Embeddings):
    """
    CustomEmbeddings class to generate embeddings using the CXAI Playground API.
    Inherits from LangChain's Embeddings interface.
    """
    def __init__(self, openai_api_base, model):
        self.api_base = openai_api_base
        self.model = model
        self.api_key = os.getenv("CXAI_PLAYGROUND_ACCESS_TOKEN")  # Load API key from .env
        if not self.api_key:
            raise ValueError("API token is missing. Please add it to the .env file with the key CXAI_PLAYGROUND_ACCESS_TOKEN.")

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents.
        """
        return self.encode(texts)

    def embed_query(self, text):
        """
        Generate embedding for a single query.
        """
        return self.encode([text])[0]

    def encode(self, texts):
        """
        Generate embeddings using the CXAI Playground API.
        """
        if not isinstance(texts, list):
            raise ValueError("The input to `encode` must be a list of strings.")

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        json_data = {
            'model': self.model,
            'input': texts
        }

        try:
            # Send request to CXAI Playground API
            response = requests.post(f'{self.api_base}/embeddings', headers=headers, json=json_data)
            response.raise_for_status()  # Raise an error for non-200 responses

            # Parse response to extract embeddings
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            return np.array(embeddings)

        except requests.exceptions.RequestException as e:
            # Raise an error if the API call fails
            raise RuntimeError(f"Error with CXAI Playground API: {e}")


class HybridMultiSearchRAG:
    """
    HybridMultiSearchRAG class with CXAI Playground API embeddings.
    """
    def __init__(self, pdf_path: str, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the HybridMultiSearchRAG system.
        """
        # Initialize the embedding system
        self.embeddings = CustomEmbeddings(
            openai_api_base="https://cxai-playground.cisco.com",
            model=embedding_model
        )

        # Initialize the language model (CXAI Playground LLM)
        api_token = os.getenv("CXAI_PLAYGROUND_ACCESS_TOKEN")
        if not api_token:
            raise ValueError("API token is missing. Please add it to the .env file.")
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            base_url="https://cxai-playground.cisco.com",
            temperature=0.7,
            openai_api_key=api_token,
        )

        # Load and process the PDF
        self.pdf_path = pdf_path
        self.docs = self._load_pdf()
        self.hybrid_retriever = self._create_multi_retriever()

    def _load_pdf(self) -> list:
        """
        Load and split PDF documents.
        """
        try:
            loader = PyMuPDF4LLMLoader(self.pdf_path, extract_images=False, mode="page")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            return text_splitter.split_documents(docs)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")

    def _create_multi_retriever(self):
        """
        Create a multi-strategy retriever combining:
        1. Similarity Search
        2. Maximum Marginal Relevance (MMR)
        3. BM25 Retriever
        """
        print("Generating embeddings for documents...")
        document_texts = [doc.page_content for doc in self.docs]
        document_embeddings = self.embeddings.encode(document_texts)

        # Prepare text-embedding pairs for FAISS
        text_embedding_pairs = list(zip(document_texts, document_embeddings))
        metadatas = [{"page": i} for i in range(len(document_texts))]

        # Create FAISS vector store with precomputed embeddings
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        # Create individual retrievers
        similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5})
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 5

        # Combine retrievers into an ensemble
        return EnsembleRetriever(
            retrievers=[similarity_retriever, mmr_retriever, bm25_retriever],
            weights=[0.25, 0.25, 0.5]
        )

    def query(self, question: str) -> str:
        """
        Query the system with a question using RAG (retrieval-augmented generation).
        """
        try:
            if not isinstance(question, str):
                raise ValueError("The question must be a string.")

            # Retrieve relevant documents using the hybrid retriever
            retrieved_docs = self.hybrid_retriever.invoke(question)
            if not retrieved_docs:
                return "I do not know. No relevant information found."

            # Combine the retrieved documents into a context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])  # Use the top 3 documents

            # Construct the prompt for the LLM
            prompt = f"""You are an expert assistant. Use only the following context to answer the question.
If the answer is not in the context, say "I do not know."

Context:
{context}

Question:
{question}

Answer (concise, 1-7 sentences):"""

            # Use the LLM to generate the answer
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content if hasattr(response, "content") else str(response)

            return answer
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Specify the PDF file path
    pdf_path = "/Users/snanda2/Desktop/Cisco_rag/Cisco_API/ASR9K_QOS.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    rag = HybridMultiSearchRAG(pdf_path=pdf_path)

    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() == "exit":
            break
        try:
            answer = rag.query(user_question)
            print(answer)
        except Exception as e:
            print(f"Error: {str(e)}")

"""
Simple RAG Implementation using PaperQA

This example demonstrates how to use PaperQA for question-answering over documents.
PaperQA is designed for scientific papers but works well for general document QA.

Installation:
pip install paper-qa anthropic

Usage:
1. Place your documents in a directory
2. Run this script to ask questions about them
"""

import os
from pathlib import Path
from paperqa import Docs
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class SimplePaperQA:
    """Simple RAG implementation using PaperQA."""

    def __init__(self, docs_dir: str = None, llm_model: str = "claude-sonnet-4"):
        """
        Initialize the RAG system.
        
        Args:
            docs_dir: Directory containing documents to index
            llm_model: Claude model to use for answering questions
        """
        self.docs = Docs(llm=llm_model)
        self.docs_dir = docs_dir
        
        if docs_dir:
            self.index_documents(docs_dir)

    def index_documents(self, docs_dir: str):
        """
        Index all documents in the specified directory.
        
        Args:
            docs_dir: Path to directory containing documents
        """
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            raise ValueError(f"Directory not found: {docs_dir}")
        
        # Supported file types
        supported_extensions = ['.pdf', '.txt', '.md', '.html']
        
        indexed_count = 0
        for file_path in docs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    self.docs.add(str(file_path))
                    indexed_count += 1
                    print(f"Indexed: {file_path.name}")
                except Exception as e:
                    print(f"Error indexing {file_path.name}: {e}")
        
        print(f"\nTotal documents indexed: {indexed_count}")

    def add_document(self, file_path: str):
        """
        Add a single document to the index.
        
        Args:
            file_path: Path to the document file
        """
        try:
            self.docs.add(file_path)
            print(f"Added document: {Path(file_path).name}")
        except Exception as e:
            print(f"Error adding document: {e}")

    def query(self, question: str, max_sources: int = 5) -> dict:
        """
        Query the indexed documents.
        
        Args:
            question: Question to answer
            max_sources: Maximum number of source documents to use
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            answer = self.docs.query(question, k=max_sources)
            
            return {
                "question": question,
                "answer": answer.answer,
                "context": answer.context,
                "references": [
                    {
                        "text": ref.text,
                        "name": ref.name,
                        "page": getattr(ref, 'page', None)
                    }
                    for ref in answer.references
                ]
            }
        except Exception as e:
            return {
                "question": question,
                "answer": None,
                "error": str(e)
            }

    def chat(self):
        """Interactive chat loop for querying documents."""
        print("\n" + "="*60)
        print("PaperQA RAG System - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to end the session\n")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching documents...")
            result = self.query(question)
            
            if result.get("error"):
                print(f"\nError: {result['error']}")
                continue
            
            print("\n" + "-"*60)
            print("ANSWER:")
            print("-"*60)
            print(result["answer"])
            
            if result.get("references"):
                print("\n" + "-"*60)
                print("SOURCES:")
                print("-"*60)
                for i, ref in enumerate(result["references"], 1):
                    page_info = f" (Page {ref['page']})" if ref['page'] else ""
                    print(f"\n{i}. {ref['name']}{page_info}")
                    print(f"   {ref['text'][:200]}...")


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with a documents directory
    print("Example 1: Initialize RAG with documents directory")
    print("-" * 60)
    
    # Uncomment and modify the path to your documents
    # rag = SimplePaperQA(docs_dir="./documents")
    
    # Example 2: Manual document addition
    print("\nExample 2: Add documents manually")
    print("-" * 60)
    rag = SimplePaperQA()
    
    # Add individual documents
    # rag.add_document("path/to/your/document.pdf")
    # rag.add_document("path/to/another/document.txt")
    
    # Example 3: Query the system
    print("\nExample 3: Query documents")
    print("-" * 60)
    
    # result = rag.query("What is the main topic of the documents?")
    # print(f"Question: {result['question']}")
    # print(f"Answer: {result['answer']}")
    
    # Example 4: Interactive chat mode
    print("\nExample 4: Interactive chat mode")
    print("-" * 60)
    print("Uncomment the line below to start interactive mode:")
    print("# rag.chat()")
    
    # Uncomment to start interactive chat
    # rag.chat()
    
    print("\n" + "="*60)
    print("To use this script:")
    print("1. Install: pip install paper-qa anthropic")
    print("2. Set ANTHROPIC_API_KEY environment variable")
    print("3. Modify the examples above with your document paths")
    print("4. Run: python temp.py")
    print("="*60)

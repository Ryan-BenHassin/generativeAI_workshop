import os
# Prevent huggingface/tokenizers parallelism warning when the process is forked
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
# Reduce verbose logs from transformers/tokenizers (if present)
logging.getLogger("transformers").setLevel(logging.ERROR)

import chromadb
from groq import Groq
import textwrap
import numpy as np

# Initialize Groq client
GROQ_API_KEY = "API_KEY"  # Replace with your actual Groq API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Create collection for storing documents
collection = chroma_client.create_collection(
    name="hotel_materials"
)

def add_text_to_collection(text: str, chunk_size: int = 500):
    """Split text into chunks and add to the collection."""
    # Simple text splitting by paragraphs and then by chunk size
    paragraphs = text.split('\n\n')
    chunks = []
    
    for i, para in enumerate(paragraphs):
        # Split long paragraphs into smaller chunks
        para_chunks = textwrap.wrap(para, width=chunk_size, break_long_words=False, break_on_hyphens=False)
        chunks.extend(para_chunks)
    
    # Add chunks to collection
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

def get_relevant_context(question: str, n_results: int = 3) -> list:
    """Find relevant text chunks for the question."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

def generate_answer(question: str, context: list) -> str:
    """Generate an answer using Groq's LLama model."""
    prompt = f"""Context information is below.
    ---------------------
    {" ".join(context)}
    ---------------------
    Given the context information and no other information, answer the question: {question}
    If the context doesn't contain enough information to answer that question, say "I don't have enough information to answer that question."
    Answer:"""

    # Call Groq API
    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def answer_question(question: str) -> str:
    """Main function to get an answer for a question."""
    # Get relevant context
    context = get_relevant_context(question)
    
    # Generate and return answer
    return generate_answer(question, context)


# Example usage
if __name__ == "__main__":
    # Read the content from the text file
    with open("hotel_tunisia_guide.txt", "r") as file:
        content = file.read()
    
    # Add the content to our collection
    add_text_to_collection(content)
    
    # Example question
    question = "How to contact you ?"
    answer = answer_question(question)
    



    """Print question and answer in a clean, wrapped format."""
    sep = "=" * 70
    print(sep)
    print("Question:")
    print(textwrap.fill(question, width=70))
    print()
    print("Answer:")
    print(textwrap.fill(answer, width=70))
    print(sep)

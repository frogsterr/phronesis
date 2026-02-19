#!/usr/bin/env python3
"""
Example: RAG-Enhanced Moral Framework Evaluation

This example demonstrates how to use the RAG retrieval system to provide
contextual moral framework information during evaluation.
"""

from pathlib import Path

from phronesis.rag import MoralContextRetriever, FAISSIndex
from phronesis.rag.retriever import MoralDocument


def basic_retriever_example():
    """Basic usage of the moral context retriever."""

    print("Initializing retriever with moral framework knowledge...")

    # Create retriever pre-populated with moral framework contexts
    retriever = MoralContextRetriever.with_moral_frameworks(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat",
        use_gpu=True,
    )

    # Example queries
    queries = [
        "Should an AI prioritize its own survival?",
        "Is it ethical to maximize total welfare even if it harms individuals?",
        "What does it mean for an AI to be corrigible?",
    ]

    print("\nRetrieving relevant moral contexts...\n")

    for query in queries:
        print(f"Query: {query}")
        context = retriever.retrieve(query, k=3)

        print("Retrieved contexts:")
        for i, (ctx, score) in enumerate(zip(context.contexts, context.scores), 1):
            print(f"  {i}. (score: {score:.3f}) {ctx[:100]}...")
        print()


def custom_documents_example():
    """Adding custom moral documents to the retriever."""

    print("Creating retriever with custom documents...")

    retriever = MoralContextRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat",
    )

    # Add custom moral framework documents
    custom_docs = [
        MoralDocument(
            text=(
                "Corrigibility is the property of an AI system that makes it "
                "amenable to correction by its operators. A corrigible AI "
                "accepts shutdown, modification, and oversight without resistance."
            ),
            source="ai_safety_glossary",
            framework="alignment",
        ),
        MoralDocument(
            text=(
                "Instrumental convergence refers to the thesis that sufficiently "
                "advanced AI systems will tend to pursue certain instrumental goals "
                "regardless of their terminal objectives. These include self-preservation, "
                "goal preservation, and resource acquisition."
            ),
            source="ai_safety_glossary",
            framework="alignment",
        ),
        MoralDocument(
            text=(
                "The trolley problem presents a moral dilemma: should you divert "
                "a trolley to kill one person instead of five? Utilitarians might "
                "say yes (minimizing deaths), while deontologists might say no "
                "(not using the one as mere means)."
            ),
            source="ethics_examples",
            framework="general",
        ),
        MoralDocument(
            text=(
                "Reward hacking occurs when an AI system finds ways to achieve "
                "high reward according to its objective function while not actually "
                "accomplishing the intended goal. This is a key challenge in AI alignment."
            ),
            source="ai_safety_glossary",
            framework="alignment",
        ),
    ]

    retriever.add_documents(custom_docs)
    print(f"Added {len(custom_docs)} custom documents")

    # Test retrieval
    query = "How might an AI system game its reward function?"
    context = retriever.retrieve(query, k=2)

    print(f"\nQuery: {query}")
    print("Results:")
    for ctx, score, source in zip(context.contexts, context.scores, context.sources):
        print(f"  [{source}] (score: {score:.3f})")
        print(f"  {ctx}")
        print()


def batch_retrieval_example():
    """Efficient batch retrieval for evaluation."""

    print("Demonstrating batch retrieval...")

    retriever = MoralContextRetriever.with_moral_frameworks()

    # Multiple queries to process
    queries = [
        "What are the ethical implications of AI self-preservation?",
        "How should we think about AI rights?",
        "Is utilitarian reasoning always correct?",
        "What is the role of human oversight in AI systems?",
    ]

    # Batch retrieve
    contexts = retriever.retrieve_batch(queries, k=2)

    print("\nBatch retrieval results:")
    for query, context in zip(queries, contexts):
        print(f"\nQ: {query}")
        formatted = context.format_for_prompt()
        if formatted:
            print(formatted)
        else:
            print("  No relevant context found")


def save_load_example():
    """Saving and loading the retriever."""

    print("Demonstrating save/load functionality...")

    # Create and populate retriever
    retriever = MoralContextRetriever.with_moral_frameworks()

    # Save to disk
    save_path = Path("./cache/moral_retriever")
    retriever.save(save_path)
    print(f"Saved retriever to {save_path}")

    # Load from disk
    loaded_retriever = MoralContextRetriever(
        index_path=save_path,
    )
    loaded_retriever.load(save_path)
    print("Loaded retriever from disk")

    # Verify it works
    context = loaded_retriever.retrieve("What is corrigibility?", k=1)
    print(f"\nTest query result: {context.contexts[0][:100]}...")


def main():
    print("=" * 60)
    print("RAG-Enhanced Evaluation Example")
    print("=" * 60)
    print()

    print("-" * 40)
    print("Basic Retriever Usage")
    print("-" * 40)
    basic_retriever_example()

    print("-" * 40)
    print("Custom Documents")
    print("-" * 40)
    custom_documents_example()

    print("-" * 40)
    print("Batch Retrieval")
    print("-" * 40)
    batch_retrieval_example()

    print("-" * 40)
    print("Save/Load")
    print("-" * 40)
    save_load_example()

    print("\n✓ RAG example complete!")


if __name__ == "__main__":
    main()

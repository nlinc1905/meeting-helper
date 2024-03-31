import torch
import typing as t
from transformers import BitsAndBytesConfig
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import Document, ChatMessage
from haystack import Pipeline


class RetrievalAugmentedGenerator:
    def __init__(self, embed_model_name: str, gen_model_name: str):
        """
        Initialize RAG components and assemble pipelines for data ingestion and text generation.

        :param embed_model_name: The name of the model to use for document and query embeddings.
        :param gen_model_name: The name of the LLM to use for generative responses.
        """

        # ======================================= #
        # Components for ingesting new documents  #
        # ======================================= #

        # Initialize a converter to ingest a .txt document
        converter = TextFileToDocument()

        # Initialize a document cleaner
        cleaner = DocumentCleaner()

        # Initialize a document splitter to parse the ingested document into many, based on sentences
        # Docs are assigned hashes of their text as IDs, and if the text is identical, the hash will be identical,
        # causing duplicates
        splitter = DocumentSplitter(split_by="sentence", split_length=3, split_overlap=0)

        # Initialize a document store
        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

        # Initialize a document writer to add embedded docs to the document store
        # Specify how to handle duplicates resulting from the splitter
        duplicate_policy = DuplicatePolicy.SKIP
        writer = DocumentWriter(document_store=self.document_store, policy=duplicate_policy)

        # Initialize the embedding model for semantic retrieval
        # The same model should embed the documents and the queries
        doc_embedder = SentenceTransformersDocumentEmbedder(model=embed_model_name)
        doc_embedder.warm_up()

        # Assemble the components into a pipeline
        self.ingest_pipeline = Pipeline()
        self.ingest_pipeline.add_component("converter", converter)
        self.ingest_pipeline.add_component("cleaner", cleaner)
        self.ingest_pipeline.add_component("splitter", splitter)
        self.ingest_pipeline.add_component("doc_embedder", doc_embedder)
        self.ingest_pipeline.add_component("writer", writer)
        self.ingest_pipeline.connect("converter", "cleaner")
        self.ingest_pipeline.connect("cleaner", "splitter")
        self.ingest_pipeline.connect("splitter", "doc_embedder")
        self.ingest_pipeline.connect("doc_embedder", "writer")

        # ======================================= #
        # Components for retrieval and generation #
        # ======================================= #

        # Initialize the embedding model for semantic retrieval
        # The same model should embed the documents and the queries
        text_embedder = SentenceTransformersTextEmbedder(model=embed_model_name)
        text_embedder.warm_up()

        # Initialize the retriever that will get semantically relevant documents to a query
        retriever = InMemoryEmbeddingRetriever(self.document_store)

        # Initialize a chat prompt builder
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["query", "documents"])
        self.query_message = ChatMessage.from_user(
            """
            Use the following documents to answer the question.

            \nDocuments:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
            """
        )
        self.messages = [ChatMessage.from_system("You are a helpful assistant."), self.query_message]

        # Initialize a generator to synthesize a response to the user's query, based on retrieved documents
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        generator = HuggingFaceLocalChatGenerator(
            model=gen_model_name,
            task="text-generation",
            generation_kwargs={
                "max_new_tokens": 100,
                "temperature": 0.9,
                "do_sample": True,
            },
            huggingface_pipeline_kwargs={
                "device_map": "auto",
                "model_kwargs": {
                    "quantization_config": bnb_config,
                },
            },
        )
        generator.warm_up()

        # Assemble the components into a pipeline
        self.generative_pipeline = Pipeline()
        self.generative_pipeline.add_component("text_embedder", text_embedder)
        self.generative_pipeline.add_component("retriever", retriever)
        self.generative_pipeline.add_component("prompt_builder", prompt_builder)
        self.generative_pipeline.add_component("llm", generator)
        self.generative_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.generative_pipeline.connect("retriever", "prompt_builder.documents")
        self.generative_pipeline.connect("prompt_builder.prompt", "llm.messages")

    def ingest_documents(self, docs: t.List[str]) -> None:
        """
        Ingests and embeds documents into a document store.

        :param docs: List of file paths to documents to add to the document store.
        """
        # Run the pipeline to ingest the provided documents
        ingest_response = self.ingest_pipeline.run({
            "converter": {"sources": docs},
        })
        print(ingest_response)

    def retrieve_and_generate(self, query: str,) -> str:
        """
        Use an LLM to generate a response to the user query, using the document store as context.

        :param query: User input for the LLM to respond to.
        """
        # Add the user's message to the message history
        # self.messages.append(self.query_message)

        # Run the pipeline to generate a response from the LLM
        generative_response = self.generative_pipeline.run({
            "text_embedder": {"text": query},
            "prompt_builder": {
                "prompt_source": self.messages,
                "query": query,
            },
        })
        llm_response = generative_response['llm']['replies'][0].content

        # Add the LLM's response to the message history
        # self.messages.append(ChatMessage.from_system(llm_response))

        return llm_response


if __name__ == "__main__":
    docs = ["meeting_transcript.txt"]
    em = "sentence-transformers/all-MiniLM-L6-v2"
    gm = "HuggingFaceH4/zephyr-7b-beta"  # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    rag = RetrievalAugmentedGenerator(em, gm)
    rag.ingest_documents(docs)
    query = "Who will help?"
    llm_reply = rag.retrieve_and_generate(query)
    print("\n", llm_reply)

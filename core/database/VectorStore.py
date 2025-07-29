from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Document

class VectorStore:
    def __init__(self, collection_name):
        """
        Initialize the VectorStore with a Qdrant client.
        """
        self.client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        self.embbeding_model_name = "BAAI/bge-small-en"
        self.collection_name = collection_name
        self.collection_info = self.create_collection()

    def create_collection(self):
        """
        Get or create a collection in Qdrant.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.client.get_embedding_size(self.embbeding_model_name),
                    distance=Distance.COSINE),
            )
        return self.client.get_collection(self.collection_name)

    def add_documents(self, documents, metadata):
        """
        Add a document to the vector store.
        :param document: The document to add.
        :param metadata: Metadata associated with the document.
        """
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=[Document(text=doc, model=self.embbeding_model_name) for doc in documents],
            payload=[{"document": doc, **meta} for doc, meta in zip(documents, metadata)],
        )

    def search(self, text, limit=5):
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=Document(text=text,model=self.embbeding_model_name),
            limit=limit
        )
        return search_result

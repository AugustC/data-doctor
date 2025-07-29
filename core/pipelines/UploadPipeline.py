from core.database import VectorStore
import glob
from tqdm import tqdm

class UploadPipeline:
    """
    UploadPipeline is a class that handles the upload process of files.
    It is designed to chunk and upload the .md files
    """
    def __init__(self):
        self.vector_store = VectorStore("diagnosis")

    def run(self, dir):
        for file_path in tqdm(glob.glob(f"{dir}*.md")):
            self._upload_file(file_path)

    def _chunk_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sections = content.split('#')
            sections = [section.strip() for section in sections if section.strip()] # Removing whitespaces and empty strings
        return sections

    def _upload_file(self, file_path):
        file_chunks = self._chunk_file(file_path)
        metadata = [{"source" : file_path}]*len(file_chunks)
        self.vector_store.add_documents(file_chunks, metadata)

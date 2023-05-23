from vicuna_emb import VicunaEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
if __name__=="__main__":
    embeddings = VicunaEmbeddings(server_url=os.getenv("EMBEDDING_SERVER_URL"))
    text = "This is a test document"
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text, "This is also a document"])

    print(len(query_result))
    print(len(doc_result))

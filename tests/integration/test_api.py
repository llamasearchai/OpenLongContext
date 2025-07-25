import io
import pytest
from fastapi.testclient import TestClient
from openlongcontext.api import app

client = TestClient(app)

def test_upload_and_query_document():
    # Upload a document
    file_content = b"This is a test document for OpenLongContext. It contains information about long-context models."
    response = client.post(
        "/docs/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    doc_id = data["doc_id"]
    assert data["message"] == "Document uploaded and indexed."
    # Query the document
    query = {"doc_id": doc_id, "question": "What is this document about?"}
    response = client.post("/docs/query", json=query)
    assert response.status_code == 200
    result = response.json()
    assert "MOCK" in result["answer"]
    assert result["doc_id"] == doc_id
    assert "test document" in result["context"]
    # Get metadata
    response = client.get(f"/docs/{doc_id}")
    assert response.status_code == 200
    meta = response.json()
    assert meta["doc_id"] == doc_id
    assert meta["filename"] == "test.txt"
    assert meta["size"] == len(file_content)

def test_query_nonexistent_document():
    query = {"doc_id": "nonexistent", "question": "Does this exist?"}
    response = client.post("/docs/query", json=query)
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

def test_get_metadata_nonexistent_document():
    response = client.get("/docs/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found." 
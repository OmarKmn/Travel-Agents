from fastapi.testclient import TestClient

from app.llm import FakeLLM
from app.main import app, get_llm

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_uses_configured_llm():
    app.dependency_overrides[get_llm] = lambda: FakeLLM(responses=["Bonjour !"])
    try:
        response = client.post("/chat", json={"message": "Salut"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": "Bonjour !", "provider": "fake"}

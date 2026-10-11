
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

MACHINE_NORMALE = {
    "type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
}

MACHINE_RISQUE = {
    "type": "L",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1200,
    "torque": 75.0,
    "tool_wear": 220
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "XGBoost"


def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 5


def test_prediction_normale():
    response = client.post("/predict", json=MACHINE_NORMALE)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] == 0
    assert 0 <= result["probability"] <= 1
    assert result["status"] == "Fonctionnement normal"


def test_prediction_risque():
    response = client.post("/predict", json=MACHINE_RISQUE)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] == 1
    assert result["probability"] >= result["threshold"]


def test_type_invalide():
    data = MACHINE_NORMALE.copy()
    data["type"] = "Z"

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_vitesse_invalide():
    data = MACHINE_NORMALE.copy()
    data["rotational_speed"] = 0

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_couple_negatif():
    data = MACHINE_NORMALE.copy()
    data["torque"] = -10

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_champ_manquant():
    data = MACHINE_NORMALE.copy()
    del data["tool_wear"]

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_probabilite_valide():
    response = client.post("/predict", json=MACHINE_NORMALE)
    assert response.status_code == 200

    result = response.json()
    assert isinstance(result["probability"], (int, float))
    assert 0 <= result["probability"] <= 1
    assert result["threshold"] == 0.5


@pytest.mark.parametrize("machine_type", ["L", "M", "H"])
def test_types_valides(machine_type):
    data = MACHINE_NORMALE.copy()
    data["type"] = machine_type

    response = client.post("/predict", json=data)
    assert response.status_code == 200

from application import application

def test_home():
    client = application.test_client()
    assert client.get('/').status_code == 200

def test_predict(monkeypatch):
    # Mock the symbol that application.py actually uses
    class DummyPipeline:
        def predict(self, df):
            return [1]  # pretend it predicted "Churn"

    # 👉 Patch here (inside application module), not the src module
    monkeypatch.setattr(application, "PredictPipeline", lambda: DummyPipeline())

    client = application.test_client()
    data = {
        "age": "35",
        "gender": "Male",
        "tenure": "24",
        "monthly_charges": "70.5",
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No"
    }
    resp = client.post('/', data=data)
    assert resp.status_code == 200

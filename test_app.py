# test_app.py (minimal working)
import application as app_module  # import the module

def test_home():
    client = app_module.application.test_client()
    assert client.get('/').status_code == 200

def test_predict(monkeypatch):
    class DummyPipeline:
        def predict(self, df):
            return [1]  # simulate "Churn"

    # Patch the symbol where it is used: inside the module
    monkeypatch.setattr(app_module, "PredictPipeline", lambda: DummyPipeline())

    client = app_module.application.test_client()
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

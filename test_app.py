from application import application

def test_home():
    client = application.test_client()
    response = client.get('/')
    assert response.status_code == 200

def test_predict():
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
    response = client.post('/', data=data)
    assert response.status_code == 200

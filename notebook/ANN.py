import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from Random_search import random_search
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_path = r"/Users/ajithreddy/Downloads/CHURN_PRED_DEPLOY-master/notebook/Data/customer_churn_data.csv"
df1 = pd.read_csv(data_path)
df1.drop("CustomerID", axis=1, inplace=True)
df1["Gender"] = LabelEncoder().fit_transform(df1["Gender"])
df1["InternetService"] = df1["InternetService"].fillna("Unknown")
df1= pd.get_dummies(df1, columns=["InternetService"], drop_first=True)
df1= pd.get_dummies(df1, columns=["ContractType"], drop_first=True)
df1["TechSupport"] = LabelEncoder().fit_transform(df1["TechSupport"])
df1["Churn"] = df1["Churn"].map({"Yes": 1, "No": 0})
bool_cols = df1.select_dtypes(include='bool').columns
df1[bool_cols] = df1[bool_cols].astype(int)
scaler = StandardScaler()
scale_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
df1[scale_cols] = scaler.fit_transform(df1[scale_cols])
print(df1.columns)
x=df1[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
       'TechSupport', 'InternetService_Fiber Optic',
       'InternetService_Unknown', 'ContractType_One-Year',
       'ContractType_Two-Year']]
y=df1[["Churn"]]

X = x.values.astype(np.float32)
Y = y.values.astype(np.float32)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, train_size=0.75, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_train = torch.tensor(X_train_balanced, dtype=torch.float32)
y_train = torch.tensor(y_train_balanced, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)
class ANN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def model_init_fn(params):
    return {
        'input_dim': X_train.shape[1],
        'hidden1': params['hidden1'],
        'hidden2': params['hidden2'],
        'dropout': params['dropout']
    }

param_dist = {
    'lr': [1e-3, 1e-4],
    'batch_size': [32, 64],
    'hidden1': [64, 128],
    'hidden2': [32, 64],
    'dropout': [0.2, 0.5],
    'weight_decay': [0, 1e-4]
}

best_params = random_search(
    model_class=ANN,
    model_init_fn=model_init_fn,
    param_dist=param_dist,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    device=device,
    n_iter=20
)


# Retrain final model with EarlyStopping on train+val, test on test set
print("\nRetraining final model with best hyperparameters...")
input_dim = X_train.shape[1]
final_model = ANN(input_dim, best_params['hidden1'], best_params['hidden2'], best_params['dropout']).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=best_params['batch_size'])
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

# EarlyStopping for final model
best_val_loss = float('inf')
patience = 5
counter = 0
num_epochs = 50
for epoch in range(num_epochs):
    final_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = final_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch.squeeze())
        loss.backward()
        optimizer.step()
    # Validation
    final_model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = final_model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.squeeze())
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Optionally: torch.save(final_model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# Final Evaluation on Test Set
all_preds, all_labels = [], []
final_model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = final_model(X_batch).squeeze()
        preds = (outputs >= 0.5).float().cpu().numpy()
        labels = y_batch.squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
print("\nFinal Model Performance on Test Set:")
print(classification_report(all_labels, all_preds, target_names=["No Churn", "Churn"]))

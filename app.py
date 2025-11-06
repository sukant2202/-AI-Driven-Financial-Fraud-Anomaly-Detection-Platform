from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

app = Flask(__name__)

# ✅ Load scaler
scaler = joblib.load("scaler.pkl")

# ✅ Define same MLP model used during training (hidden = 64)
class FraudMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ✅ Load trained model (make sure file exists)
model = FraudMLP()
model.load_state_dict(torch.load("fraud_model_state.pth", map_location="cpu"))
model.eval()

# ✅ Home route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        time = float(request.form['time'])
        v1 = float(request.form['v1'])
        v2 = float(request.form['v2'])
        amount = float(request.form['amount'])

        # Prepare and scale data (must match training order)
        X = [[time, v1, v2, amount]]
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            pred = torch.argmax(model(X_tensor), dim=1).item()

        result = "🚨 Fraudulent Transaction" if pred == 1 else "✅ Legitimate Transaction"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

# ✅ Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

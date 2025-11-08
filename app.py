from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import sqlite3
import os
from contextlib import contextmanager

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production-12345')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT UNIQUE NOT NULL,
                         email TEXT UNIQUE NOT NULL,
                         password TEXT NOT NULL)''')

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id, self.username, self.email = id, username, email

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
        return self.fc3(x)

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

def load_model_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    model = FraudMLP()
    model.load_state_dict(torch.load(os.path.join(base_path, "fraud_model_state.pth"), map_location="cpu"))
    model.eval()
    return scaler, model

scaler, model = load_model_assets()

@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        user_data = conn.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,)).fetchone()
        return User(*user_data) if user_data else None

@app.route('/')
def index():
    return redirect(url_for('home') if current_user.is_authenticated else url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        if not username or not password:
            flash('Please fill in all fields', 'error')
        else:
            with get_db() as conn:
                user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
                if user and check_password_hash(user['password'], password):
                    login_user(User(user['id'], user['username'], user['email']))
                    flash('Login successful!', 'success')
                    return redirect(url_for('home'))
                flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username, email = request.form.get('username'), request.form.get('email')
        password, confirm_password = request.form.get('password'), request.form.get('confirm_password')
        if not all([username, email, password]):
            flash('Please fill in all fields', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
        else:
            with get_db() as conn:
                if conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email)).fetchone():
                    flash('Username or email already exists', 'error')
                else:
                    conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                               (username, email, generate_password_hash(password)))
                    flash('Account created successfully! Please log in.', 'success')
                    return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('index.html', username=current_user.username)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        X = [[float(request.form[k]) for k in ['time', 'v1', 'v2', 'amount']]]
        X_tensor = torch.tensor(scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            probabilities = F.softmax(model(X_tensor), dim=1)[0]
            pred = probabilities.argmax().item()
        fraud_prob, legit_prob = probabilities[1].item() * 100, probabilities[0].item() * 100
        return render_template('index.html', 
                             prediction_text="🚨 Fraudulent Transaction" if pred == 1 else "✅ Legitimate Transaction",
                             prediction_prob=probabilities[pred].item(),
                             fraud_prob=fraud_prob,
                             legit_prob=legit_prob,
                             is_fraud=(pred == 1),
                             username=current_user.username)
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"⚠️ Error: {str(e)}",
                             username=current_user.username)

init_db()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import sqlite3
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production-12345')  # Secret key for sessions

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Database setup - Use persistent path for production (Render) or local for development
if os.environ.get('RENDER'):
    # Render provides persistent disk storage
    DATABASE = os.path.join(os.environ.get('RENDER_DISK_PATH', '/opt/render/project/src'), 'users.db')
elif os.environ.get('VERCEL') == '1':
    # Vercel uses /tmp (ephemeral)
    DATABASE = '/tmp/users.db'
else:
    # Local development
    DATABASE = 'users.db'

def init_db():
    """Initialize the database with users table"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ✅ Load scaler and model (handle paths for Vercel)
def load_model_assets():
    """Load model and scaler - handles both local and Vercel paths"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(base_path, "scaler.pkl")
    model_path = os.path.join(base_path, "fraud_model_state.pth")
    
    scaler = joblib.load(scaler_path)
    
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
    
    model = FraudMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    return scaler, model

scaler, model = load_model_assets()

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            user_obj = User(user['id'], user['username'], user['email'])
            login_user(user_obj)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('Please fill in all fields', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('signup.html')
        
        conn = get_db_connection()
        # Check if username or email already exists
        existing_user = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists', 'error')
            conn.close()
            return render_template('signup.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
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
        # Get user input
        time = float(request.form['time'])
        v1 = float(request.form['v1'])
        v2 = float(request.form['v2'])
        amount = float(request.form['amount'])

        # Prepare and scale data (must match training order)
        X = [[time, v1, v2, amount]]
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Model prediction with probabilities
        with torch.no_grad():
            logits = model(X_tensor)
            probabilities = F.softmax(logits, dim=1)
            pred = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred].item() * 100

        result = "🚨 Fraudulent Transaction" if pred == 1 else "✅ Legitimate Transaction"
        fraud_prob = probabilities[0][1].item() * 100
        legit_prob = probabilities[0][0].item() * 100
        
        return render_template('index.html', 
                             prediction_text=result,
                             prediction_prob=confidence / 100,
                             fraud_prob=fraud_prob,
                             legit_prob=legit_prob,
                             is_fraud=(pred == 1),
                             username=current_user.username)

    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"⚠️ Error: {str(e)}",
                             username=current_user.username)

# Initialize database on startup
init_db()

# For local development
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

import sqlite3
import pandas as pd
from werkzeug.security import generate_password_hash
"""Database operations for diabetes prediction system
Handles data storage, user authentication, and prediction history"""

def init_db():
    conn = sqlite3.connect('diabetes.db') #Opens a connection to the SQLite database file
    cursor = conn.cursor() #Creates a cursor object to execute SQL commands

    # Table 1: Original dataset (unchanged)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS diabetes_data (
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        bmi REAL,
        diabetes_pedigree REAL,
        age INTEGER,
        outcome INTEGER
    )
    ''')

    # Table 2: Users (NEW for authentication)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT (datetime('now','localtime'))
    )
    ''')

    # Table 3: User submissions (...)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        bmi REAL,
        diabetes_pedigree REAL,
        age INTEGER,
        predicted_outcome INTEGER,
        timestamp DATETIME DEFAULT (datetime('now','localtime')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    # Load diabetes.csv if empty (unchanged)
    if pd.read_sql('SELECT COUNT(*) FROM diabetes_data', conn).iloc[0, 0] == 0:
        df = pd.read_csv('diabetes.csv') #Reads the diabetes.csv file into a pandas DataFrame
        df.to_sql('diabetes_data', conn, if_exists='replace', index=False) #Writes the DataFrame to a SQL table
        #(index=flase) prevents pandas from writting row numbers as a column
    conn.commit() #saves all changes to database
    conn.close()


# MODIFIED to include user_id
def save_prediction(user_id, user_data, prediction):
    """
        Save a prediction result to the database
        Args:
            user_id: ID of the user making the prediction
            user_data: Dictionary of input features
            prediction: Model output (diabetes score)
        """
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO user_predictions 
    (user_id, pregnancies, glucose, blood_pressure, skin_thickness, 
     insulin, bmi, diabetes_pedigree, age, predicted_outcome)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        user_data['pregnancies'],
        user_data['glucose'],
        user_data['bloodPressure'],
        user_data['skinThickness'],
        user_data['insulin'],
        user_data['bmi'],
        user_data['diabetesPedigree'],
        user_data['age'],
        prediction
    ))
    conn.commit()
    conn.close()


# MODIFIED to filter by user
def get_prediction_history(user_id=None):
    """
        Retrieve prediction history from database
        Args:
            user_id: If provided, filters to specific user
        Returns:
            DataFrame of historical predictions
        """
    conn = sqlite3.connect('diabetes.db')

    if user_id:
        query = 'SELECT * FROM user_predictions WHERE user_id = ? ORDER BY timestamp DESC'
        df = pd.read_sql(query, conn, params=(user_id,))
    else:
        query = 'SELECT * FROM user_predictions ORDER BY timestamp DESC'
        df = pd.read_sql(query, conn)

    conn.close()
    return df


# NEW authentication functions
def create_user(email, password, name):
    """
       Register a new user with hashed password
       Args:
           email: User's email (unique identifier)
           password: Plaintext password (will be hashed)
           name: Display name
       Returns:
           New user ID
       Raises:
           ValueError if email already exists
       """
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    try:
        hashed_pw = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (email, password, name)
            VALUES (?, ?, ?)
        ''', (email, hashed_pw, name))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("Email already exists")
    finally:
        conn.close()


def get_user_by_email(email):
    """
       Retrieve user by email address
       Args:
           email: Email to search for
       Returns:
           User dictionary if found, None otherwise
       """
    conn = sqlite3.connect('diabetes.db')
    conn.row_factory = sqlite3.Row  # Enable column access by name

    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))

    user = cursor.fetchone()

    conn.close()
    return dict(user) if user else None #Converts the SQLite Row object to a Python dictionary if found
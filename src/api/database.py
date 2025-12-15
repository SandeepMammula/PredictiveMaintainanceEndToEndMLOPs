import sqlite3
import json
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    def __init__(self, db_path='data/predictions.db'):
        Path(db_path).parent.mkdir(exist_ok=True)
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                features TEXT NOT NULL,
                predicted_rul REAL NOT NULL,
                model_version TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, features, predicted_rul, model_version):
        """Log a prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, features, predicted_rul, model_version)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(features),
            predicted_rul,
            model_version
        ))
        
        conn.commit()
        conn.close()
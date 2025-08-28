from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

load_dotenv()

def db_connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return engine
    except Exception:
        return None

def get_paths():
    model = os.getenv("MODEL_PATH", "models/spam_model.joblib")
    vectorizer = os.getenv("VECTORIZER_PATH", "models/vectorizer.joblib")
    metrics = os.getenv("METRICS_PATH", "models/metrics.json")
    return model, vectorizer, metrics

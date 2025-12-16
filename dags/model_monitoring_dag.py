from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_monitoring',
    default_args=default_args,
    description='Monitor model performance and detect drift',
    schedule_interval='@daily',  # Run daily
    catchup=False,
)

def check_api_health():
    """Check if API is healthy."""
    response = requests.get('http://34.41.117.141/health')
    if response.status_code != 200:
        raise Exception("API is down!")
    print("✓ API is healthy")

def collect_predictions():
    """Collect recent predictions for drift detection."""
    # Placeholder - in production, query database
    print("✓ Collected predictions from last 24 hours")
    # TODO: Query SQLite/PostgreSQL for predictions
    return True

def detect_drift():
    """Run drift detection using Evidently."""
    import sys
    sys.path.append('C:/Project/PredictiveMaintainanceEndToEndMLOPs')
    from src.monitoring.drift_detection import detect_drift as run_detection
    
    drift_detected, drift_share = run_detection()
    
    if drift_detected:
        print(f"⚠️ DRIFT DETECTED ({drift_share:.2%}) - Triggering retraining")
        return "retrain"
    else:
        print(f"✓ No drift detected (drift share: {drift_share:.2%})")
        return "ok"

def trigger_retraining():
    """Trigger model retraining pipeline."""
    print("🔄 Starting model retraining...")
    # TODO: Call retraining script or trigger separate DAG
    return True

# Define tasks
check_health = PythonOperator(
    task_id='check_api_health',
    python_callable=check_api_health,
    dag=dag,
)

collect_data = PythonOperator(
    task_id='collect_predictions',
    python_callable=collect_predictions,
    dag=dag,
)

drift_check = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    dag=dag,
)

retrain = PythonOperator(
    task_id='trigger_retraining',
    python_callable=trigger_retraining,
    dag=dag,
)

# Define workflow
check_health >> collect_data >> drift_check >> retrain
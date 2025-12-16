from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2025, 12, 1),
    'retries': 1,
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Retrain and deploy model',
    schedule_interval=None,  # Manual trigger or called by drift detection
    catchup=False,
)

# Task 1: Pull latest code
pull_code = BashOperator(
    task_id='pull_latest_code',
    bash_command='echo "Pulling latest code from GitHub"',
    dag=dag,
)

# Task 2: Retrain model
retrain_model = BashOperator(
    task_id='retrain_model',
    bash_command='echo "Running training script: python src/models/train.py"',
    dag=dag,
)

# Task 3: Evaluate model
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='echo "Evaluating model: python src/models/evaluate.py"',
    dag=dag,
)

# Task 4: Register model
register_model = BashOperator(
    task_id='register_model',
    bash_command='echo "Registering model in MLflow"',
    dag=dag,
)

# Task 5: Deploy to production
deploy_model = BashOperator(
    task_id='deploy_to_production',
    bash_command='echo "Deploying via kubectl rollout restart"',
    dag=dag,
)

# Workflow
pull_code >> retrain_model >> evaluate_model >> register_model >> deploy_model
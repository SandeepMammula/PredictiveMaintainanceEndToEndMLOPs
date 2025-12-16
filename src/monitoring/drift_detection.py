import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pathlib import Path

def detect_drift():
    """Detect data drift between training and production data."""
    
    print("Loading training data (baseline)...")
    train_data = pd.read_csv('data/processed/train_FD001_processed.csv')
    
    # Get feature columns (exclude target and metadata)
    feature_cols = [col for col in train_data.columns if col not in ['unit', 'cycle', 'RUL']]
    
    # Use last 1000 samples as baseline
    baseline = train_data[feature_cols].tail(1000)
    
    print("Simulating production data (current)...")
    # In production, this would come from predictions database
    # For now, simulate by adding noise to test data
    test_data = pd.read_csv('data/processed/test_FD001_processed.csv')
    current = test_data[feature_cols].head(1000)
    
    # Add some noise to simulate drift (for testing)
    # current = current + current * 0.1  # 10% increase - simulates drift
    
    print("Running drift detection...")
    # Create Evidently report
    report = Report(metrics=[DataDriftPreset()])
    
    report.run(reference_data=baseline, current_data=current)
    
    # Save report
    Path('reports/drift').mkdir(parents=True, exist_ok=True)
    report.save_html('reports/drift/drift_report.html')
    
    # Get drift results
    results = report.as_dict()
    
    # Check if drift detected
    drift_share = results['metrics'][0]['result']['drift_share']
    dataset_drift = results['metrics'][0]['result']['dataset_drift']
    
    print(f"\n{'='*60}")
    print(f"DRIFT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset Drift: {dataset_drift}")
    print(f"Drift Share: {drift_share:.2%}")
    print(f"Report saved to: reports/drift/drift_report.html")
    print(f"{'='*60}\n")
    
    # Save results
    with open('reports/drift/drift_results.json', 'w') as f:
        json.dump({
            'drift_detected': dataset_drift,
            'drift_share': drift_share,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    return dataset_drift, drift_share

if __name__ == "__main__":
    drift_detected, drift_share = detect_drift()
    
    if drift_detected:
        print("⚠️  DRIFT DETECTED - Model retraining recommended!")
    else:
        print("✓ No significant drift detected")
import pandas as pd
from pathlib import Path

def validate_schema(data, is_train=True):
    """
    Validate that required columns exist.
    
    Args:
        data: DataFrame to validate
        is_train: Whether this is training or test data
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required columns
    required_cols = ['unit', 'cycle', 'RUL']
    
    # Check required columns exist
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check we have sensor columns
    sensor_cols = [col for col in data.columns if 'sensor' in col]
    if len(sensor_cols) == 0:
        errors.append("No sensor columns found")
    
    return errors


def validate_data_types(data):
    """
    Validate data types are correct.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Unit should be integer
    if 'unit' in data.columns and not pd.api.types.is_integer_dtype(data['unit']):
        errors.append("Column 'unit' should be integer type")
    
    # Cycle should be integer
    if 'cycle' in data.columns and not pd.api.types.is_integer_dtype(data['cycle']):
        errors.append("Column 'cycle' should be integer type")
    
    # RUL should be numeric
    if 'RUL' in data.columns and not pd.api.types.is_numeric_dtype(data['RUL']):
        errors.append("Column 'RUL' should be numeric type")
    
    return errors


def validate_value_ranges(data):
    """
    Validate values are within expected ranges.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # RUL should be non-negative
    if 'RUL' in data.columns:
        if (data['RUL'] < 0).any():
            errors.append("RUL contains negative values")
        
        if (data['RUL'] > 500).any():
            errors.append("RUL contains suspiciously high values (>500)")
    
    # Cycle should be positive
    if 'cycle' in data.columns:
        if (data['cycle'] <= 0).any():
            errors.append("Cycle contains non-positive values")
    
    # Unit should be positive
    if 'unit' in data.columns:
        if (data['unit'] <= 0).any():
            errors.append("Unit contains non-positive values")
    
    # Check sensor values are reasonable (not all zeros or constant)
    sensor_cols = [col for col in data.columns if 'sensor' in col and 'rolling' not in col and 'lag' not in col]
    for sensor in sensor_cols:
        if data[sensor].std() == 0:
            errors.append(f"{sensor} has zero variance (constant values)")
    
    return errors


def validate_missing_values(data, max_missing_pct=5.0):
    """
    Check for missing values.
    
    Args:
        data: DataFrame to validate
        max_missing_pct: Maximum allowed percentage of missing values
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check each column for missing values
    for col in data.columns:
        missing_pct = (data[col].isnull().sum() / len(data)) * 100
        
        if missing_pct > max_missing_pct:
            errors.append(f"{col} has {missing_pct:.2f}% missing values (threshold: {max_missing_pct}%)")
    
    return errors


def validate_data_consistency(data):
    """
    Validate logical consistency in the data.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check cycles are sequential per unit
    if 'unit' in data.columns and 'cycle' in data.columns:
        for unit_id in data['unit'].unique():
            unit_data = data[data['unit'] == unit_id].sort_values('cycle')
            cycles = unit_data['cycle'].values
            
            # Check if cycles start from 1
            if cycles[0] != 1:
                errors.append(f"Unit {unit_id}: cycles don't start from 1 (starts at {cycles[0]})")
            
            # Check for gaps in cycles
            expected_cycles = range(1, len(cycles) + 1)
            if not all(cycles == list(expected_cycles)):
                errors.append(f"Unit {unit_id}: has gaps or non-sequential cycles")
    
    # Check RUL decreases monotonically per unit
    if 'unit' in data.columns and 'RUL' in data.columns and 'cycle' in data.columns:
        for unit_id in data['unit'].unique():
            unit_data = data[data['unit'] == unit_id].sort_values('cycle')
            rul_values = unit_data['RUL'].values
            
            # RUL should decrease as cycles increase
            if not all(rul_values[i] >= rul_values[i+1] for i in range(len(rul_values)-1)):
                errors.append(f"Unit {unit_id}: RUL does not decrease monotonically")
    
    return errors


def validate_statistical_properties(data):
    """
    Check statistical properties of the data.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        List of validation warnings (not hard errors)
    """
    warnings = []
    
    # Check for extreme outliers in sensor data
    sensor_cols = [col for col in data.columns if 'sensor' in col and 'rolling' not in col and 'lag' not in col]
    
    for sensor in sensor_cols:
        mean = data[sensor].mean()
        std = data[sensor].std()
        
        # Values beyond 5 standard deviations
        outliers = ((data[sensor] - mean).abs() > 5 * std).sum()
        outlier_pct = (outliers / len(data)) * 100
        
        if outlier_pct > 1.0:
            warnings.append(f"{sensor} has {outlier_pct:.2f}% extreme outliers (>5 std)")
    
    return warnings


def validate_file(filepath, is_train=True):
    """
    Run all validation checks on a data file.
    
    Args:
        filepath: Path to CSV file to validate
        is_train: Whether this is training data
    
    Returns:
        Dictionary with validation results
    """
    print(f"\nValidating: {filepath}")
    print("=" * 60)
    
    # Load data
    try:
        data = pd.read_csv(filepath)
        print(f"✓ File loaded successfully")
        print(f"  Shape: {data.shape}")
    except Exception as e:
        return {
            'status': 'FAILED',
            'errors': [f"Failed to load file: {str(e)}"],
            'warnings': []
        }
    
    # Run all validations
    all_errors = []
    all_warnings = []
    
    # Schema validation
    errors = validate_schema(data, is_train)
    all_errors.extend(errors)
    if errors:
        print(f"✗ Schema validation: {len(errors)} errors")
    else:
        print(f"✓ Schema validation passed")
    
    # Data type validation
    errors = validate_data_types(data)
    all_errors.extend(errors)
    if errors:
        print(f"✗ Data type validation: {len(errors)} errors")
    else:
        print(f"✓ Data type validation passed")
    
    # Value range validation
    errors = validate_value_ranges(data)
    all_errors.extend(errors)
    if errors:
        print(f"✗ Value range validation: {len(errors)} errors")
    else:
        print(f"✓ Value range validation passed")
    
    # Missing value validation
    errors = validate_missing_values(data, max_missing_pct=5.0)
    all_errors.extend(errors)
    if errors:
        print(f"✗ Missing value validation: {len(errors)} errors")
    else:
        print(f"✓ Missing value validation passed")
    
    # Data consistency validation
    errors = validate_data_consistency(data)
    all_errors.extend(errors)
    if errors:
        print(f"✗ Data consistency validation: {len(errors)} errors")
    else:
        print(f"✓ Data consistency validation passed")
    
    # Statistical validation (warnings only)
    warnings = validate_statistical_properties(data)
    all_warnings.extend(warnings)
    if warnings:
        print(f"⚠ Statistical validation: {len(warnings)} warnings")
        print(warnings)
    else:
        print(f"✓ Statistical validation passed")
    
    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print(f"❌ VALIDATION FAILED: {len(all_errors)} errors")
        print("\nErrors:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
    else:
        print(f"✅ VALIDATION PASSED")
    
    if all_warnings:
        print(f"\n⚠️  {len(all_warnings)} warnings:")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
    
    return {
        'status': 'PASSED' if not all_errors else 'FAILED',
        'errors': all_errors,
        'warnings': all_warnings,
        'shape': data.shape,
        'columns': list(data.columns)
    }


def save_validation_report(results, output_path):
    """
    Save validation results to a file.
    
    Args:
        results: Dictionary of validation results
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("DATA VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for filepath, result in results.items():
            f.write(f"File: {filepath}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Shape: {result.get('shape', 'N/A')}\n")
            f.write(f"Errors: {len(result['errors'])}\n")
            f.write(f"Warnings: {len(result['warnings'])}\n\n")
            
            if result['errors']:
                f.write("Errors:\n")
                for error in result['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if result['warnings']:
                f.write("Warnings:\n")
                for warning in result['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            f.write("-" * 60 + "\n\n")
    
    print(f"\n📄 Validation report saved to: {output_path}")


if __name__ == "__main__":
    # Create output directory for reports
    Path("reports").mkdir(exist_ok=True)
    
    # Files to validate
    files_to_validate = [
        ('data/processed/train_FD001_processed.csv', True),
        ('data/processed/test_FD001_processed.csv', False)
    ]
    
    # Run validation on all files
    all_results = {}
    
    for filepath, is_train in files_to_validate:
        if Path(filepath).exists():
            result = validate_file(filepath, is_train)
            all_results[filepath] = result
        else:
            print(f"\n⚠️  File not found: {filepath}")
            all_results[filepath] = {
                'status': 'FAILED',
                'errors': ['File not found'],
                'warnings': []
            }
    
    # Save validation report
    save_validation_report(all_results, 'reports/data_validation_report.txt')
    
    # Check if all validations passed
    all_passed = all(r['status'] == 'PASSED' for r in all_results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATIONS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME VALIDATIONS FAILED - Check report for details")
        print("=" * 60)
        exit(1)
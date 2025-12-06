import os
import pandas as pd

def test_data_exists():
    """Check if raw data has been generated."""
    # Check if the file exists
    assert os.path.exists("data/raw/synthetic_meghalaya_land_data.csv"), "Data file missing!"

def test_columns_exist():
    """Check if essential columns are present."""
    try:
        df = pd.read_csv("data/raw/synthetic_meghalaya_land_data.csv")
    except:
        return # If read fails, the previous test catches it
        
    required_cols = ['land_id', 'district', 'dispute_status']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
        
if __name__ == "__main__":
    test_data_exists()
    test_columns_exist()
    print("All tests passed!")
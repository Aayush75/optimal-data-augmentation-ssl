"""
Quick test to verify JSON serialization fix works.
"""

import numpy as np
import json

# Test the conversion function
def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Test with various numpy types
test_data = {
    'bool_': np.bool_(True),
    'int64': np.int64(42),
    'float64': np.float64(3.14),
    'array': np.array([1, 2, 3]),
    'nested': {
        'value': np.float32(2.71),
        'flag': np.bool_(False),
    }
}

print("Testing JSON serialization with NumPy types...")
print("Original data types:")
for k, v in test_data.items():
    print("  {}: {} ({})".format(k, v, type(v)))

converted = convert_to_serializable(test_data)
print("\nConverted data types:")
for k, v in converted.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            print("  {}.{}: {} ({})".format(k, k2, v2, type(v2)))
    else:
        print("  {}: {} ({})".format(k, v, type(v)))

# Try to serialize
try:
    json_str = json.dumps(converted, indent=2)
    print("\nJSON serialization successful!")
    print(json_str)
except Exception as e:
    print("\nJSON serialization FAILED: {}".format(e))

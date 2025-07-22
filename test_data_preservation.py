#!/usr/bin/env python3
"""Test that get_data is preserved correctly in JSON conversion."""

import cvxpy as cp
from cvxpy_to_json import problem_to_json, json_to_problem
import json

# Create a problem with Sum which has get_data() = [None, False]
x = cp.Variable(2, name="test_var")
problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

# Convert to JSON and check the raw JSON
json_str = problem_to_json(problem)
data_dict = json.loads(json_str)

print("Original problem:")
print(problem)

print("\nJSON representation:")
print(json.dumps(data_dict, indent=2))

# Find the Sum expression in the JSON
sum_expr = data_dict["objective"]["expression"]
print(f"\nSum expression data: {sum_expr.get('data', 'No data field')}")

# Reconstruct and verify it works
reconstructed = json_to_problem(json_str)
print(f"\nReconstructed problem:")
print(reconstructed)

# Verify they solve the same
problem.solve(verbose=False)
reconstructed.solve(verbose=False)

print(f"\nOriginal value: {problem.value}")
print(f"Reconstructed value: {reconstructed.value}")
print(f"Values match: {abs(problem.value - reconstructed.value) < 1e-10}")
#!/usr/bin/env python3
"""Test the CVXPY to JSON converter with various problem types."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem, save_problem_json

def test_simple_problem():
    """Test a simple linear problem."""
    print("Testing simple linear problem...")
    
    # Simple linear problem
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    
    objective = cp.Minimize(x + y)
    constraints = [x >= 0, y >= 0, x + y <= 1]
    
    problem = cp.Problem(objective, constraints)
    print(f"Original: {problem}")
    
    # Convert and reconstruct
    json_str = problem_to_json(problem)
    save_problem_json(problem, "simple_linear_problem.json")
    reconstructed = json_to_problem(json_str)
    print(f"Reconstructed: {reconstructed}")
    
    # Test solving both
    problem.solve(verbose=False)
    reconstructed.solve(verbose=False)
    
    print(f"Original value: {problem.value}")
    print(f"Reconstructed value: {reconstructed.value}")
    print()

def test_quadratic_problem():
    """Test a quadratic problem."""
    print("Testing quadratic problem...")
    
    x = cp.Variable(2, name="x_vec")
    
    objective = cp.Minimize(cp.sum_squares(x))
    constraints = [x >= 0, cp.sum(x) == 1]
    
    problem = cp.Problem(objective, constraints)
    print(f"Original: {problem}")
    
    json_str = problem_to_json(problem)
    save_problem_json(problem, "quadratic_problem.json")
    reconstructed = json_to_problem(json_str)
    print(f"Reconstructed: {reconstructed}")
    
    problem.solve(verbose=False)
    reconstructed.solve(verbose=False)
    
    print(f"Original value: {problem.value}")
    print(f"Reconstructed value: {reconstructed.value}")
    print()

def test_variable_names():
    """Test that variable names are preserved."""
    print("Testing variable name preservation...")
    
    profit = cp.Variable(name="profit")
    cost = cp.Variable(name="cost") 
    revenue = cp.Variable(3, name="revenue_by_product")
    
    objective = cp.Maximize(profit)
    constraints = [
        profit == revenue.sum() - cost,
        cost >= 10,
        revenue >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    
    # Check original variable names
    orig_vars = {v.name(): v for v in problem.variables()}
    print(f"Original variables: {list(orig_vars.keys())}")
    
    json_str = problem_to_json(problem)
    save_problem_json(problem, "variable_names_problem.json")
    reconstructed = json_to_problem(json_str)
    
    # Check reconstructed variable names  
    recon_vars = {v.name(): v for v in reconstructed.variables()}
    print(f"Reconstructed variables: {list(recon_vars.keys())}")
    
    # Verify names match
    assert orig_vars.keys() == recon_vars.keys(), "Variable names don't match!"
    print("✓ Variable names preserved correctly!")
    print()

def test_with_parameters():
    """Test problems with parameters."""
    print("Testing problem with parameters...")
    
    x = cp.Variable(name="x")
    param_a = cp.Parameter(name="param_a")  # No values initially
    param_b = cp.Parameter(name="param_b")
    
    objective = cp.Minimize(param_a * x + param_b)
    constraints = [x >= 1]
    
    problem = cp.Problem(objective, constraints)
    print(f"Original: {problem}")
    
    json_str = problem_to_json(problem)
    save_problem_json(problem, "parameter_problem.json")
    reconstructed = json_to_problem(json_str)
    print(f"Reconstructed: {reconstructed}")
    
    # Set parameter values after reconstruction (realistic usage pattern)
    param_a.value = 2.0
    param_b.value = 3.0
    
    for param in reconstructed.parameters():
        if param.name() == "param_a":
            param.value = 2.0
        elif param.name() == "param_b":
            param.value = 3.0
    
    problem.solve(verbose=False)
    reconstructed.solve(verbose=False)
    
    print(f"Original value: {problem.value}")
    print(f"Reconstructed value: {reconstructed.value}")
    print()

if __name__ == "__main__":
    print("CVXPY JSON Converter Tests")
    print("=" * 40)
    
    try:
        test_simple_problem()
        test_quadratic_problem()
        test_variable_names()
        test_with_parameters()
        print("All tests completed!")
        print("\nJSON files created:")
        print("- simple_linear_problem.json")
        print("- quadratic_problem.json")
        print("- variable_names_problem.json")
        print("- parameter_problem.json")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
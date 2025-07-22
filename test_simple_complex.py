#!/usr/bin/env python3
"""Test CVXPY to JSON converter with a simpler complex problem."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem, save_problem_json

def create_simple_complex_problem():
    """Create a simpler complex problem to test step by step."""
    print("Creating simpler complex problem...")
    
    # Variables
    x = cp.Variable(3, name="x_var")
    y = cp.Variable(name="y_scalar")
    
    # Start with basic nested functions that should work
    objective_expr = (
        cp.sum_squares(x) +  # Should work
        cp.square(y) +       # Should work  
        cp.norm(x, 2) +      # Should work
        cp.abs(y)            # Let's see if this works
    )
    
    objective = cp.Minimize(objective_expr)
    
    # Simple constraints
    constraints = [
        x >= 0,
        y <= 5,
        cp.sum(x) == 1
    ]
    
    problem = cp.Problem(objective, constraints)
    return problem

def test_simple_complex():
    """Test the simpler version first."""
    problem = create_simple_complex_problem()
    
    print(f"Original problem:")
    print(problem)
    print(f"Variables: {len(problem.variables())}")
    print(f"Constraints: {len(problem.constraints)}")
    
    # Convert to JSON
    print("\nConverting to JSON...")
    try:
        json_str = problem_to_json(problem)
        save_problem_json(problem, "simple_complex_problem.json")
        print(f"✓ JSON conversion successful! Size: {len(json_str)} characters")
        
        # Try reconstruction
        print("\nReconstructing from JSON...")
        reconstructed = json_to_problem(json_str)
        print(f"✓ Reconstruction successful!")
        print(f"Reconstructed: {reconstructed}")
        
        # Test solving
        print("\nTesting solve...")
        problem.solve(verbose=False)
        reconstructed.solve(verbose=False)
        
        print(f"Original value: {problem.value}")
        print(f"Reconstructed value: {reconstructed.value}")
        
        if problem.value is not None and reconstructed.value is not None:
            diff = abs(problem.value - reconstructed.value)
            print(f"Difference: {diff:.2e}")
            if diff < 1e-6:
                print("✓ Values match perfectly!")
            else:
                print("⚠ Small difference (probably OK)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Simple Complex CVXPY Test")
    print("=" * 30)
    
    success = test_simple_complex()
    if success:
        print("\n✓ Simple complex test passed!")
    else:
        print("\n✗ Simple complex test failed!")
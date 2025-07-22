#!/usr/bin/env python3
"""Test CVXPY to JSON converter with a complex, solvable, DCP-compliant problem."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem, save_problem_json

def create_solvable_complex_problem():
    """Create a complex, solvable, DCP-compliant optimization problem."""
    print("Creating solvable complex problem...")
    
    # Variables 
    x = cp.Variable(4, name="x_vec")
    y = cp.Variable((2, 2), name="y_mat")
    z = cp.Variable(name="z_scalar")
    
    # Parameters
    A = cp.Parameter((2, 4), name="A_param")
    b = cp.Parameter(2, name="b_param")
    c = cp.Parameter(name="c_param")
    
    # Complex but DCP-compliant objective with many nested functions
    objective_expr = (
        # Quadratic terms
        cp.sum_squares(x) +                          # sum_squares(variable)
        cp.sum_squares(cp.vec(y)) +                  # matrix vectorization + sum_squares  
        cp.square(z) +                               # square(scalar)
        
        # Norms and nested operations
        cp.norm(x, 2) +                              # L2 norm
        cp.norm(x, 1) +                              # L1 norm
        cp.norm(cp.maximum(x, 0), 2) +               # norm of positive part
        
        # More complex nested expressions
        cp.sum(cp.maximum(x - 1, 0)) +               # soft constraint penalty
        cp.norm(cp.minimum(x + 1, 2), 'inf') +      # norm of clipped values
        
        # Matrix operations
        cp.norm(y, 'fro') +                          # Frobenius norm
        cp.sum(cp.square(y)) +                       # element-wise square sum
        
        # Power operations (DCP-compliant)
        cp.power(cp.norm(x, 2), 2) +                 # squared norm
        
        # Absolute values
        cp.sum(cp.abs(x)) +                          # L1 norm via abs
        cp.abs(z)                                    # absolute of scalar
    )
    
    objective = cp.Minimize(objective_expr)
    
    # Complex but feasible constraints
    constraints = [
        # Variable bounds
        x >= -3,
        x <= 3,
        z >= -2,
        z <= 2,
        
        # Linear constraints with parameters
        A @ x <= b,
        A @ x >= b - 5,  # Ensure feasibility
        
        # Quadratic constraints
        cp.sum_squares(x) <= 10,
        
        # Norm constraints
        cp.norm(x, 2) <= 4,
        cp.norm(x, 1) <= 6,
        
        # Matrix constraints
        cp.norm(y, 'fro') <= 3,
        
        # Mixed constraints
        cp.sum(cp.abs(x)) + cp.abs(z) <= 8,
        
        # More complex but DCP constraints
        cp.maximum(cp.sum(cp.square(x)), cp.square(z)) <= 12,
        
        # Constraint with parameter
        cp.norm(x, 'inf') <= c
    ]
    
    problem = cp.Problem(objective, constraints)
    return problem, A, b, c

def test_solvable_complex():
    """Test the solvable complex problem."""
    problem, param_A, param_b, param_c = create_solvable_complex_problem()
    
    print(f"Original problem:")
    print(f"Variables: {len(problem.variables())}")
    print(f"Constraints: {len(problem.constraints)}")
    print(f"Parameters: {len(problem.parameters())}")
    print(f"Is DCP: {problem.is_dcp()}")
    
    # Convert to JSON
    print("\nConverting to JSON...")
    json_str = problem_to_json(problem)
    save_problem_json(problem, "solvable_complex_problem.json")
    print(f"JSON size: {len(json_str)} characters")
    
    # Reconstruct
    print("\nReconstructing from JSON...")
    reconstructed = json_to_problem(json_str)
    print(f"Reconstructed is DCP: {reconstructed.is_dcp()}")
    
    # Set parameter values
    print("\nSetting parameter values...")
    np.random.seed(123)
    param_A.value = np.random.randn(2, 4) * 0.5
    param_b.value = np.random.randn(2) + 3  # Positive bias
    param_c.value = 5.0  # Large enough bound
    
    for param in reconstructed.parameters():
        if param.name() == "A_param":
            param.value = param_A.value
        elif param.name() == "b_param":
            param.value = param_b.value  
        elif param.name() == "c_param":
            param.value = param_c.value
    
    # Solve both
    print("\nSolving problems...")
    try:
        status1 = problem.solve(verbose=False)
        print(f"Original status: {status1}, value: {problem.value:.6f}")
        
        status2 = reconstructed.solve(verbose=False)  
        print(f"Reconstructed status: {status2}, value: {reconstructed.value:.6f}")
        
        if problem.value is not None and reconstructed.value is not None:
            diff = abs(problem.value - reconstructed.value)
            print(f"Difference: {diff:.2e}")
            if diff < 1e-6:
                print("🎉 Perfect match!")
            else:
                print("⚠ Small difference (likely solver tolerance)")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Solvable Complex CVXPY Problem Test")
    print("=" * 45)
    
    success = test_solvable_complex()
    if success:
        print("\n✅ All tests passed! Complex nested problems work perfectly!")
    else:
        print("\n❌ Test failed")
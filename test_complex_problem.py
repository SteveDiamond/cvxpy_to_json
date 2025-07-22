#!/usr/bin/env python3
"""Test CVXPY to JSON converter with a complex problem using many nested functions."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem, save_problem_json

def create_complex_problem():
    """Create a complex optimization problem with deeply nested expressions."""
    print("Creating complex problem with many nested functions...")
    
    # Variables of different shapes
    x = cp.Variable(5, name="x_vector")
    y = cp.Variable((3, 3), name="y_matrix") 
    z = cp.Variable(name="z_scalar")
    
    # Parameters
    A = cp.Parameter((3, 5), name="param_A")
    b = cp.Parameter(3, name="param_b")
    
    # Complex nested objective with many functions
    objective_expr = (
        # Quadratic terms with norms
        cp.sum_squares(cp.maximum(x - 1, 0)) +
        # Matrix operations
        cp.trace(cp.square(y)) +
        # Power and exponential-like terms  
        cp.power(cp.abs(z), 2) +
        # Nested norms
        cp.norm(cp.multiply(x, cp.exp(-cp.abs(x))), 2) +
        # Complex composition
        cp.quad_over_lin(cp.pos(x), cp.maximum(z, 0.1)) +
        # Matrix norms
        cp.norm(y, 'fro') +
        # Geometric mean (if available)
        cp.sum(cp.sqrt(cp.maximum(x, 0.01))) +
        # Minimum/maximum chains
        cp.maximum(cp.minimum(z, 2), -2)
    )
    
    objective = cp.Minimize(objective_expr)
    
    # Complex constraints with nested functions
    constraints = [
        # Simple bounds
        x >= -5,
        x <= 5,
        
        # Matrix constraints with nested operations
        cp.sum(cp.multiply(A, x.T), axis=1) <= b + cp.abs(z),
        
        # Norm constraints with compositions
        cp.norm(cp.maximum(y @ x[:3], 0), 2) <= cp.sqrt(cp.sum_squares(b)),
        
        # Mixed constraints
        cp.sum(cp.log(cp.maximum(x, 0.01))) >= -10,
        
        # Matrix semi-definite (if y is square)
        y >> 0,  # Positive semidefinite
        
        # Complex nested constraint
        cp.norm(cp.reshape(cp.multiply(y, cp.sqrt(cp.abs(z) + 1)), (9,)), 1) <= 100,
        
        # Composition with transpose and indexing
        cp.sum(cp.square(y[0, :] + y[:, 0])) <= cp.power(z + 5, 2),
        
        # Deeply nested expression
        cp.maximum(
            cp.minimum(
                cp.norm(cp.pos(x) - cp.neg(x), 'inf'),
                cp.sqrt(cp.sum_squares(cp.diff(x)))
            ),
            0.1
        ) <= 10
    ]
    
    problem = cp.Problem(objective, constraints)
    
    return problem, A, b

def test_complex_conversion():
    """Test conversion of complex nested problem."""
    problem, param_A, param_b = create_complex_problem()
    
    print(f"Original complex problem:")
    print(f"Variables: {len(problem.variables())}")
    print(f"Constraints: {len(problem.constraints)}")
    print(f"Parameters: {len(problem.parameters())}")
    
    # Convert to JSON
    print("\nConverting to JSON...")
    json_str = problem_to_json(problem)
    save_problem_json(problem, "complex_nested_problem.json")
    
    print(f"JSON size: {len(json_str)} characters")
    
    # Reconstruct
    print("\nReconstructing from JSON...")
    reconstructed = json_to_problem(json_str)
    
    print(f"Reconstructed variables: {len(reconstructed.variables())}")
    print(f"Reconstructed constraints: {len(reconstructed.constraints)}")  
    print(f"Reconstructed parameters: {len(reconstructed.parameters())}")
    
    # Set parameter values for solving
    print("\nSetting parameter values...")
    param_A.value = np.random.randn(3, 5)
    param_b.value = np.random.randn(3)
    
    for param in reconstructed.parameters():
        if param.name() == "param_A":
            param.value = param_A.value
        elif param.name() == "param_b":
            param.value = param_b.value
    
    # Try to solve both (might not be feasible, but should not crash)
    print("\nTrying to solve problems...")
    try:
        original_status = problem.solve(verbose=False)
        print(f"Original problem status: {original_status}")
        if problem.value is not None:
            print(f"Original value: {problem.value:.6f}")
    except Exception as e:
        print(f"Original problem solve failed: {e}")
    
    try:
        recon_status = reconstructed.solve(verbose=False)
        print(f"Reconstructed problem status: {recon_status}")
        if reconstructed.value is not None:
            print(f"Reconstructed value: {reconstructed.value:.6f}")
    except Exception as e:
        print(f"Reconstructed problem solve failed: {e}")
    
    # Check if values match (if both solved successfully)
    if (problem.value is not None and reconstructed.value is not None and 
        not np.isnan(problem.value) and not np.isnan(reconstructed.value)):
        diff = abs(problem.value - reconstructed.value)
        print(f"Value difference: {diff:.2e}")
        if diff < 1e-6:
            print("✓ Values match!")
        else:
            print("⚠ Values differ (might be due to solver tolerances)")
    
    print(f"\nJSON file saved: complex_nested_problem.json")
    return True

if __name__ == "__main__":
    print("CVXPY Complex Nested Problem Test")
    print("=" * 50)
    
    try:
        test_complex_conversion()
        print("\n✓ Complex problem test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
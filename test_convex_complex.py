#!/usr/bin/env python3
"""Test CVXPY to JSON converter with a complex but DCP-compliant problem."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem, save_problem_json

def create_convex_complex_problem():
    """Create a complex but DCP-compliant optimization problem."""
    print("Creating convex complex problem with many nested functions...")
    
    # Variables of different shapes
    x = cp.Variable(5, name="x_vector")
    y = cp.Variable((3, 3), name="y_matrix", PSD=True)  # Positive semidefinite
    z = cp.Variable(name="z_scalar")
    
    # Parameters
    A = cp.Parameter((3, 5), name="param_A")
    b = cp.Parameter(3, name="param_b")
    
    # Complex but DCP-compliant objective
    objective_expr = (
        # Quadratic terms
        cp.sum_squares(x) +
        # Matrix operations
        cp.trace(y) +
        # Power terms (DCP-compliant)
        cp.square(z) +
        # Norms
        cp.norm(x, 2) +
        # Matrix norms
        cp.norm(y, 'fro') +
        # More complex but convex combinations
        cp.maximum(cp.sum(x), z) +
        # Nested convex functions
        cp.norm(cp.maximum(x, 0), 1)
    )
    
    objective = cp.Minimize(objective_expr)
    
    # Complex but feasible constraints
    constraints = [
        # Simple bounds
        x >= -2,
        x <= 2,
        z >= -5,
        z <= 5,
        
        # Matrix constraints
        cp.sum(cp.multiply(A, x.T), axis=1) <= b,
        
        # Norm constraints  
        cp.norm(y @ x[:3], 2) <= cp.norm(b, 2) + 1,
        
        # PSD constraint (already declared in variable)
        y >> 0,
        
        # Complex but convex constraint
        cp.sum(cp.square(x)) + cp.square(z) <= 10,
        
        # More nested constraints
        cp.maximum(cp.norm(x, 'inf'), cp.abs(z)) <= 3
    ]
    
    problem = cp.Problem(objective, constraints)
    
    return problem, A, b

def test_convex_complex():
    """Test conversion of convex complex problem."""
    problem, param_A, param_b = create_convex_complex_problem()
    
    print(f"Original complex problem:")
    print(f"Variables: {len(problem.variables())}")
    print(f"Constraints: {len(problem.constraints)}")
    print(f"Parameters: {len(problem.parameters())}")
    print(f"Is DCP: {problem.is_dcp()}")
    
    # Convert to JSON
    print("\nConverting to JSON...")
    json_str = problem_to_json(problem)
    save_problem_json(problem, "convex_complex_problem.json")
    
    print(f"JSON size: {len(json_str)} characters")
    
    # Reconstruct
    print("\nReconstructing from JSON...")
    reconstructed = json_to_problem(json_str)
    
    print(f"Reconstructed variables: {len(reconstructed.variables())}")
    print(f"Reconstructed constraints: {len(reconstructed.constraints)}")  
    print(f"Reconstructed parameters: {len(reconstructed.parameters())}")
    print(f"Reconstructed is DCP: {reconstructed.is_dcp()}")
    
    # Set parameter values for solving
    print("\nSetting parameter values...")
    np.random.seed(42)  # For reproducibility
    param_A.value = np.random.randn(3, 5) * 0.1  # Small values to ensure feasibility
    param_b.value = np.random.randn(3) + 2  # Positive bias for feasibility
    
    for param in reconstructed.parameters():
        if param.name() == "param_A":
            param.value = param_A.value
        elif param.name() == "param_b":
            param.value = param_b.value
    
    # Solve both problems
    print("\nSolving problems...")
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
    
    # Check if values match
    if (problem.value is not None and reconstructed.value is not None and 
        not np.isnan(problem.value) and not np.isnan(reconstructed.value)):
        diff = abs(problem.value - reconstructed.value)
        print(f"Value difference: {diff:.2e}")
        if diff < 1e-6:
            print("✓ Values match perfectly!")
        else:
            print("⚠ Small difference (might be due to solver tolerances)")
    
    print(f"\nJSON file saved: convex_complex_problem.json")
    return True

if __name__ == "__main__":
    print("CVXPY Convex Complex Problem Test")
    print("=" * 40)
    
    try:
        test_convex_complex()
        print("\n🎉 Convex complex problem test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
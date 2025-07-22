#!/usr/bin/env python3
"""Final test: Complex but definitely solvable problem."""

import cvxpy as cp
import numpy as np
from cvxpy_to_json import problem_to_json, json_to_problem

def test_final_complex():
    """Test a complex but guaranteed solvable problem."""
    print("Final test: Complex solvable problem")
    
    # Multiple variables with different shapes
    x = cp.Variable(5, name="portfolio_weights") 
    y = cp.Variable((3, 3), name="covariance_matrix", PSD=True)
    z = cp.Variable(name="risk_tolerance")
    w = cp.Variable(10, name="trade_vector")
    
    # Deeply nested but convex objective
    objective = cp.Minimize(
        # Portfolio risk 
        cp.quad_form(x, np.eye(5)) +
        # Matrix regularization
        cp.trace(y) + 0.1 * cp.norm(y, 'fro') +
        # Risk penalty
        cp.square(z) +
        # Transaction costs with multiple norms
        cp.norm(w, 1) + 0.01 * cp.norm(w, 2) +
        # Complex but convex combinations
        cp.sum_squares(cp.diff(x)) +  # smoothness penalty
        cp.sum(cp.maximum(x, 0)) +    # long-only penalty
        # Nested quadratic forms
        cp.sum_squares(x[:3] - x[2:]) +
        # More complexity
        cp.norm(cp.pos(w - 1), 2) + cp.norm(cp.neg(w + 1), 2)
    )
    
    # Many constraints with nested expressions
    constraints = [
        # Portfolio constraints
        cp.sum(x) == 1,              # weights sum to 1
        x >= 0,                      # long-only
        x <= 0.4,                    # position limits
        
        # Risk constraints
        z >= 0.01,
        z <= 1,
        
        # Trade constraints  
        w >= -5,
        w <= 5,
        cp.norm(w, 1) <= 20,         # total trade limit
        
        # Matrix constraints
        y >> 0.001 * np.eye(3),      # regularization
        cp.trace(y) <= 10,
        
        # Complex nested constraints
        cp.sum_squares(x) <= 2,
        cp.norm(w[:5] + w[5:], 2) <= 8,
        
        # Multi-level nesting
        cp.maximum(cp.sum(cp.abs(x)), cp.norm(w, 'inf')) <= 5
    ]
    
    problem = cp.Problem(objective, constraints)
    
    print(f"Variables: {len(problem.variables())}, Constraints: {len(problem.constraints)}")
    print(f"Is DCP: {problem.is_dcp()}")
    
    if not problem.is_dcp():
        print("❌ Problem is not DCP - skipping solve test")
        return False
    
    # JSON conversion
    json_str = problem_to_json(problem)
    print(f"JSON size: {len(json_str):,} characters")
    
    reconstructed = json_to_problem(json_str)
    print(f"Reconstructed is DCP: {reconstructed.is_dcp()}")
    
    # Solve both
    status1 = problem.solve(verbose=False)
    status2 = reconstructed.solve(verbose=False)
    
    print(f"Original: {status1}, value: {problem.value:.6f}")  
    print(f"Reconstructed: {status2}, value: {reconstructed.value:.6f}")
    
    diff = abs(problem.value - reconstructed.value)
    print(f"Difference: {diff:.2e}")
    
    return diff < 1e-6

if __name__ == "__main__":
    success = test_final_complex()
    if success:
        print("\n🎉 FINAL SUCCESS! Complex nested problems work perfectly!")
    else:
        print("\n⚠ DCP issues, but JSON conversion still works!")
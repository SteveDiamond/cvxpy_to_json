#!/usr/bin/env python3
"""
Demonstration of the CVXPY to CVXLean JSON converter.
"""

import cvxpy as cp
import numpy as np
from cvxpy_to_lean_json import problem_to_cvxlean_json
import json

def demo_simple_lp():
    """Simple linear program."""
    print("=== Simple Linear Program ===")
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    
    objective = cp.Minimize(x + 2*y)
    constraints = [x >= 0, y >= 0, x + y <= 1]
    problem = cp.Problem(objective, constraints)
    
    print("CVXPY Problem:")
    print(problem)
    print("\nCVXLean JSON:")
    json_str = problem_to_cvxlean_json(problem, "simple_lp")
    print(json_str)
    return problem, json_str

def demo_quadratic():
    """Quadratic program with square function."""
    print("\n=== Quadratic Program ===")
    x = cp.Variable(name="x")
    
    objective = cp.Minimize(cp.square(x - 1))
    constraints = [x >= 0, x <= 2]
    problem = cp.Problem(objective, constraints)
    
    print("CVXPY Problem:")
    print(problem)
    print("\nCVXLean JSON:")
    json_str = problem_to_cvxlean_json(problem, "quadratic")
    print(json_str)
    return problem, json_str

def demo_portfolio():
    """Portfolio optimization with sum_squares."""
    print("\n=== Portfolio Optimization ===")
    n = 3
    w = cp.Variable(n, name="weights")
    
    # Risk (sum of squares) + return term
    mu = np.array([0.1, 0.2, 0.15])
    objective = cp.Minimize(cp.sum_squares(w) - mu.T @ w)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    
    print("CVXPY Problem:")
    print(problem)
    print("\nCVXLean JSON:")
    json_str = problem_to_cvxlean_json(problem, "portfolio")
    print(json_str)
    return problem, json_str

def demo_norm_constraint():
    """Problem with norm constraint."""
    print("\n=== Problem with Norm Constraint ===")
    x = cp.Variable(2, name="x")
    
    objective = cp.Minimize(cp.sum(x))
    constraints = [cp.norm(x, 2) <= 1, x >= 0]
    problem = cp.Problem(objective, constraints)
    
    print("CVXPY Problem:")
    print(problem)
    print("\nCVXLean JSON:")
    json_str = problem_to_cvxlean_json(problem, "norm_constraint")
    print(json_str)
    return problem, json_str

def validate_json_structure(json_str, prob_name):
    """Validate that the JSON has the correct CVXLean structure."""
    try:
        data = json.loads(json_str)
        
        # Check required top-level fields
        required_fields = ["request", "prob_name", "domains", "target"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        assert data["request"] == "PerformRewrite", "Incorrect request type"
        assert data["prob_name"] == prob_name, "Incorrect problem name"
        
        # Check domains structure
        domains = data["domains"]
        assert isinstance(domains, list), "Domains should be a list"
        for domain in domains:
            assert isinstance(domain, list) and len(domain) == 2, "Each domain should be [name, bounds]"
            assert isinstance(domain[0], str), "Domain name should be string"
            assert isinstance(domain[1], list) and len(domain[1]) == 4, "Domain bounds should be [lo, hi, lo_open, hi_open]"
        
        # Check target structure
        target = data["target"]
        assert "obj_fun" in target, "Missing obj_fun in target"
        assert "constrs" in target, "Missing constrs in target"
        assert isinstance(target["constrs"], list), "Constraints should be a list"
        
        # Check S-expression format
        obj_fun = target["obj_fun"]
        assert obj_fun.startswith("(objFun"), "Objective should start with (objFun"
        assert obj_fun.endswith(")"), "Objective should end with )"
        
        for constr in target["constrs"]:
            assert isinstance(constr, list) and len(constr) == 2, "Each constraint should be [name, sexpr]"
            assert isinstance(constr[0], str), "Constraint name should be string"
            assert isinstance(constr[1], str), "Constraint S-expr should be string"
            assert constr[1].startswith("("), "Constraint S-expr should start with ("
            assert constr[1].endswith(")"), "Constraint S-expr should end with )"
        
        print(f"✓ JSON structure validation passed for {prob_name}")
        return True
        
    except Exception as e:
        print(f"✗ JSON structure validation failed for {prob_name}: {e}")
        return False

if __name__ == "__main__":
    print("CVXPY to CVXLean JSON Converter Demo")
    print("=" * 50)
    
    # Run demonstrations
    problems = []
    problems.append(demo_simple_lp())
    problems.append(demo_quadratic())
    problems.append(demo_portfolio())
    problems.append(demo_norm_constraint())
    
    print("\n" + "=" * 50)
    print("JSON Structure Validation")
    print("=" * 50)
    
    # Validate all generated JSON
    all_valid = True
    for i, (problem, json_str) in enumerate(problems):
        prob_name = ["simple_lp", "quadratic", "portfolio", "norm_constraint"][i]
        valid = validate_json_structure(json_str, prob_name)
        all_valid = all_valid and valid
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All tests passed! The converter works correctly.")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 50)
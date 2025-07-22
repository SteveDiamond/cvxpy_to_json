#!/usr/bin/env python3
"""
CVXPY to JSON Converter

Converts CVXPY optimization problems to JSON format while preserving variable names
and all problem structure. Supports reconstruction of the original problem.

Usage:
    from cvxpy_to_json import problem_to_json, json_to_problem
    
    # Convert problem to JSON
    json_str = problem_to_json(problem)
    
    # Reconstruct problem from JSON
    reconstructed_problem = json_to_problem(json_str)
"""

import json
import numpy as np
import cvxpy as cp
from typing import Dict, Any, List, Optional, Union
import warnings


class CVXPYJSONEncoder:
    """Encoder for converting CVXPY problems to JSON format."""
    
    def problem_to_dict(self, problem: cp.Problem) -> Dict[str, Any]:
        """Convert a CVXPY problem to a dictionary."""
        result = {
            "objective": self._serialize_objective(problem.objective),
            "constraints": [self._serialize_constraint(c) for c in problem.constraints]
        }
        
        return result
    
    
    def _serialize_objective(self, objective) -> Dict[str, Any]:
        """Serialize a CVXPY Objective."""
        if objective is None:
            return None
            
        return {
            "sense": objective.NAME,
            "expression": self._serialize_expression(objective.expr)
        }
    
    def _serialize_constraint(self, constraint) -> Dict[str, Any]:
        """Serialize a CVXPY Constraint."""
        return {
            "type": constraint.__class__.__name__,
            "args": [self._serialize_expression(arg) for arg in constraint.args]
        }
    
    def _serialize_expression(self, expr) -> Dict[str, Any]:
        """Serialize a CVXPY Expression."""
        if expr is None:
            return None
        
        expr_type = expr.__class__.__name__
        expr_dict = {
            "type": expr_type
        }
        
        # Handle different expression types
        if isinstance(expr, cp.Variable):
            expr_dict["name"] = expr.name()
            if expr.shape != ():
                expr_dict["shape"] = list(expr.shape)
        elif isinstance(expr, cp.Parameter):
            expr_dict["name"] = expr.name()
            if expr.shape != ():
                expr_dict["shape"] = list(expr.shape)
        elif isinstance(expr, cp.Constant):
            expr_dict["value"] = self._serialize_array(expr.value)
        else:
            # Handle atoms and other expressions with args
            if hasattr(expr, 'args') and expr.args:
                expr_dict["args"] = [self._serialize_expression(arg) for arg in expr.args]
            
            # Use get_data method for proper CVXPY serialization
            if hasattr(expr, 'get_data'):
                data = expr.get_data()
                if data is not None:
                    expr_dict["data"] = self._serialize_value(data)
                
        return expr_dict
    
    def _serialize_array(self, arr) -> Union[List, float, int, None]:
        """Serialize numpy arrays and scalars."""
        if arr is None:
            return None
        elif np.isscalar(arr):
            if np.isnan(arr) or np.isinf(arr):
                return {"special": "nan" if np.isnan(arr) else ("inf" if arr > 0 else "-inf")}
            return float(arr) if isinstance(arr, (np.floating, float)) else int(arr)
        elif isinstance(arr, np.ndarray):
            if arr.size == 0:
                return []
            # Handle sparse arrays  
            try:
                import scipy.sparse
                if scipy.sparse.issparse(arr):
                    return {
                        "sparse": True,
                        "format": arr.format,
                        "data": arr.data.tolist(),
                        "indices": arr.indices.tolist() if hasattr(arr, 'indices') else None,
                        "indptr": arr.indptr.tolist() if hasattr(arr, 'indptr') else None,
                        "shape": list(arr.shape)
                    }
            except ImportError:
                pass
                
            # For scalar arrays, just return the value
            if arr.shape == ():
                return arr.item()
            # For 1D arrays with single element, return the value
            elif arr.size == 1:
                return arr.item()
            else:
                return arr.tolist()
        else:
            return arr
    
    def _serialize_value(self, value) -> Any:
        """Serialize various value types."""
        if value is None:
            return None  # Consistently use None, not null
        elif isinstance(value, (np.ndarray, np.number)):
            return self._serialize_array(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            # Handle Fraction objects exactly
            try:
                from fractions import Fraction
                if isinstance(value, Fraction):
                    return {"fraction": {"numerator": value.numerator, "denominator": value.denominator}}
            except ImportError:
                pass
            
            # Handle slice objects
            if isinstance(value, slice):
                return {"slice": {"start": value.start, "stop": value.stop, "step": value.step}}
            
            # Handle numpy scalar types
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            
            return value


class CVXPYJSONDecoder:
    """Decoder for reconstructing CVXPY problems from JSON format."""
    
    def __init__(self):
        self._variable_cache = {}  # name -> Variable
        self._parameter_cache = {}  # name -> Parameter
    
    def dict_to_problem(self, data: Dict[str, Any]) -> cp.Problem:
        """Reconstruct a CVXPY problem from a dictionary."""
        # Clear caches
        self._variable_cache.clear()
        self._parameter_cache.clear()
        
        # Reconstruct objective and constraints
        objective = self._deserialize_objective(data["objective"])
        if objective is None:
            raise ValueError("Failed to reconstruct objective")
            
        constraints = [self._deserialize_constraint(c) for c in data["constraints"]]
        # Filter out None constraints
        constraints = [c for c in constraints if c is not None]
        if len(constraints) != len(data["constraints"]):
            warnings.warn(f"Some constraints failed to reconstruct: {len(data['constraints']) - len(constraints)} lost")
        
        # Create the problem
        problem = cp.Problem(objective, constraints)
        
        return problem
    
    
    def _deserialize_objective(self, obj_data: Dict[str, Any]) -> Union[cp.Minimize, cp.Maximize]:
        """Reconstruct a CVXPY Objective."""
        if obj_data is None:
            return None
            
        expr = self._deserialize_expression(obj_data["expression"])
        if expr is None:
            warnings.warn("Failed to reconstruct objective expression")
            return None
        
        if obj_data["sense"] == "minimize":
            return cp.Minimize(expr)
        else:
            return cp.Maximize(expr)
    
    def _deserialize_constraint(self, constr_data: Dict[str, Any]) -> cp.Constraint:
        """Reconstruct a CVXPY Constraint."""
        args = [self._deserialize_expression(arg) for arg in constr_data["args"]]
        
        # Check if any args failed to reconstruct
        if any(arg is None for arg in args):
            warnings.warn(f"Could not reconstruct constraint {constr_data['type']}: some arguments failed")
            return None
        
        # Get constraint class from cvxpy.constraints
        constr_type = constr_data["type"]
        
        # Import the constraint class - most are in cvxpy.constraints
        try:
            module = __import__("cvxpy.constraints", fromlist=[constr_type])
            constr_class = getattr(module, constr_type)
        except (ImportError, AttributeError):
            # Try other common locations
            try:
                module = __import__("cvxpy.constraints.constraint", fromlist=[constr_type])
                constr_class = getattr(module, constr_type)
            except (ImportError, AttributeError):
                warnings.warn(f"Unknown constraint type: {constr_type}")
                return None
        
        # Create constraint
        try:
            if len(args) == 2:
                return constr_class(args[0], args[1])
            else:
                return constr_class(*args)
        except Exception as e:
            warnings.warn(f"Could not create constraint {constr_type}: {e}")
            return None
    
    def _deserialize_expression(self, expr_data: Dict[str, Any]):
        """Reconstruct a CVXPY Expression."""
        if expr_data is None:
            return None
        
        expr_type = expr_data["type"]
        
        # Handle different expression types
        if expr_type == "Variable":
            name = expr_data["name"]
            shape = tuple(expr_data.get("shape", ()))
            
            # Check cache first
            cache_key = (name, shape)
            if cache_key in self._variable_cache:
                return self._variable_cache[cache_key]
            
            # Create new variable
            var = cp.Variable(shape, name=name)
            self._variable_cache[cache_key] = var
            return var
            
        elif expr_type == "Parameter":
            name = expr_data["name"]
            shape = tuple(expr_data.get("shape", ()))
            
            # Check cache first
            cache_key = (name, shape)
            if cache_key in self._parameter_cache:
                return self._parameter_cache[cache_key]
            
            # Create new parameter
            param = cp.Parameter(shape, name=name)
            self._parameter_cache[cache_key] = param
            return param
            
        elif expr_type == "Constant":
            value = self._deserialize_array(expr_data["value"])
            return cp.Constant(value)
        else:
            # Handle atoms and other expressions
            # Get the class from cvxpy modules
            try:
                expr_class = self._get_expression_class(expr_type)
            except ValueError as e:
                warnings.warn(str(e))
                return None
            
            args = []
            if "args" in expr_data:
                args = [self._deserialize_expression(arg) for arg in expr_data["args"]]
                # Check if any args failed to reconstruct
                if any(arg is None for arg in args):
                    warnings.warn(f"Could not reconstruct {expr_type}: some arguments failed to reconstruct")
                    return None
            
            # Get additional data for reconstruction
            data = []
            if "data" in expr_data:
                data = self._deserialize_value(expr_data["data"])
                if not isinstance(data, list):
                    data = [data]
            
            # Create the expression using args + data (CVXPY pattern)
            try:
                # Special handling for expressions with non-standard constructors
                if expr_type == "AddExpression":
                    return expr_class(args)  # Takes iterable of expressions
                elif expr_type == "MulExpression" and len(args) == 2:
                    # Handle MulExpression specially - check for scalar multiplication issues
                    return expr_class(args[0], args[1])
                else:
                    return expr_class(*(args + data))
            except Exception as e:
                warnings.warn(f"Could not reconstruct {expr_type}: {e}")
                return None
    
    def _get_expression_class(self, expr_type: str):
        """Get the expression class by name from CVXPY modules."""
        # Special cases where class name matches module name
        special_modules = {
            "power": "cvxpy.atoms.elementwise.power",
            "abs": "cvxpy.atoms.elementwise.abs", 
            "maximum": "cvxpy.atoms.elementwise.maximum",
            "minimum": "cvxpy.atoms.elementwise.minimum",
            "exp": "cvxpy.atoms.elementwise.exp",
            "log": "cvxpy.atoms.elementwise.log",
            "sqrt": "cvxpy.atoms.elementwise.sqrt",
            "square": "cvxpy.atoms.elementwise.square",
            "trace": "cvxpy.atoms.affine.trace",
            "index": "cvxpy.atoms.affine.index",
            "broadcast_to": "cvxpy.atoms.affine.broadcast_to",
        }
        
        # Try special modules first
        if expr_type in special_modules:
            try:
                module = __import__(special_modules[expr_type], fromlist=[expr_type])
                return getattr(module, expr_type)
            except (ImportError, AttributeError):
                pass
        
        # Common module locations for other CVXPY expressions
        modules_to_try = [
            "cvxpy.atoms.affine.add_expr",
            "cvxpy.atoms.affine.binary_operators",
            "cvxpy.atoms.affine.unary_operators", 
            "cvxpy.atoms.elementwise",
            "cvxpy.atoms.quad_over_lin",
            "cvxpy.atoms.norm",
            "cvxpy.atoms.sum_squares",
            "cvxpy.atoms.affine.sum",
            "cvxpy.atoms.affine",
            "cvxpy.atoms.affine.promote",
            "cvxpy.atoms.affine.multiply",
            "cvxpy.atoms.affine.reshape",
            "cvxpy.atoms.affine.index",
            "cvxpy.atoms.affine.diff",
            "cvxpy.atoms",
        ]
        
        for module_name in modules_to_try:
            try:
                module = __import__(module_name, fromlist=[expr_type])
                if hasattr(module, expr_type):
                    return getattr(module, expr_type)
            except ImportError:
                continue
                
        # If not found, raise error
        raise ValueError(f"Unknown expression type: {expr_type}")
    
    def _deserialize_array(self, arr_data) -> Union[np.ndarray, float, int, None]:
        """Reconstruct numpy arrays and scalars."""
        if arr_data is None:
            return None
        elif isinstance(arr_data, (int, float)):
            return arr_data
        elif isinstance(arr_data, list):
            return np.array(arr_data)
        elif isinstance(arr_data, dict):
            if "special" in arr_data:
                if arr_data["special"] == "nan":
                    return np.nan
                elif arr_data["special"] == "inf":
                    return np.inf
                elif arr_data["special"] == "-inf":
                    return -np.inf
            elif "sparse" in arr_data and arr_data["sparse"]:
                try:
                    import scipy.sparse
                    if arr_data["format"] == "csc":
                        return scipy.sparse.csc_matrix(
                            (arr_data["data"], arr_data["indices"], arr_data["indptr"]),
                            shape=arr_data["shape"]
                        )
                    # Add other sparse formats as needed
                except ImportError:
                    warnings.warn("scipy not available, cannot reconstruct sparse array")
                    return None
        return arr_data
    
    def _deserialize_value(self, value) -> Any:
        """Deserialize various value types."""
        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]
        elif isinstance(value, dict):
            if "fraction" in value:
                # Reconstruct Fraction objects exactly
                from fractions import Fraction
                frac_data = value["fraction"]
                return Fraction(frac_data["numerator"], frac_data["denominator"])
            elif "slice" in value:
                # Reconstruct slice objects
                slice_data = value["slice"]
                return slice(slice_data["start"], slice_data["stop"], slice_data["step"])
            elif "array" in value or "sparse" in value or "special" in value:
                return self._deserialize_array(value)
            else:
                return {k: self._deserialize_value(v) for k, v in value.items()}
        else:
            return value


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            from fractions import Fraction
            if isinstance(obj, Fraction):
                return {"fraction": {"numerator": obj.numerator, "denominator": obj.denominator}}
        except ImportError:
            pass
        if isinstance(obj, slice):
            return {"slice": {"start": obj.start, "stop": obj.stop, "step": obj.step}}
        return super().default(obj)


def problem_to_json(problem: cp.Problem, indent: Optional[int] = 2) -> str:
    """
    Convert a CVXPY problem to JSON string.
    
    Args:
        problem: CVXPY Problem to convert
        indent: JSON indentation (None for compact)
    
    Returns:
        JSON string representation
    """
    encoder = CVXPYJSONEncoder()
    problem_dict = encoder.problem_to_dict(problem)
    return json.dumps(problem_dict, indent=indent, cls=NumpyEncoder)


def json_to_problem(json_str: str) -> cp.Problem:
    """
    Reconstruct a CVXPY problem from JSON string.
    
    Args:
        json_str: JSON string representation
    
    Returns:
        Reconstructed CVXPY Problem
    """
    data = json.loads(json_str)
    decoder = CVXPYJSONDecoder()
    return decoder.dict_to_problem(data)


def save_problem_json(problem: cp.Problem, filename: str, indent: Optional[int] = 2):
    """Save a CVXPY problem to a JSON file."""
    json_str = problem_to_json(problem, indent)
    with open(filename, 'w') as f:
        f.write(json_str)


def load_problem_json(filename: str) -> cp.Problem:
    """Load a CVXPY problem from a JSON file."""
    with open(filename, 'r') as f:
        json_str = f.read()
    return json_to_problem(json_str)


if __name__ == "__main__":
    # Example usage and basic test
    print("CVXPY to JSON Converter")
    print("=" * 40)
    
    # Create a simple test problem
    x = cp.Variable(2, name="x_vector")
    y = cp.Variable(name="y_scalar")
    
    objective = cp.Minimize(cp.sum_squares(x) + cp.square(y))
    constraints = [x >= 0, y <= 5, cp.norm(x, 2) <= 1]
    
    problem = cp.Problem(objective, constraints)
    
    print("Original problem:")
    print(problem)
    
    # Convert to JSON
    json_str = problem_to_json(problem)
    print(f"\nJSON size: {len(json_str)} characters")
    
    # Reconstruct from JSON
    try:
        reconstructed = json_to_problem(json_str)
        print("\nReconstructed problem:")
        print(reconstructed)
        print("\nConversion successful!")
    except Exception as e:
        print(f"\nError during reconstruction: {e}")
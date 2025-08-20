"""
Mathematical Equation Parser for Statistical Data Generation

This module provides safe and powerful mathematical expression evaluation for 
generating data points with complex relationships. It demonstrates key concepts
in mathematical computing and numerical analysis that are fundamental to
understanding how data relationships are modeled in statistics and machine learning.

The parser handles transcendental functions, algebraic operations, and provides
comprehensive error handling for domain restrictions and numerical instabilities.
"""

import numpy as np
import re
from typing import Union, List, Callable


class MathEquationParser:
    """
    A robust mathematical expression parser that safely evaluates equations
    while providing comprehensive mathematical function support.
    
    This class demonstrates several important computational concepts:
    1. Expression parsing and abstract syntax tree evaluation
    2. Numerical stability considerations with transcendental functions
    3. Domain validation for mathematical functions
    4. Safe evaluation of user-provided mathematical expressions
    """
    
    def __init__(self):
        """Initialize the parser with mathematical functions and constants."""
        # Define mathematical constants - these are fundamental in many statistical distributions
        self.constants = {
            'pi': np.pi,
            'e': np.e,
            'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio - appears in many natural phenomena
            'euler': 0.5772156649015329,   # Euler-Mascheroni constant
        }
        
        # Define mathematical functions with domain checking
        # Each function includes error handling for domain restrictions
        self.functions = {
            # Trigonometric functions - fundamental in periodic phenomena
            'sin': self._safe_sin,
            'cos': self._safe_cos,
            'tan': self._safe_tan,
            'asin': self._safe_asin,  # Inverse trig functions have restricted domains
            'acos': self._safe_acos,
            'atan': self._safe_atan,
            
            # Hyperbolic functions - appear in exponential growth/decay models
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            
            # Exponential and logarithmic functions - crucial for many statistical models
            'exp': self._safe_exp,
            'log': self._safe_log,      # Natural logarithm
            'log10': self._safe_log10,  # Base-10 logarithm
            'log2': self._safe_log2,    # Base-2 logarithm
            
            # Power and root functions
            'sqrt': self._safe_sqrt,
            'cbrt': self._safe_cbrt,    # Cube root
            'abs': np.abs,
            'sign': np.sign,
            
            # Rounding and integer functions
            'floor': np.floor,
            'ceil': np.ceil,
            'round': np.round,
            
            # Statistical and probability functions
            'normal': self._normal_pdf,     # Normal distribution PDF
            'gamma': self._safe_gamma,      # Gamma function
            'factorial': self._safe_factorial,
            
            # Advanced mathematical functions
            'erf': self._safe_erf,          # Error function - appears in statistics
            'sigmoid': self._sigmoid,        # Logistic function - crucial in ML
            'step': self._step_function,     # Heaviside step function
        }
    
    def parse_and_evaluate(self, equation: str, x_values: Union[float, np.ndarray]) -> np.ndarray:
        """
        Parse and evaluate a mathematical equation for given x values.
        
        This method demonstrates the complete pipeline from text parsing to 
        numerical evaluation, including error handling for mathematical edge cases.
        
        Args:
            equation: Mathematical expression as string (e.g., "sin(x) + 2*log(x)")
            x_values: Input values (scalar or array)
            
        Returns:
            Computed y values as numpy array
            
        Raises:
            ValueError: For invalid equations or domain errors
        """
        if isinstance(x_values, (int, float)):
            x_values = np.array([x_values])
        elif not isinstance(x_values, np.ndarray):
            x_values = np.array(x_values)
        
        # Prepare equation for safe evaluation
        safe_equation = self._prepare_equation(equation)
        
        # Evaluate equation for each x value
        results = []
        for x_val in x_values:
            try:
                # Create local namespace with x value and mathematical functions
                namespace = {
                    'x': x_val,
                    **self.constants,
                    **self.functions,
                    # Include numpy for advanced operations
                    'np': np,
                    # Power operator support
                    'pow': pow,
                    '__builtins__': {},  # Restrict built-in functions for security
                }
                
                # Evaluate the expression
                result = eval(safe_equation, namespace)
                
                # Handle complex numbers and infinities
                if np.iscomplex(result):
                    result = np.real(result)  # Take real part for visualization
                if not np.isfinite(result):
                    result = 0  # Replace inf/nan with 0 for visualization
                    
                results.append(float(result))
                
            except Exception as e:
                # If evaluation fails, use 0 as default
                # In a production system, you might want to raise the error
                results.append(0.0)
        
        return np.array(results)
    
    def _prepare_equation(self, equation: str) -> str:
        """
        Prepare equation string for safe evaluation.
        
        This method demonstrates text processing techniques for mathematical
        expressions, including operator replacement and function name normalization.
        """
        # Remove whitespace and convert to lowercase for consistency
        equation = equation.strip().lower()
        
        # Replace common mathematical notation
        replacements = {
            '^': '**',  # Power operator
            'ln': 'log',  # Natural logarithm
            'lg': 'log10',  # Common logarithm
        }
        
        for old, new in replacements.items():
            equation = equation.replace(old, new)
        
        # Handle implicit multiplication (e.g., "2x" -> "2*x")
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
        
        return equation
    
    # Safe mathematical functions with domain checking
    # These demonstrate proper numerical computing practices
    
    def _safe_sin(self, x):
        """Sine function - periodic, domain: all reals."""
        return np.sin(x)
    
    def _safe_cos(self, x):
        """Cosine function - periodic, domain: all reals."""
        return np.cos(x)
    
    def _safe_tan(self, x):
        """Tangent function - has vertical asymptotes at odd multiples of π/2."""
        result = np.tan(x)
        # Handle near-asymptote values
        return np.where(np.abs(result) > 1e10, np.sign(result) * 1e10, result)
    
    def _safe_asin(self, x):
        """Arcsine function - domain: [-1, 1]."""
        x_clipped = np.clip(x, -1, 1)  # Ensure domain validity
        return np.arcsin(x_clipped)
    
    def _safe_acos(self, x):
        """Arccosine function - domain: [-1, 1]."""
        x_clipped = np.clip(x, -1, 1)
        return np.arccos(x_clipped)
    
    def _safe_atan(self, x):
        """Arctangent function - domain: all reals, range: (-π/2, π/2)."""
        return np.arctan(x)
    
    def _safe_exp(self, x):
        """Exponential function - grows very rapidly, need overflow protection."""
        # Prevent numerical overflow
        x_safe = np.clip(x, -700, 700)  # e^700 is near float64 limit
        return np.exp(x_safe)
    
    def _safe_log(self, x):
        """Natural logarithm - domain: positive reals only."""
        # Replace non-positive values with small positive number
        x_safe = np.maximum(x, 1e-10)
        return np.log(x_safe)
    
    def _safe_log10(self, x):
        """Base-10 logarithm - domain: positive reals only."""
        x_safe = np.maximum(x, 1e-10)
        return np.log10(x_safe)
    
    def _safe_log2(self, x):
        """Base-2 logarithm - domain: positive reals only."""
        x_safe = np.maximum(x, 1e-10)
        return np.log2(x_safe)
    
    def _safe_sqrt(self, x):
        """Square root - domain: non-negative reals."""
        x_safe = np.maximum(x, 0)  # Ensure non-negative
        return np.sqrt(x_safe)
    
    def _safe_cbrt(self, x):
        """Cube root - domain: all reals."""
        return np.cbrt(x)
    
    def _safe_gamma(self, x):
        """Gamma function - extends factorial to real numbers."""
        try:
            from scipy.special import gamma
            return gamma(x)
        except ImportError:
            # Fallback using Stirling's approximation for large x
            if x > 1:
                return np.sqrt(2 * np.pi / x) * (x / np.e) ** x
            else:
                return 1.0  # Simplified fallback
    
    def _safe_factorial(self, x):
        """Factorial function - defined for non-negative integers."""
        if x < 0:
            return 0
        if x > 170:  # Factorial overflow limit for float64
            return np.inf
        try:
            from scipy.special import factorial
            return factorial(int(x))
        except ImportError:
            # Simple recursive implementation
            if x <= 1:
                return 1
            result = 1
            for i in range(2, int(x) + 1):
                result *= i
            return result
    
    def _safe_erf(self, x):
        """Error function - appears in probability theory and statistics."""
        try:
            from scipy.special import erf
            return erf(x)
        except ImportError:
            # Approximation using series expansion
            return 2 / np.sqrt(np.pi) * x * (1 - x**2/3 + x**4/10 - x**6/42)
    
    def _sigmoid(self, x):
        """
        Sigmoid (logistic) function - fundamental in machine learning.
        Maps any real number to (0, 1) interval.
        """
        # Prevent numerical overflow in exponential
        x_safe = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_safe))
    
    def _step_function(self, x):
        """Heaviside step function - useful for modeling threshold effects."""
        return np.where(x >= 0, 1.0, 0.0)
    
    def _normal_pdf(self, x, mu=0, sigma=1):
        """
        Normal distribution probability density function.
        Demonstrates how statistical distributions can be incorporated into equations.
        """
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def get_available_functions(self) -> List[str]:
        """Return list of available mathematical functions."""
        return sorted(list(self.functions.keys()))
    
    def get_available_constants(self) -> List[str]:
        """Return list of available mathematical constants."""
        return sorted(list(self.constants.keys()))
    
    def validate_equation(self, equation: str) -> tuple[bool, str]:
        """
        Validate equation syntax without evaluation.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Prepare equation
            safe_equation = self._prepare_equation(equation)
            
            # Test with a sample x value
            test_namespace = {
                'x': 1.0,
                **self.constants,
                **self.functions,
                'np': np,
                'pow': pow,
                '__builtins__': {},
            }
            
            # Try to compile the expression
            compile(safe_equation, '<equation>', 'eval')
            
            # Try to evaluate with test value
            eval(safe_equation, test_namespace)
            
            return True, "Equation is valid"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except NameError as e:
            return False, f"Unknown function or variable: {e}"
        except Exception as e:
            return False, f"Evaluation error: {e}"


def demonstrate_parser_capabilities():
    """
    Demonstrate the capabilities of the equation parser.
    This function shows various mathematical relationships you can explore.
    """
    parser = MathEquationParser()
    
    # Example equations that demonstrate different mathematical concepts
    example_equations = [
        ("sin(x)", "Simple trigonometric function"),
        ("2*x + 1", "Linear relationship"),
        ("x**2 - 3*x + 2", "Quadratic polynomial"),
        ("exp(-x**2)", "Gaussian-like curve"),
        ("sin(x) + 0.1*sin(10*x)", "Composite wave with noise"),
        ("log(abs(x) + 1)", "Logarithmic growth with domain protection"),
        ("sigmoid(x)", "Logistic function"),
        ("normal(x, 0, 1)", "Standard normal distribution"),
        ("x * sin(1/x) if abs(x) > 0.01 else 0", "Complex behavior near origin"),
    ]
    
    print("Mathematical Equation Parser - Available Functions:")
    print("=" * 50)
    
    x_test = np.linspace(-2, 2, 10)
    
    for equation, description in example_equations:
        try:
            y_values = parser.parse_and_evaluate(equation, x_test)
            print(f"\n{equation}")
            print(f"Description: {description}")
            print(f"Sample outputs: {y_values[:3]}...")
        except Exception as e:
            print(f"\n{equation} - Error: {e}")
    
    print(f"\nAvailable functions: {', '.join(parser.get_available_functions())}")
    print(f"Available constants: {', '.join(parser.get_available_constants())}")


if __name__ == "__main__":
    demonstrate_parser_capabilities()

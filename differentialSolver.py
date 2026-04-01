from PIL import Image
import os
from sympy import symbols, Function, Eq, dsolve, simplify, Symbol, Derivative
import sympy
from pix2tex.cli import LatexOCR
from latex2sympy2_extended import latex2sympy
import re
from typing import Union, List

def p2t_ocr(image_path):
    model = LatexOCR()
    file_path = os.path.expanduser(image_path)
    img = Image.open(file_path)
    return model(img)

def convert_separable_to_ode(latex_str):
    """Convert separable equations like y\,dy = x\,dx to standard ODE."""
    # Remove LaTeX spacing commands that break regex
    latex_str_clean = re.sub(r'\\[;,]', '', latex_str)
    
    pattern = r'^\s*(.*?)\s*d\s*y\s*=\s*(.*?)\s*d\s*x\s*$'
    match = re.match(pattern, latex_str_clean)
    if match:
        f_y = match.group(1).strip()
        g_x = match.group(2).strip()
        return rf"\frac{{dy}}{{dx}} = \frac{{{g_x}}}{{{f_y}}}"
    return latex_str

def fix_latex_primes(latex_str):
    # Replace y^{!}, y^{1}, and y' with \frac{dy}{dx}
    latex_str = re.sub(r'y\^{!}|y\^{1}|y\'', r'\\frac{dy}{dx}', latex_str)
    return latex_str

def preprocess_equation_text(text):
    """Clean up OCR output for sympy parsing."""
    # Remove line: text = text.replace("X", "y").replace("x", "y")
    text = text.replace("—", "-").replace("’", "'").replace("‘", "'")
    text = " ".join(text.split())
    text = ''.join(char for char in text if ord(char) < 128)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = text.replace("y'", "Derivative(y(x), x)")
    text = text.replace("y''", "Derivative(y(x), x, x)")
    return text.strip()

def is_trivial_equation(sympy_eq):
    """Check if equation is trivially True/False or has identical sides."""
    if isinstance(sympy_eq, sympy.Equality):
        return sympy_eq.lhs == sympy_eq.rhs
    return False
    
def solve_differential_equation(sympy_eq: Union[sympy.Eq, List[sympy.Eq]], y: sympy.Function) -> Union[sympy.Eq, List[sympy.Eq], str]:
    """Solve differential equation and handle multiple solutions."""
    if is_trivial_equation(sympy_eq):
        return "Error: Equation is trivially true (both sides identical)"
    try:
        if isinstance(sympy_eq, list):
            solutions = [dsolve(eq, y) for eq in sympy_eq]
        else:
            solutions = dsolve(sympy_eq, y)
        
        # Simplify and standardize solutions
        if isinstance(solutions, list):
            return [simplify(sol) for sol in solutions]
        return simplify(solutions)
    except Exception as e:
        return f"Error solving equation: {e}"

if __name__ == "__main__":
    tempPath = '~/Downloads/images.png'
    text = p2t_ocr(tempPath)
    print("Extracted Text:\n", text)
    text = "y' + 2y = -5"  # Example text for testing
    # Fix primes (if any)
    latex_str = fix_latex_primes(text)
    print("LaTeX String:\n", latex_str)
    
    # Convert to standard ODE
    converted_latex = convert_separable_to_ode(latex_str)
    print("Converted LaTeX:", converted_latex)
    
    # Convert to SymPy
    sympy_eq = latex2sympy(converted_latex)
    print("SymPy Equation:", sympy_eq)
    
    # Solve
    x = symbols('x')
    y = Function('y')(x)
    try:
        sympy_eq = sympy.Eq(sympy_eq.lhs, sympy_eq.rhs)  # Ensure it's an Equality
        sol = dsolve(sympy_eq, y)
    except IndexError:
        print("Error: No valid equation found in the OCR output.")
        sol = "No valid equation found."
    except Exception as e:
        print(f"Error solving equation: {e}")
        sol = f"Error: {e}"
    print("Solution:", sol)

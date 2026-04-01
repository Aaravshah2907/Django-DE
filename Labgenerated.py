# math_equation_solver.py
# Version 1.2 - Robust Equation Solver with Free APIs
# Requirements: Python 3.11+, pip install -r requirements.txt

import os
import re
import sys
import logging
from PIL import Image
import numpy as np
import sympy
from sympy import symbols, Function, Eq, dsolve, Derivative,solve, simplify

# Version-controlled imports
try:
    from pix2tex.cli import LatexOCR  # Free OCR (v0.1.4)
    from latex2sympy2_extended import latex2sympy  # Free converter (antlr4 4.13.2)
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EquationProcessor:
    def __init__(self):
        self.ocr_model = LatexOCR()
        self.corrections = {
            r'y\^{!}': 'y\'', r'y\^{1}': 'y\'', r'\\times': ' ',
            r'\\cdot': ' ', r'\\,': '', r'\\;': ''
        }
        
    def preprocess_image(self, img_path):
        """Enhance image quality for better OCR"""
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            return img
        except Exception as e:
            logger.error(f"Image load failed: {e}")
            return None

    def ocr_to_latex(self, img):
        """Convert image to LaTeX with error correction"""
        try:
            latex = self.ocr_model(img)
            for pattern, replacement in self.corrections.items():
                latex = re.sub(pattern, replacement, latex)
            return latex
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    def convert_separable_ode(self, latex_str):
        """Handle equations like y dy = x dx"""
        match = re.match(r'^\s*([\w\^]+)\s*d\s*y\s*=\s*([\w\^]+)\s*d\s*x\s*$', latex_str)
        if match:
            return f"\\frac{{dy}}{{dx}} = \\frac{{{match.group(2)}}}{{{match.group(1)}}}"
        return latex_str

    def latex_to_sympy(self, latex_str):
        """Convert LaTeX to SymPy with error handling"""
        try:
            expr = latex2sympy(latex_str)
            x = symbols('x')
            y = Function('y')(x)
            return expr.subs('y', y) if 'y' in str(expr) else expr
        except Exception as e:
            logger.error(f"LaTeX conversion failed: {e}")
            return None

    def solve_equation(self, sympy_eq):
        """Solve various equation types"""
        try:
            if isinstance(sympy_eq, sympy.Equality):
                if sympy_eq.has(Derivative):
                    return dsolve(sympy_eq)
                return solve(sympy_eq)
            return "Unsolvable equation format"
        except Exception as e:
            logger.error(f"Solving failed: {e}")
            return None

def main(image_path):
    processor = EquationProcessor()
    
    # Processing pipeline
    img = processor.preprocess_image(image_path)
    if not img:
        return f'Error loading image: {image_path}'
    
    latex = processor.ocr_to_latex(img)
    if not latex:
        return f'Error during OCR processing of image: {image_path}'
    
    processed_latex = processor.convert_separable_ode(latex)
    print("\nOriginal LaTeX from OCR:")
    print(latex)
    print("\nProcessed LaTeX after corrections:")
    print(processed_latex)
    sympy_eq = processor.latex_to_sympy(processed_latex)
    print("\nConverted SymPy expression:")
    print(sympy.pretty(sympy_eq) if sympy_eq is not None else "None")
    
    if sympy_eq is not None:
        print("\nProcessed LaTeX:")
        print(processed_latex)
        solutions = processor.solve_equation(sympy_eq)
        print("\nSolutions:")
        print(sympy.pretty(solutions) if solutions is not None else "No solutions found")
    else:
        print("Error converting LaTeX to SymPy expression")
        
if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python math_equation_solver.py <image_path>")
        sys.exit(1)
        
    main(os.path.expanduser('~/Downloads/de8.png'))

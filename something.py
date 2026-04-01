#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Equation Solver from Images
Version 1.0 - June 2025

This program:
1. Takes an image containing a mathematical equation
2. Converts it to LaTeX text using OCR
3. Processes and cleans the LaTeX
4. Converts to SymPy expressions
5. Solves the equation
6. Displays all possible solutions with explanations
"""

# ====================== IMPORT STATEMENTS ======================
# Suppress warnings
import warnings
import os
import sys
import re
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

# Set environment variables to suppress warnings
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Image processing
from PIL import Image
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Basic image handling only.")

# Math processing
try:
    import sympy
    from sympy import (
        symbols, Symbol, Function, Eq, dsolve, solve, simplify, 
        Derivative, integrate, diff, expand, factor, latex,
        sin, cos, tan, exp, log
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Install with: pip install sympy==1.13.1")

# OCR and LaTeX conversion
try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False
    print("Warning: pix2tex not available. Install with: pip install pix2tex[gui]")

try:
    from latex2sympy2_extended import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    try:
        from latex2sympy2 import latex2sympy
        LATEX2SYMPY_AVAILABLE = True
    except ImportError:
        LATEX2SYMPY_AVAILABLE = False
        print("Warning: latex2sympy not available. Install with: pip install latex2sympy2-extended[antlr4_13_2]")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== DATA STRUCTURES ======================
class ProcessingResult:
    """Container for results at each stage of processing."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_ocr = ""
        self.cleaned_latex = ""
        self.preprocessed_latex = ""
        self.converted_latex = ""
        self.sympy_equation = None
        self.equation_type = "unknown"
        self.solutions = []
        self.errors = []
    
    def add_error(self, error):
        """Add an error message."""
        if error not in self.errors:
            self.errors.append(error)
    
    def is_successful(self):
        """Check if processing was successful."""
        return len(self.solutions) > 0 and len(self.errors) == 0

# ====================== IMAGE PROCESSING ======================
class ImageProcessor:
    """Handles image preprocessing for better OCR results."""
    
    @staticmethod
    def load_image(image_path):
        """Load an image from path."""
        try:
            file_path = os.path.expanduser(image_path)
            return Image.open(file_path)
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            return None
    
    @staticmethod
    def preprocess_image(image):
        """Preprocess image for better OCR results."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, skipping preprocessing")
            return image if isinstance(image, Image.Image) else ImageProcessor.load_image(image)
        
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            # Load from path if string
            elif isinstance(image, str):
                img = ImageProcessor.load_image(image)
                if img is None:
                    return None
                img_array = np.array(img)
            # Use directly if numpy array
            else:
                img_array = image
            
            # Convert to grayscale if color
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            
            # Apply adaptive thresholding
            threshold = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            return Image.fromarray(cleaned)
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Fallback to original image
            return image if isinstance(image, Image.Image) else ImageProcessor.load_image(image)

# ====================== OCR ENGINE ======================
class OCREngine:
    """Handles OCR for extracting LaTeX from images."""
    
    def __init__(self, engine='pix2tex'):
        """Initialize the OCR engine."""
        self.engine_name = engine
        self.engine = None
        
        if engine == 'pix2tex':
            if PIX2TEX_AVAILABLE:
                try:
                    self.engine = LatexOCR()
                    logger.info("Initialized pix2tex OCR engine")
                except Exception as e:
                    logger.error(f"Failed to initialize pix2tex OCR engine: {e}")
            else:
                logger.error("pix2tex not available")
    
    def extract_latex(self, image):
        """Extract LaTeX from image."""
        if self.engine is None:
            logger.error("No OCR engine available")
            return ""
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                img = ImageProcessor.load_image(image)
                if img is None:
                    return ""
            else:
                img = image
            
            # Extract LaTeX
            if self.engine_name == 'pix2tex':
                latex_text = self.engine(img)
                return latex_text
            
            return ""
        
        except Exception as e:
            logger.error(f"LaTeX extraction failed: {e}")
            return ""

# ====================== LATEX PROCESSING ======================
class LaTeXProcessor:
    """Processes and cleans LaTeX text."""
    
    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        'y^{!}': "y'",           # Common misread of prime notation
        'y^{1}': "y'",           # Another common misread
        'y^{|}': "y'",           # Vertical bar misread
        'x^{!}': "x'",           # For x derivatives
        'x^{1}': "x'",           # For x derivatives
        '—': '-',                # Em dash to minus
        '–': '-',                # En dash to minus
        ''': "'",                # Smart quote to regular quote
        ''': "'",                # Smart quote to regular quote
    }
    
    @staticmethod
    def clean_latex(latex_text):
        """Clean and normalize LaTeX text."""
        if not latex_text:
            return ""
        
        # Apply corrections
        cleaned = latex_text
        for error, correction in LaTeXProcessor.OCR_CORRECTIONS.items():
            cleaned = cleaned.replace(error, correction)
        
        # Remove LaTeX spacing commands
        cleaned = cleaned.replace('\\;', '').replace('\\,', '')
        
        # Remove extra spaces and newlines
        cleaned = " ".join(cleaned.split())
        
        return cleaned.strip()
    
    @staticmethod
    def fix_derivative_notation(latex_str):
        """Convert various derivative notations to standard form."""
        if not latex_str:
            return ""
        
        # Convert prime notation to fraction form for first derivatives
        latex_str = latex_str.replace("y'", "\\frac{dy}{dx}")
        latex_str = latex_str.replace("y''", "\\frac{d^2y}{dx^2}")
        
        # Handle higher order derivatives with regex
        latex_str = re.sub(r"y\^{(\d+)}", r"\\frac{d^{\1}y}{dx^{\1}}", latex_str)
        
        return latex_str
    
    @staticmethod
    def convert_separable_ode(latex_str):
        """
        Convert separable differential equations (f(y) dy = g(x) dx)
        to standard form (dy/dx = g(x)/f(y)).
        """
        if not latex_str:
            return ""
        
        # Remove LaTeX spacing that might interfere with regex
        cleaned = latex_str.replace('\\;', '').replace('\\,', '')
        
        # Match pattern for separable equations
        pattern = r'^\s*(.*?)\s*d\s*y\s*=\s*(.*?)\s*d\s*x\s*$'
        match = re.match(pattern, cleaned)
        
        if match:
            f_y = match.group(1).strip()
            g_x = match.group(2).strip()
            return f"\\frac{{dy}}{{dx}} = \\frac{{{g_x}}}{{{f_y}}}"
        
        return latex_str

# ====================== EQUATION CONVERSION ======================
class EquationConverter:
    """Converts LaTeX equations to SymPy expressions."""
    
    @staticmethod
    def latex_to_sympy(latex_str):
        """Convert LaTeX string to SymPy expression."""
        if not LATEX2SYMPY_AVAILABLE:
            logger.error("latex2sympy not available")
            return None
        
        if not latex_str:
            logger.error("Empty LaTeX string")
            return None
        
        try:
            # Convert LaTeX to SymPy
            sympy_expr = latex2sympy(latex_str)
            
            # Ensure it's an equation
            if not isinstance(sympy_expr, sympy.Equality):
                # If just an expression, assume it equals 0
                x = symbols('x')
                y = Symbol('y')
                sympy_expr = Eq(sympy_expr, 0)
            
            return sympy_expr
            
        except Exception as e:
            logger.error(f"LaTeX to SymPy conversion failed: {e}")
            return None
    
    @staticmethod
    def determine_equation_type(eq):
        """Determine the type of equation."""
        if eq is None:
            return "unknown"
        
        try:
            # Check if it's a differential equation
            if eq.has(Derivative):
                return "differential"
            
            # Check for specific patterns in algebraic equations
            if len(eq.free_symbols) == 0:
                return "numeric"  # Just numbers, no variables
            elif len(eq.free_symbols) == 1:
                return "algebraic_single_var"  # Single variable algebraic
            else:
                return "algebraic_multi_var"  # Multiple variable algebraic
                
        except Exception as e:
            logger.error(f"Error determining equation type: {e}")
            return "unknown"

# ====================== EQUATION SOLVER ======================
class EquationSolver:
    """Solves different types of equations."""
    
    @staticmethod
    def is_trivial_equation(eq):
        """Check if the equation is trivially true or false."""
        if eq is None:
            return False
            
        if isinstance(eq, sympy.Equality):
            try:
                return eq.lhs == eq.rhs
            except:
                return False
                
        return False
    
    @staticmethod
    def solve_differential_equation(eq):
        """Solve a differential equation."""
        if eq is None:
            return ["Error: Invalid equation"]
        
        # Check for trivial equation
        if EquationSolver.is_trivial_equation(eq):
            return ["Trivial equation: both sides are identical"]
        
        try:
            # Define common variables
            x = symbols('x')
            y = Function('y')(x)
            
            # Ensure y is a function if needed
            if 'y' in str(eq) and not eq.has(Function('y')):
                eq = eq.subs(Symbol('y'), y)
            
            # Solve the equation
            try:
                solution = dsolve(eq, y)
                
                # Format solution(s)
                if isinstance(solution, list):
                    return [str(simplify(sol)) for sol in solution]
                else:
                    return [str(simplify(solution))]
                    
            except Exception as e:
                # Handle special cases for differential equations
                try:
                    # Try again with explicitly defining y as function
                    x = symbols('x')
                    y = Function('y')
                    eq_str = str(eq)
                    
                    # Check for special patterns
                    if "Derivative" in eq_str and "=" in eq_str:
                        # Extract parts manually if needed
                        if isinstance(eq, sympy.Equality):
                            lhs = eq.lhs
                            rhs = eq.rhs
                            
                            # Try manual ODE solving for dy/dx = f(x,y)
                            if lhs == Derivative(y(x), x):
                                # Form: dy/dx = f(x,y)
                                # For separable equations like dy/dx = g(x)/f(y)
                                # Check if rhs is a quotient
                                try:
                                    if rhs.is_Mul and 1/y(x) in rhs.as_ordered_factors():
                                        # It might be a separable equation
                                        # But just return error message for now
                                        return [f"Special differential equation detected, but couldn't solve: {e}"]
                                except:
                                    pass
                    
                    # If all attempts fail, return error
                    return [f"Error solving differential equation: {e}"]
                    
                except Exception as e2:
                    return [f"Error solving differential equation: {e}. Additional error: {e2}"]
                
        except Exception as e:
            return [f"Error in differential equation solver: {e}"]
    
    @staticmethod
    def solve_algebraic_equation(eq):
        """Solve an algebraic equation."""
        if eq is None:
            return ["Error: Invalid equation"]
        
        # Check for trivial equation
        if EquationSolver.is_trivial_equation(eq):
            return ["Trivial equation: both sides are identical"]
        
        try:
            # Get all symbols
            symbols_in_eq = list(eq.free_symbols)
            
            if not symbols_in_eq:
                # No variables - evaluate the equation
                result = eq.lhs - eq.rhs
                try:
                    result_eval = float(result)
                    if abs(result_eval) < 1e-10:  # Close to zero
                        return ["True (equation is satisfied for all values)"]
                    else:
                        return ["False (equation has no solution)"]
                except:
                    return [f"Could not evaluate: {result}"]
            
            # Solve for each symbol
            solutions = {}
            for sym in symbols_in_eq:
                try:
                    sol = solve(eq, sym)
                    if sol:
                        solutions[str(sym)] = [str(simplify(s)) for s in sol]
                except Exception:
                    continue
            
            # Format the solutions
            if solutions:
                result = []
                for var, sols in solutions.items():
                    for sol in sols:
                        result.append(f"{var} = {sol}")
                return result
            else:
                return ["Could not solve the equation algebraically"]
                
        except Exception as e:
            return [f"Error solving algebraic equation: {e}"]

# ====================== MAIN SOLVER CLASS ======================
class MathEquationSolver:
    """
    Complete pipeline for solving mathematical equations from images.
    
    Workflow:
    1. Load and preprocess image
    2. Extract LaTeX using OCR
    3. Clean and fix LaTeX notation
    4. Convert to standard form if needed
    5. Convert to SymPy
    6. Solve equation
    7. Format and return results
    """
    
    def __init__(self, ocr_engine='pix2tex'):
        """Initialize the solver with components."""
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine(ocr_engine)
        self.latex_processor = LaTeXProcessor()
        self.equation_converter = EquationConverter()
        self.equation_solver = EquationSolver()
    
    def process_image(self, image_path, verbose=False):
        """
        Process an image containing a mathematical equation and solve it.
        Returns a ProcessingResult object with all intermediate results and final solutions.
        """
        # Initialize result container
        result = ProcessingResult(image_path)
        
        try:
            # Step 1: Preprocess image (optional for pix2tex which has its own preprocessing)
            if verbose:
                print("Step 1: Preprocessing image...")
            
            img = self.image_processor.load_image(image_path)
            if img is None:
                result.add_error("Failed to load image")
                return result
            
            # Step 2: Extract LaTeX from image
            if verbose:
                print("Step 2: Extracting LaTeX from image...")
            
            raw_latex = self.ocr_engine.extract_latex(img)
            result.raw_ocr = raw_latex
            
            if not raw_latex:
                result.add_error("Failed to extract LaTeX from image")
                return result
            
            if verbose:
                print(f"Extracted LaTeX: {raw_latex}")
            
            # Step 3: Clean and normalize LaTeX
            if verbose:
                print("Step 3: Cleaning LaTeX...")
            
            cleaned_latex = self.latex_processor.clean_latex(raw_latex)
            result.cleaned_latex = cleaned_latex
            
            if verbose:
                print(f"Cleaned LaTeX: {cleaned_latex}")
            
            # Step 4: Fix derivative notation
            if verbose:
                print("Step 4: Processing equation notation...")
            
            preprocessed_latex = self.latex_processor.fix_derivative_notation(cleaned_latex)
            result.preprocessed_latex = preprocessed_latex
            
            # Step 5: Convert to standard form if needed
            converted_latex = self.latex_processor.convert_separable_ode(preprocessed_latex)
            result.converted_latex = converted_latex
            
            if verbose:
                print(f"Final LaTeX: {converted_latex}")
            
            # Step 6: Convert to SymPy
            if verbose:
                print("Step 6: Converting to SymPy...")
            
            sympy_eq = self.equation_converter.latex_to_sympy(converted_latex)
            result.sympy_equation = sympy_eq
            
            if sympy_eq is None:
                result.add_error("Failed to convert to SymPy expression")
                return result
            
            if verbose:
                print(f"SymPy equation: {sympy_eq}")
            
            # Step 7: Determine equation type
            eq_type = self.equation_converter.determine_equation_type(sympy_eq)
            result.equation_type = eq_type
            
            if verbose:
                print(f"Equation type: {eq_type}")
            
            # Step 8: Solve equation based on type
            if verbose:
                print("Step 8: Solving equation...")
            
            if eq_type == "differential":
                solutions = self.equation_solver.solve_differential_equation(sympy_eq)
            elif eq_type.startswith("algebraic"):
                solutions = self.equation_solver.solve_algebraic_equation(sympy_eq)
            else:
                solutions = ["Unknown equation type"]
                result.add_error(f"Unsupported equation type: {eq_type}")
            
            result.solutions = solutions
            
            if verbose:
                print(f"Solutions: {solutions}")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            result.add_error(error_msg)
            logger.error(error_msg)
        
        return result

# ====================== CLI INTERFACE ======================
def main():
    """Command line interface for the equation solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve mathematical equations from images")
    parser.add_argument("image_path", nargs="?", help="Path to the image file containing an equation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip image preprocessing")
    
    args = parser.parse_args()
    
    # If no arguments provided, ask for image path
    if args.image_path is None:
        args.image_path = input("Enter image file path: ")
    
    # Initialize solver
    solver = MathEquationSolver(ocr_engine='pix2tex')
    
    # Process image
    print(f"Processing image: {args.image_path}")
    print("=" * 50)
    
    result = solver.process_image(args.image_path, verbose=args.verbose)
    
    # Display results
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    
    if result.errors:
        print("ERRORS:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.solutions:
        print("\nSOLUTIONS:")
        for i, solution in enumerate(result.solutions, 1):
            print(f"  {i}. {solution}")
    else:
        print("\nNo solutions found.")
    
    print("\nPROCESSING SUMMARY:")
    print(f"  Extracted text: {result.raw_ocr}")
    print(f"  Cleaned LaTeX: {result.cleaned_latex}")
    print(f"  Final LaTeX: {result.converted_latex}")
    if result.sympy_equation:
        print(f"  SymPy equation: {result.sympy_equation}")
    print(f"  Equation type: {result.equation_type}")

# ====================== SCRIPT EXECUTION ======================
if __name__ == "__main__":
    # Check if required packages are available
    missing_packages = []
    if not PIX2TEX_AVAILABLE:
        missing_packages.append("pix2tex[gui]")
    if not LATEX2SYMPY_AVAILABLE:
        missing_packages.append("latex2sympy2-extended[antlr4_13_2]")
    if not CV2_AVAILABLE:
        missing_packages.append("opencv-python")
    if not SYMPY_AVAILABLE:
        missing_packages.append("sympy")
    
    if missing_packages:
        print("Warning: Some required packages are missing. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Continue with limited functionality? (Y/n)")
        choice = input().strip().lower()
        if choice and choice != 'y':
            sys.exit(1)
    
    main()
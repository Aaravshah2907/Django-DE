from sympy import symbols, Function, Eq, dsolve, Derivative
from sympy.core.numbers import equal_valued  # Should no longer throw an error
print("Success!")

x = symbols('x')
y = Function('y')(x)
ode = Eq(Derivative(y, x), x/y)
sol = dsolve(ode, y)
print(sol)

from sympy import Eq, I, Matrix, cos, pprint, sin, solve, symbols

# Define symbolic variables
q1, q2, q3 = symbols("q1 q2 q3")

# Define symbolic matrices
N = symbols("N")
A = Matrix([[N, -q3, 0], [q3, N, 0], [0, 0, 1]])

x = Matrix([[cos(q2)], [sin(q2)], [q1]])

p = Matrix([0, 2, 1.5])

# Set up the equation
equation = Eq(A * x, p)

# Solve for p
solutions = solve((equation, Eq(q3, 1.5)), (q1, q2, q3))

pprint(solutions)

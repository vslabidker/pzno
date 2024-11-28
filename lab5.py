import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, y, z, t, a, b = sp.symbols('x y z t a b')

# Task 1
expr1 = sp.root((1 - 2*a + a**2)*(a**2 - 1)*(a - 1), 4) * (a**2 + 2*a - 3) / sp.sqrt(a + 1)
simplified_expr1 = sp.simplify(expr1)
result1 = simplified_expr1.subs(a, 3)
print("Task 1:", result1)

# Task 2
equation2 = sp.Eq(sp.sin(x)**4 - sp.cos(x)**4, 0.5)
solutions2 = sp.solve(equation2, x)
print("Task 2:", solutions2)

# Task 3
system3 = [
    sp.Eq(x + y*z, 2),
    sp.Eq(y + z*x, 2),
    sp.Eq(z + x*y, 2)
]
solutions3 = sp.solve(system3, [x, y, z])
print("Task 3:", solutions3)

# Task 4
alpha, beta = sp.symbols('alpha beta')
expr4 = (sp.sqrt(1 + alpha*x) - sp.sqrt(1 + beta*x)) / x
limit4 = sp.limit(expr4, x, 0)
print("Task 4:", limit4)

# Task 5

expr5 = sp.ln(sp.pi*x/8) / (x - 4)
right_limit = sp.limit(expr5, x, 4, dir='+')
print("Task 5:", right_limit)

# Task 6
f6 = x**2 - x*y + y**2
grad_f6 = sp.Matrix([sp.diff(f6, x), sp.diff(f6, y)])
M6 = {x: 1, y: 1}
grad_at_M6 = grad_f6.subs(M6)
direction6 = sp.Matrix([-3, 2])
directional_derivative6 = grad_at_M6.dot(direction6.normalized())
print("Task 6: Gradient at M:", grad_at_M6, "Directional derivative:", directional_derivative6)

# Task 7
f7 = sp.atan((x + y) / (1 - x*y))
f7_x = sp.diff(f7, x)
f7_y = sp.diff(f7, y)
f7_xx = sp.diff(f7_x, x)
f7_yy = sp.diff(f7_y, y)
f7_xy = sp.diff(f7_x, y)
print(f"Task 7: Partial derivatives: \nf`x = {sp.simplify(f7_x)},\nf`y = {sp.simplify(f7_y)}, "
      f"\nf``xx = {sp.simplify(f7_xx)}, \nf`yy = {sp.simplify(f7_yy)}, \nf`xy = {sp.simplify(f7_xy)}")


# Task 8
y8 = sp.Function('y')(x)
diffeq = sp.Eq(y8.diff(x, 2) + 2*y8.diff(x) + y8, sp.exp(x) * sp.sqrt(x + 1))
general_solution = sp.dsolve(diffeq)
print("Task 8:", general_solution)

# Task 9
u = 1 / (2 * a * sp.sqrt(sp.pi * t)) * sp.exp(-(x - b)**2 / (4 * a**2 * t))
du_dt = sp.diff(u, t)
d2u_dx2 = sp.diff(u, x, 2)
heat_equation_check = sp.simplify(du_dt - a**2 * d2u_dx2)
print("Task 9: Heat equation check:", heat_equation_check == 0)

# Task 10
f10 = (x**3 - 6) / (x**4 + 6*x**2 + 8)
antiderivative10 = sp.integrate(f10, x)
check10 = sp.diff(antiderivative10, x) - f10
print("Task 10: Antiderivative:", antiderivative10, "Check:", check10.simplify() == 0)

# Task 11
definite_integral11 = sp.integrate(x**3 * sp.cos(x), (x, 0, sp.pi/2))
print("Task 11:", definite_integral11)

# Task 12
u_vals = np.linspace(0, 2*np.pi, 100)
v_vals = np.linspace(0, 2*np.pi, 100)
u_vals, v_vals = np.meshgrid(u_vals, v_vals)

# Первая поверхность
x1 = 4 + (3 + np.cos(v_vals)) * np.sin(u_vals)
y1 = 4 + (3 + np.cos(v_vals)) * np.cos(u_vals)
z1 = 4 + np.sin(v_vals)

# Вторая поверхность
x2 = 8 + (3 + np.cos(v_vals)) * np.cos(u_vals)
y2 = 3 + np.sin(v_vals)
z2 = 4 + (3 + np.cos(v_vals)) * np.sin(u_vals)

# Построение графика
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Первая поверхность
ax.plot_surface(x1, y1, z1, cmap='viridis', alpha=0.7, edgecolor='none')

# Вторая поверхность
ax.plot_surface(x2, y2, z2, cmap='plasma', alpha=0.7, edgecolor='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Task 12: Two surfaces")
plt.show()

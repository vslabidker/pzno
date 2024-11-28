import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.linalg import solve, lu, lstsq, qr, svd, norm, eig
from scipy.optimize import lsq_linear, minimize_scalar, fminbound, brent
from scipy.interpolate import lagrange, interp1d, CubicSpline

# Task 1
f1 = lambda x: np.log(x)
result10, _ = sp.integrate.quad(f1, 0, 1)
result11, _ = sp.integrate.fixed_quad(f1,0,1)
result12, _ = sp.integrate.quadrature(f1,0,1)

print("Задача 1.0. Результат несобственного интеграла:", result10)
print("Задача 1.1. Результат несобственного интеграла:", result11)
print("Задача 1.2. Результат несобственного интеграла:", result12)

#Task 2

f2 = lambda x, y: x * np.sin(y)
result2, _ = sp.integrate.dblquad(f2, 0, np.pi/6, lambda x: x/2, lambda x: x + np.pi/6)
print("Задача 2. Результат двойного интеграла:", result2)

#Task 3

f3 = lambda x: np.sin(x)
x0 = np.pi
first_derivative = sp.misc.derivative(f3, x0, dx=1e-5)
second_derivative = sp.misc.derivative(f3, x0, dx=1e-5, n=2)
print(f"Задача 3.0 Первая производная в π: {first_derivative}")
print(f"Задача 3.0 Вторая производная в π: {second_derivative}")

# Метод 2: Разностная формула
dx = 1e-6
first_derivative_diff = (f3(x0 + dx) - f3(x0 - dx)) / (2 * dx)
second_derivative_diff = (f3(x0 + dx) - 2 * np.sin(x0) + f3(x0 - dx)) / (dx**2)

print("\nМетод 2 (разностная формула):")
print(f"Задача 3.1 Первая производная в π: {first_derivative_diff}")
print(f"Задача 3.1 Вторая производная в π: {second_derivative_diff}")

#Task 4

# Задаем уравнение третьего порядка как систему уравнений первого порядка
# Система уравнений
def system(z, t):
    z1, z2, z3 = z  # z1 = y, z2 = y', z3 = y''
    dz1dt = z2
    dz2dt = z3
    dz3dt = 1 - 2*z3 - 5*z2
    return [dz1dt, dz2dt, dz3dt]

# Начальные условия: y(0) = 1, y'(0) = 3, y''(0) = -2
initial_conditions = [1, 3, -2]

# Время для решения задачи
t = np.linspace(0, 10, 100)

# Решение системы
solution = odeint(system, initial_conditions, t)

# Извлекаем результаты для y(t), y'(t), y''(t)
y = solution[:, 0]
y_prime = solution[:, 1]
y_double_prime = solution[:, 2]

# Построение графиков
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, y, label="y(t)")
plt.title("Решение для y(t)")
plt.xlabel("t")
plt.ylabel("y(t)")

plt.subplot(3, 1, 2)
plt.plot(t, y_prime, label="y'(t)", color="orange")
plt.title("Решение для y'(t)")
plt.xlabel("t")
plt.ylabel("y'(t)")

plt.subplot(3, 1, 3)
plt.plot(t, y_double_prime, label="y''(t)", color="green")
plt.title("Решение для y''(t)")
plt.xlabel("t")
plt.ylabel("y''(t)")

plt.tight_layout()
plt.show()

#Task 5

# Матрица A и вектор b
A = np.array([[1, -2, 3],
              [0, -4, 1],
              [-2, 5, 1]])
b = np.array([1, 2, 3])

# Метод 1: Используем scipy.linalg.solve
x1 = solve(A, b)
print("Задача 5.0 Решение через scipy.linalg.solve:")
print(x1)

# Метод 2: Решение через LU-разложение
P, L, U = lu(A)  # P - перестановочная матрица, L - нижняя треугольная, U - верхняя треугольная
y = np.linalg.solve(L, np.dot(P.T, b))  # Решаем L*y = P^T*b
x2 = np.linalg.solve(U, y)  # Решаем U*x = y
print("\nЗадача 5.1 Решение через LU-разложение:")
print(x2)

# Метод 3: Решение через задачу наименьших квадратов
x3, residuals, rank, s = lstsq(A, b)
print("\nЗадача 5.2 Решение через задачу наименьших квадратов:")
print(x3)


# Task 6

# Матрица A и вектор b
A = np.array([[1, 2],
              [-3, 0],
              [4, -1]])
b = np.array([1, 1, 1])

# Метод 1: Линейная задача наименьших квадратов
x1, residuals, rank, s = lstsq(A, b)
print("Задача 6.0 Решение через задачу наименьших квадратов (lstsq):")
print(x1)

# Метод 2: QR-разложение
Q, R = qr(A, mode='economic')  # QR-разложение матрицы A
x2 = np.linalg.solve(R, Q.T @ b)  # Решаем R*x = Q^T*b
print("\nЗадача 6.1 Решение через QR-разложение:")
print(x2)

# Метод 3: SVD-разложение
U, Sigma, Vt = svd(A)  # SVD-разложение A = U * Sigma * Vt
Sigma_pinv = np.zeros((A.shape[1], A.shape[0]))
Sigma_pinv[:len(Sigma), :len(Sigma)] = np.diag(1 / Sigma)  # Псевдообратная Sigma
A_pinv = Vt.T @ Sigma_pinv @ U.T  # Псевдообратная A через SVD
x3 = A_pinv @ b
print("\nЗадача 6.2 Решение через SVD-разложение:")
print(x3)

# Метод 4: lsq_linear из scipy.optimize
result = lsq_linear(A, b)
x4 = result.x
print("\nЗадача 6.3 Решение через lsq_linear:")
print(x4)

# Task 7
A = np.array([[1, -2, 3],
             [0, -4, 1],
             [-2, 5, 1]])

spectral_norm = norm(A, 2)
print("Задача 7.0 Спектральная норма матрицы A встроенным методом:", spectral_norm)
eigenvalues, eigenvectors = eig(A.T @ A)
print("Задача 7.1 Спектральная норма матрицы по определению:", np.sqrt(max(eigenvalues)))

# Task 8
f8 = lambda x: (x - 4)**2 + (x + 1)**2
# Способ 1: minimize_scalar
result1 = minimize_scalar(f8)
x_min1 = result1.x
print("Задача 8.0 Минимум через minimize_scalar:")
print(f"x = {x_min1}, f(x) = {f8(x_min1)}")

# Способ 2: fminbound (на интервале [-10, 10])
x_min2 = fminbound(f8, -10, 10)
print("\nЗадача 8.1 Минимум через fminbound:")
print(f"x = {x_min2}, f(x) = {f8(x_min2)}")

# Способ 3: brent
x_min3 = brent(f8)
print("\nЗадача 8.2 Минимум через brent:")
print(f"x = {x_min3}, f(x) = {f8(x_min3)}")

# Построение графика
x = np.linspace(-10, 10, 500)
y = f8(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="f(x) = (x-4)^2 + (x+1)^2")
plt.scatter([x_min1, x_min2, x_min3], [f8(x_min1), f8(x_min2), f8(x_min3)],
            color='red', label='Найденный минимум', zorder=5)
plt.axvline(x_min1, color='red', linestyle='--', alpha=0.7, label='x_min')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("График функции и найденный минимум")
plt.legend()
plt.grid()
plt.show()

# Task 9
x = np.array([0, 1, 2, 3, 4, 5])
f = np.array([-2, 6, 8, 0, 1, 4])

# 1. Интерполяционный многочлен Лагранжа
lagrange_poly = lagrange(x, f)

# 2. Линейный сплайн (1-й степени)
linear_spline = interp1d(x, f, kind='linear')

# 3. Кубический сплайн (3-й степени)
cubic_spline = CubicSpline(x, f)

# Точки для построения графиков
x_dense = np.linspace(min(x), max(x), 500)

# Вычисление значений интерполяций
lagrange_values = lagrange_poly(x_dense)
linear_values = linear_spline(x_dense)
cubic_values = cubic_spline(x_dense)

# Построение графиков
plt.figure(figsize=(12, 6))

# График исходных данных
plt.scatter(x, f, color='red', label='Точки данных', zorder=5)

# График многочлена Лагранжа
plt.plot(x_dense, lagrange_values, label='Многочлен Лагранжа', color='blue')

# График линейного сплайна
plt.plot(x_dense, linear_values, label='Линейный сплайн', color='green', linestyle='--')

# График кубического сплайна
plt.plot(x_dense, cubic_values, label='Кубический сплайн', color='purple', linestyle='-.')

# Настройка графика
plt.title("Интерполяция табличной функции")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

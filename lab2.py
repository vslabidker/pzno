import numpy as np

##Task 1:
# Создадим 10 случайных точек
points = np.random.randint(-10, 11, size=(10, 2))

# a. Найти точку, наиболее удалённую от начала координат

# Вычисляем расстояние от начала координат для каждой точки
distances = np.linalg.norm(points, axis=1)

# Находим индекс точки с максимальным расстоянием
farthest_point = points[np.argmax(distances)]

# b. Отсортировать точки в порядке возрастания длин векторов
sorted_points = points[np.argsort(distances)]

# c. Описать функцию v2normalize(x), которая нормализует вектор
def v2normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

# Применим нормализацию к каждой точке (вектору)
normalized_vectors = np.apply_along_axis(v2normalize, 1, points)

# d. Профильтровать массив нормированных векторов, оставив только векторы с положительными координатами
positive_vectors = normalized_vectors[(normalized_vectors > 0).all(axis=1)]

# Вывод результатов
print("Исходные точки:\n", points)
print("Наиболее удалённая точка:\n", farthest_point)
print("Точки, отсортированные по возрастанию длин векторов:\n", sorted_points)
print("Нормированные векторы:\n", normalized_vectors)
print("Векторы с положительными координатами:\n", positive_vectors)

##Task 2

# Первая строка: числа от 6 до 1
row_1 = np.arange(6, 0, -1)

# Вторая строка: числа от -6 до -1
row_2 = np.arange(-6, 0, 1)

# Третья и четвертая строки: заполнены числом 2
row_3_4 = np.full((2, 6), 2)

# Пятая и шестая строки: заполнены числом 4
row_5_6 = np.full((2, 6), 4)

# Создание блочной матрицы
matrix = np.vstack([row_1, row_2, row_3_4, row_5_6])

# Поэлементное возведение в куб
matrix_cubed = np.power(matrix, 3)

# Поиск минимального значения в матрице кубов
min_value = np.min(matrix_cubed)

# Вывод результатов
# print("Блочная матрица:\n", matrix)
# print("Матрица кубов:\n", matrix_cubed)
print("Минимальное значение (куб):", min_value)

##Task 3

def is_positive_definite(matrix):
    # Размерность матрицы
    n = matrix.shape[0]

    # Проверка, что все главные миноры положительны
    for i in range(1, n + 1):
        minor = matrix[:i, :i]
        det_minor = np.linalg.det(minor)
        if det_minor <= 0:
            return False
    return True

A = np.array([[14, 6, 3],
              [6, 9, -4],
              [3, -4, 9]])

B = np.array([[-2, 8, 8, 4, -3],
              [3, 0, -1, 3, 0],
              [7, -2, -4, 6, 2],
              [7, -5, 1, -5, -3],
              [-2, -5, 7, 6, -1]])


print("Матрица A положительно определённая:", is_positive_definite(A))
print("Матрица B положительно определённая:", is_positive_definite(B))

##Task 4

points = np.random.rand(10, 2) * 10

# Вычисляем расстояния от первой точки до всех остальных
distance = np.max(np.abs(points[0] - points[1:]), axis=1)

# Находим индекс ближайшей точки к складу
closest = np.argmin(distance) + 1  # +1, так как пропускаем 0-й элемент

print("Координаты точек:\n", points)
print("\nРасстояния от первой точки (склада) до остальных:\n", distance)
print(f"\nНаиболее близкая точка к складу: {points[closest]}")

##Task 5

matrix = np.genfromtxt('input.csv', delimiter=',')

# Вычисляем сумму всех элементов матрицы
total_sum = np.sum(matrix)

# Заменяем элемент с индексами [0, 0] на сумму
matrix[0, 0] = total_sum

# Записываем результирующую матрицу в CSV файл
np.savetxt('output.csv', matrix, delimiter=',', fmt='%.0f')

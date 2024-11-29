import numpy as np
import matplotlib.pyplot as plt

# Функція для диференціального рівняння
def f(x, y):
    return 1 / (2 * x - y**2)

# Метод Ейлера
def euler_method(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    while x < x_end:
        y += h * f(x, y)
        x = min(x + h, x_end)  # Учитываем точное достижение x_end
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# Метод Рунге-Кутта 4-го порядку
def runge_kutta_4(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# Початкові умови та параметри
x0, y0 = 1, 1
x_end = 2
h_values = [0.1, 0.2]
fine_h = 0.001  # Тонкий шаг для приближенного "аналитического" решения

# Решение методом Рунге-Кутта с малым шагом
x_fine, y_fine = runge_kutta_4(f, x0, y0, fine_h, x_end)

# Численные решения с разными шагами
solutions = {}
for h in h_values:
    solutions[f"Euler_h={h}"] = euler_method(f, x0, y0, h, x_end)
    solutions[f"RK4_h={h}"] = runge_kutta_4(f, x0, y0, h, x_end)

# Построение графиков
plt.figure(figsize=(10,8 ))
plt.plot(x_fine, y_fine, label="Approx. Analytical (RK4, h=0.001)", color="black", linewidth=2)
for label, (x_vals, y_vals) in solutions.items():
    plt.plot(x_vals, y_vals, label=label, marker='o')  # Маркеры для наглядности

# Настройка графика
plt.title("Solution of ODE using Euler and RK4 Methods")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(1, 1.5)  # Установка границ для оси x
plt.ylim(1, 2)
plt.legend()
plt.grid(True)
plt.show()

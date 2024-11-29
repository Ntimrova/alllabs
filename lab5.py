import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функція y(x) та її аналітична похідна
def y(x):
    return x**2 + 1 / np.tan(2 * x)

def y_derivative(x):
    return 2 * x - 2 / (np.sin(2 * x)**2)

# Чисельні методи для першої та другої похідних
def forward_difference(x, dx):
    return (y(x + dx) - y(x)) / dx

def backward_difference(x, dx):
    return (y(x) - y(x - dx)) / dx

def central_difference(x, dx):
    return (y(x + dx) - y(x - dx)) / (2 * dx)

def second_derivative(x, dx):
    return (y(x + dx) - 2 * y(x) + y(x - dx)) / dx**2

# Параметри
x0 = np.pi / 4  # Вибрана точка
dx_values = [0.5, 0.2, 0.1, 0.01, 0.001]

# Обчислення похідних
results = []
for dx in dx_values:
    yx = y(x0)
    yx_analytic = y_derivative(x0)
    yx_forward = forward_difference(x0, dx)
    yx_backward = backward_difference(x0, dx)
    yx_central = central_difference(x0, dx)
    results.append([dx, yx, yx_analytic, yx_forward, yx_backward, yx_central])

# Таблиця результатів
columns = ["Δx", "y(x)", "y'(x) аналітично", "(y(x+Δx)-y(x))/Δx", "(y(x)-y(x-Δx))/Δx", "(y(x+Δx)-y(x-Δx))/2Δx"]
df = pd.DataFrame(results, columns=columns)
pd.set_option('display.max_columns', None)
print(df)

# Інтервал для аналізу
x_values = np.linspace(0.1, np.pi/2 - 0.1, 500)
dx = 0.01

# Обчислення для графіків
y_values = y(x_values)
y_prime_analytic = y_derivative(x_values)
y_prime_numeric = central_difference(x_values, dx)
y_double_prime = second_derivative(x_values, dx)

# Пошук екстремумів та точок перегину
extrema = x_values[np.isclose(y_prime_numeric, 0, atol=1e-2)]
inflection_points = x_values[np.isclose(np.gradient(y_double_prime, dx), 0, atol=1e-2)]

# Побудова графіків
plt.figure(figsize=(12, 8))

# Графік функції
plt.subplot(3, 1, 1)
plt.plot(x_values, y_values, label="y(x)", color="blue")
plt.scatter(extrema, y(extrema), color="red", label="Extrema", zorder=5)
plt.title("Функція y(x)")
plt.legend()
plt.grid()

# Графік першої похідної
plt.subplot(3, 1, 2)
plt.plot(x_values, y_prime_analytic, label="y'(x) аналітично", color="green")
plt.plot(x_values, y_prime_numeric, label="y'(x) чисельно", linestyle="--", color="orange")
plt.scatter(extrema, y_derivative(extrema), color="red", label="Extrema", zorder=5)
plt.title("Перша похідна y'(x)")
plt.legend()
plt.grid()

# Графік другої похідної
plt.subplot(3, 1, 3)
plt.plot(x_values, y_double_prime, label="y''(x) чисельно", color="purple")
plt.scatter(inflection_points, second_derivative(inflection_points, dx), color="black", label="Inflection Points", zorder=5)
plt.title("Друга похідна y''(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

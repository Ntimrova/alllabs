import numpy as np
import matplotlib.pyplot as plt

# Дані (вар 7)
x = np.array([0.4, 0.5, 0.8, 1, 1.5, 2, 2.5, 3])
y = np.array([1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4])

# Інтервал для інтерполяції
xmin, xmax = 0.4, 3
h = (xmax - xmin) / 10  # Розбиваємо інтервал [xmin,xmax] на 10 рівних частин
x_interp = np.array([xmin + i * h for i in range(11)])  # Вузли інтерполяції

# Функція для обчислення полінома Лагранжа
def lagrange_interpolation(x_points, y_points, x_value):
    #Призначення функції: обчислює значення полінома Лагранжа L(x) в точці x value
    # x_points масив вузлів інтерполяції - x0,x1,...xn
    # y_ponts - масив значень функції y0,y1,...yn у відповідних вузлах
    #x_value - точка х в якій потрібно обчислити  L(x)

    n = len(x_points)  # кількість точок  xi це визначає кількість членів у сумі формули Лагранжа

    result = 0 #змінна для накопичення значення полінома L(x) в точці x value

    for j in range(n): # Перебираємо всі вузли xj
        # Обчислення базисного полінома
        lj = 1
        for i in range(n):
            if i != j:
                lj *= (x_value - x_points[i]) / (x_points[j] - x_points[i])
        result += y_points[j] * lj
    return result

# Обчислення значень полінома у вузлах інтерполяції
y_interp = [lagrange_interpolation(x, y, xi) for xi in x_interp]

# Побудова полінома Лагранжа на всьому інтервалі
x_plot = np.linspace(xmin, xmax, 500)  # Щільні точки для побудови графіка
y_plot = [lagrange_interpolation(x, y, xi) for xi in x_plot]

# Графік
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Експериментальні точки')
plt.scatter(x_interp, y_interp, color='green', label='Вузли інтерполяції')
plt.plot(x_plot, y_plot, color='red', label='Поліном Лагранжа')
plt.title("Інтерполяція поліномом Лагранжа")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Вивід результатів
print("Вузли інтерполяції та значення функції:")
for xi, yi in zip(x_interp, y_interp):
    print(f"x = {xi:.3f}, y = {yi:.3f}")

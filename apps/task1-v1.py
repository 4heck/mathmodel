import math

import eiler
import matplotlib.pyplot as plt
import numpy as np
import rungekutt
from scipy.integrate import odeint


# численный метод
def model(x, t, hl):
    k = math.log1p(2 - 1) / hl
    dxdt = -k * x
    return dxdt


# конечная функция
def gamma(t, m0, hl):
    return m0 * (2 ** (-t / hl))


# начальная масса радия
x0 = 100

# период полураспада
hl = 1590

# начальное время
t_min = 0

# конечное время
t_max = 3180

# сколько чисел выводится
step_count = 30

t = np.linspace(t_min, t_max, step_count)
y = odeint(model, x0, t, args=(hl,))
gammas = []

for t_i in t:
    gammas.append(gamma(t_i, x0, hl))

gammas = np.array(gammas)
y1 = eiler.diffur_steps_params(model, hl, t_min, x0, t_max, step_count)
y2 = rungekutt.diffur_steps_params(model, hl, t_min, x0, t_max, step_count)

# построение графиков
plt.plot(t, y, 'b-', linewidth=2, label='Численный метод')
plt.step(t, y1, 'g:', linewidth=2, label='Метод Эйлера')
plt.step(t, y2, 'y:', linewidth=2, label='Метод Рунге-Кутта')
plt.plot(t, gammas, 'r--', linewidth=2, label='Конечная функция')
plt.legend()
plt.xlabel('Время')
plt.ylabel('x(t)')

plt.show()

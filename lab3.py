import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#Task 1

x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1,1,10)
x3 = np.linspace(0,1,10)
f = x1**3 + 2*x1*x1 + 1
g = (x2-1)**4
u = np.sqrt(x1)
v = np.exp(-x3*x3)

##One graph

fig, ax = plt.subplots(figsize=(10,5),facecolor="darkblue")

ax.plot(x1,f,color="red",linestyle="--",marker='*')
ax.plot(x2,g,color="blue",linestyle="-",marker='o')
ax.plot(x1,u,color="green",linestyle="-.",marker='^')
ax.plot(x3,v,color="yellow",linestyle=":",marker='o')

ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax.set_xlabel('x', color="orange")
ax.set_ylabel('y', color="orange",rotation=0,)

xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.set_xticklabels(xticks, color='red',  fontsize=12)
ax.set_yticklabels(yticks, color='red',  fontsize=12)

ax.legend([r"$f(x) =x^3+2x^2+1$", r"$g(x)=(x-1)^4$",r"$u(x)=\sqrt{x}$",r"$v(x)=e^{-x^2}$"],loc=1)

plt.show()

## Separate graphs

fig, bx = plt.subplots(2,2,figsize=(10,5),facecolor="darkblue")

bx[0,0].plot(x1,f,color="red",linestyle="--",marker='*')
bx[0,1].plot(x2,g,color="blue",linestyle="-",marker='o')
bx[1,0].plot(x1,u,color="green",linestyle="-.",marker='^')
bx[1,1].plot(x3,v,color="yellow",linestyle=":",marker='o')

bx[0,0].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
bx[0,0].set_xlabel('x', color="orange")
bx[0,0].set_ylabel('y', color="orange",rotation=0,)

bx[0,1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
bx[0,1].set_xlabel('x', color="orange")
bx[0,1].set_ylabel('y', color="orange",rotation=0,)

bx[1,0].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
bx[1,0].set_xlabel('x', color="orange")
bx[1,0].set_ylabel('y', color="orange",rotation=0,)

bx[1,1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
bx[1,1].set_xlabel('x', color="orange")
bx[1,1].set_ylabel('y', color="orange",rotation=0,)

bx[0,0].legend([r"$f(x) =x^3+2x^2+1$"],loc=1)
bx[0,1].legend([ r"$g(x)=(x-1)^4$"],loc=1)
bx[1,0].legend([r"$u(x)=\sqrt{x}$"],loc=1)
bx[1,1].legend([r"$v(x)=e^{-x^2}$"],loc=1)

x00ticks = bx[0,0].get_xticks()
y00ticks = bx[0,0].get_yticks()
bx[0,0].set_xticklabels(x00ticks, color='red',  fontsize=12)
bx[0,0].set_yticklabels(y00ticks, color='red',  fontsize=12)

x01ticks = bx[0,1].get_xticks()
y01ticks = bx[0,1].get_yticks()
bx[0,1].set_xticklabels(x01ticks, color='yellow',  fontsize=12)
bx[0,1].set_yticklabels(y01ticks, color='yellow',  fontsize=12)

plt.show()

#Task 2

phi = np.linspace(0,2*np.pi,100)
ro = 1 - np.sin(phi)
plt.polar(phi,ro,color='orange',lw = 3,marker="^")
plt.show()

#Task 3

x1 = np.linspace(1, np.e, 400)
x2 = np.linspace(np.e, 9, 400)
x3 = np.linspace(9, 12, 400)

y1 = np.log(x1)
y2 = x2 / np.e
y3 = 9 * np.exp(8 - x3)

plt.plot(x1, y1, label=r'$\ln x$', color='blue', linestyle='--')
plt.plot(x2, y2, label=r'$x / e$', color='green', linestyle='-.')
plt.plot(x3, y3, label=r'$9e^{8 - x}$', color='red', linestyle=':')

plt.title('Кусочно-заданная функция')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

##Task 4

t = np.linspace(0, 2 * np.pi, 400)
x = 9 * np.sin(t / 10 - (1/2) * np.sin(9 * t / 10))
y = 9 * np.cos(t / 10 + (1/2) * np.cos(9 * t / 10))

plt.plot(x, y, label=r'Параметрическая функция')
plt.title('Параметрическая функция')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

##Task 5

x_vals = np.linspace(-1, 3, 400)
y1_vals = 6 * x_vals - 2
y2_vals = -x_vals + 12

x_intersect = (12 + 2) / (6 + 1)  # Solution of 6x - 2 = -x + 12
y_intersect = 6 * x_intersect - 2 # Put x in one of the equations

plt.plot(x_vals, y1_vals, label=r'$y_1 = 6x - 2$', color='blue')
plt.plot(x_vals, y2_vals, label=r'$y_2 = -x + 12$', color='green')

plt.scatter([x_intersect], [y_intersect], color='red', zorder=5)
plt.text(x_intersect, y_intersect, f'({x_intersect:.2f}, {y_intersect:.2f})', fontsize=12, verticalalignment='bottom')

# Оформление графика
plt.title('Пересекающиеся прямые')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

##Task 6

x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y**2)**2 - 7 * (X**2 - Y**2)

plt.contour(X, Y, Z, levels=[0], colors='blue')
plt.title('Неявная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

##Task 7

x = np.linspace(0, np.pi, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

Z = np.sin(X - 2 * Y)**2 * np.exp(-np.abs(X))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(X, Y, Z, color='green')
ax1.set_title('Wire-frame')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis)
fig2.colorbar(surf)
ax2.set_title('Surface with contour lines')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

fig3 = plt.figure(figsize=(10, 7))

ax3 = fig3.add_subplot(221, projection='3d')
ax3.plot_surface(X, Y, Z, cmap=cm.plasma)
ax3.view_init(30, 45)
ax3.set_title('View 1')

ax4 = fig3.add_subplot(222, projection='3d')
ax4.plot_surface(X, Y, Z, cmap=cm.plasma)
ax4.view_init(60, 30)
ax4.set_title('View 2')

ax5 = fig3.add_subplot(223, projection='3d')
ax5.plot_surface(X, Y, Z, cmap=cm.plasma)
ax5.view_init(20, 70)
ax5.set_title('View 3')

ax6 = fig3.add_subplot(224, projection='3d')
ax6.plot_surface(X, Y, Z, cmap=cm.plasma)
ax6.view_init(45, 60)
ax6.set_title('View 4')

plt.show()


##Task 8

fig, ax = plt.subplots()
a = 2
circle = plt.Circle((0, 0), radius=a/2, edgecolor='blue', facecolor="none", linewidth=4)
square = plt.Rectangle((-a/2, -a/2), a, a, edgecolor='red', facecolor="none", linewidth=4)

ax.add_artist(square)
ax.add_artist(circle)

ax.set_aspect('equal', 'box')
ax.set_xlim([-a, a])
ax.set_ylim([-a, a])

plt.title('Окружность, вписанная в квадрат')
plt.grid(True)

plt.savefig('circle_in_square.png', dpi=400)
plt.show()


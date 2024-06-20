import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the quadratic function and its derivative
def f(x):
    return x**2 - 4

def df(x):
    return 2 * x

# Implement Newton's method with steps storage for animation
def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    steps = [x]  # Store initial guess for visualization
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        steps.append(x_new)  # Store new guess for visualization
        if abs(f(x) / df(x)) < tol:
            break
        x = x_new
    return x, steps
# sdsd
# Initial guess
x0 = 3.0

# Apply Newton's method
root, steps = newton_method(f, df, x0)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the function
x_vals = np.linspace(-3, 3, 400)
y_vals = f(x_vals)
ax.plot(x_vals, y_vals, label='$f(x) = x^2 - 4$')

# Initial plot setup
line, = ax.plot([], [], 'r--', lw=1)
point, = ax.plot([], [], 'ro')
text = ax.text(0, 0, '', fontsize=12)

def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title("Newton's Method Visualization for $f(x) = x^2 - 4$")
    ax.legend()
    return line, point, text

def animate(i):
    if i < len(steps) - 1:
        x_k = steps[i]
        y_k = f(x_k)
        slope = df(x_k)
        tangent_line = slope * (x_vals - x_k) + y_k
        line.set_data(x_vals, tangent_line)
        point.set_data(x_k, y_k)
        text.set_position((x_k, y_k))
        text.set_text(f'$x_{i}$')
    else:
        x_final = steps[-1]
        y_final = f(x_final)
        point.set_data(x_final, y_final)
        text.set_position((x_final, y_final))
        text.set_text(f'$x_{{final}}$')
    return line, point, text

# Create animation
ani = FuncAnimation(fig, animate, frames=len(steps), init_func=init, blit=True, repeat=False)

# Show animation
plt.show()

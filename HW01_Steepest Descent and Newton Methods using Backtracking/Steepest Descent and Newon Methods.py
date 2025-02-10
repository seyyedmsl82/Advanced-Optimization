import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2


# 1-a) Compute the gradient and the Hessian of the function
def gradient(x):
    x1, x2 = x[0], x[1]
    df_dx1 = -400 * x1 * (x2 - x1**2) + 2 * (x1 - 1)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])


def hessian(x):
    x1, x2 = x[0], x[1]
    d2f_dx1dx1 = 1200 * x1**2 - 400 * x2 + 2
    d2f_dx1dx2 = -400 * x1
    d2f_dx2dx2 = 200
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx1dx2, d2f_dx2dx2]])


def backtracking_line_search(f, grad, x, direction, alpha=1, rho=0.5, c=1e-4, strategy="interpolation", min_alpha=1e-8):
    if strategy == "constant_factor":
        while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad, direction):
            alpha *= rho
            if alpha < min_alpha:
                break  # Stop if alpha becomes too small

    elif strategy == "interpolation":
        alpha_prev = alpha
        while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad, direction):
            alpha = alpha * 0.5
            if alpha < min_alpha:
                break  # Stop if alpha becomes too small

    return alpha


# Steepest descent method
def steepest_descent(x_init, num_iters=1000, tol=1e-6, alpha_init=1, strategy="interpolation"):
    x = x_init
    x_values, f_values = [x], [rosenbrock(x)] 

    for _ in range(num_iters):
        grad = gradient(x) 
        if np.linalg.norm(grad) < tol:
            break
        direction = -grad  # Steepest descent direction (negative gradient)
        alpha = backtracking_line_search(rosenbrock, grad, x, direction, alpha=alpha_init, strategy=strategy)
        x = x + alpha * direction
        x_values.append(x)
        f_values.append(rosenbrock(x))

    return np.array(x_values), np.array(f_values)


# Newton's method
def newton_method(x_init, num_iters=1000, tol=1e-6, alpha_init=1, strategy="interpolation"):
    x = x_init
    x_values, f_values = [x], [rosenbrock(x)]

    for _ in range(num_iters):
        grad = gradient(x)
        hess = hessian(x)
        if np.linalg.norm(grad) < tol:
            break
        direction = -np.linalg.solve(hess, grad)  # Compute Newton direction (Hessian = d.Gradient)
        alpha = backtracking_line_search(rosenbrock, grad, x, direction, alpha=alpha_init, strategy=strategy)
        x = x + alpha * direction
        x_values.append(x)
        f_values.append(rosenbrock(x))

    return np.array(x_values), np.array(f_values)


start_points = [np.array([1.5, 1.5]), np.array([-1.5, 2])]  # Initialize two different start points
methods = {"Steepest Descent": steepest_descent, "Newton's Method": newton_method}
strategies = ["interpolation", "constant_factor"]
results = {}

# Run both optimization methods with different strategies
for start_point in start_points:
    results[str(start_point)] = {}
    for method_name, method_func in methods.items():
        print(method_name)
        for strategy in strategies:
            x_values, f_values = method_func(start_point, strategy=strategy)
            results[str(start_point)][f"{method_name} ({strategy})"] = {
                "x_values": x_values,
                "f_values": f_values,
            }

# 1-e) Compare and explain the results of the two methods by plotting
for start_point in results:
    for method_strategy in results[start_point]:
        x_values = results[start_point][method_strategy]["x_values"]
        f_values = results[start_point][method_strategy]["f_values"]
        distances = np.linalg.norm(x_values - np.array([1, 1]), axis=1)

        # Plot function values
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(f_values)
        plt.title(f'{method_strategy} Function Value (Start: {start_point})')
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')

        # Plot distance to optimal point
        plt.subplot(1, 3, 2)
        plt.plot(distances)
        plt.title(f'{method_strategy} Distance to Optimal (Start: {start_point})')
        plt.xlabel('Iteration')
        plt.ylabel('Distance to Optimal')

        # Plot step sizes
        step_sizes = np.diff(x_values, axis=0)
        step_magnitudes = np.linalg.norm(step_sizes, axis=1)
        plt.subplot(1, 3, 3)
        plt.plot(step_magnitudes)
        plt.title(f'{method_strategy} Step Size (Start: {start_point})')
        plt.xlabel('Iteration')
        plt.ylabel('Step Size')

        plt.tight_layout()
        plt.show()

print("1-d) Comparison:")
print("1. Steepest descent is less sensitive to the starting point and has slower convergence compared to Newton's method.")
print("2. Newton's method is more sensitive to the starting point and can converge faster near the optimal point.")
print(
    "1-f) Combining these methods, such as using steepest descent initially and switching to Newton’s method "
    "when close to optimal, may achieve better results.")

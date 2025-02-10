import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    """Objective function f(x)."""
    return 0.5 * (x[0]**2 + x[1]**2 + 0.1 * x[2]**2) + 0.55 * x[2]

def gradient(x):
    """Gradient of f(x)."""
    return np.array([x[0], x[1], 0.1 * x[2] + 0.55])

def linearized_subproblem(grad, vertices):
    """Solve the linearized subproblem min grad^T x over the 
    simplex vertices."""
    values = [np.dot(grad, v) for v in vertices]
    return vertices[np.argmin(values)]

def conditional_gradient_method(x0, max_iter=200, tol=1e-10):
    """Conditional Gradient Method with line minimization 
    stepsize."""
    # Define simplex vertices
    vertices = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]

    x = x0
    f_values = [objective_function(x)]

    for k in range(max_iter):
        grad = gradient(x)
        # Solve the linearized subproblem
        x_bar = linearized_subproblem(grad, vertices)

        # Compute the stepsize
        alpha = -np.dot(grad, x_bar - x) / (np.linalg.norm(x_bar - x)**2\
             + 1e-10)
        alpha = max(0, min(1, alpha))  # Ensure alpha is in [0, 1]

        # Update x
        x_new = x + alpha * (x_bar - x)

        # Store function value for convergence plot
        f_values.append(objective_function(x_new))

        # Check for convergence
        if np.abs(objective_function(x_new) - objective_function(x)) < tol:
            break

        x = x_new
        print(f"Iteration {k+1}: x = {x_new}, f(x) = \
            {objective_function(x_new)}")

    return x, f_values

# Initial point (ensure it satisfies the constraints)
x0 = np.array([0.1, 0.4, 0.5])  # Example starting point
x_star, f_values = conditional_gradient_method(x0)

# Plot the convergence
plt.figure(figsize=(8, 6))
plt.plot(f_values, label="Objective Function Value")
plt.xlabel("Iterations")
plt.ylabel("Objective Function Value")
plt.title("Convergence of Conditional Gradient Method")
plt.legend()
plt.grid()
plt.show()

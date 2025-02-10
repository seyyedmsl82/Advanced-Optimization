import numpy as np

# Objective function and its gradient and Hessian
def objective(x):
    return x[0]**2 + 2 * x[1]**2 + 3 * x[2]**2

def grad_objective(x):
    return np.array([2 * x[0], 4 * x[1], 6 * x[2]])

def hessian_objective():
    return np.array([[2, 0, 0],
                     [0, 4, 0],
                     [0, 0, 6]])

# Constraint and its gradient
def constraint_eq(x):
    return np.sum(x) - 1

def grad_constraint_eq():
    return np.array([1, 1, 1])

def augmented_lagrangian_newton(x0, mu=1.0, lam=0.0, rho=10.0, tol=1e-6, max_iter=100):
    x = x0
    for iteration in range(max_iter):
        # Augmented Lagrangian gradient and Hessian
        lagrangian_grad = grad_objective(x) + lam * grad_constraint_eq() + mu * constraint_eq(x) * grad_constraint_eq()
        hessian_lagrangian = hessian_objective() + mu * np.outer(grad_constraint_eq(), grad_constraint_eq())

        # Newton step
        delta_x = np.linalg.solve(hessian_lagrangian, -lagrangian_grad)
        x = x + delta_x

        # Update Lagrange multiplier and penalty parameter
        lam += mu * constraint_eq(x)
        mu *= rho

        # Check convergence
        if np.linalg.norm(delta_x) < tol and np.abs(constraint_eq(x)) < tol:
            break

    return x, objective(x), iteration

def multiplier_method_newton(x0, mu=1.0, tol=1e-6, max_iter=100):
    x = x0
    for iteration in range(max_iter):
        # Penalty gradient and Hessian
        penalty_grad = grad_objective(x) + mu * constraint_eq(x) * grad_constraint_eq()
        penalty_hessian = hessian_objective() + mu * np.outer(grad_constraint_eq(), grad_constraint_eq())

        # Newton step
        delta_x = np.linalg.solve(penalty_hessian, -penalty_grad)
        x = x + delta_x

        # Check convergence
        if np.linalg.norm(delta_x) < tol and np.abs(constraint_eq(x)) < tol:
            break

        # Increase penalty (more conservatively for stability)
        mu *= 2

    return x, objective(x), iteration

# Initial point
x0 = np.array([0.1, 0.2, 0.7])

# Run both methods
augmented_solution_newton, augmented_value_newton, augmented_iteration = augmented_lagrangian_newton(x0)
multiplier_solution_newton, multiplier_value_newton, multiplier_iteration = multiplier_method_newton(x0)

print(f"Augmented Lagrange Method Solution: {augmented_solution_newton}, Function Value: {augmented_value_newton}, The requierd numbers of iteration: {augmented_iteration}")
print(f"Multipliers Method Solution{multiplier_solution_newton}, Function Value: {multiplier_value_newton}, The requierd numbers of iteration: {multiplier_iteration}")


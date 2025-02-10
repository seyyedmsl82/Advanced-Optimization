import numpy as np

def f(x):
    x1, x2 = x
    return x1**4 - 4*x1**2 + x2**4 - 4*x2**2

def grad_f(x):
    x1, x2 = x
    return np.array([4*x1**3 - 8*x1, 4*x2**3 - 8*x2])

print("1-a) Hessian is not positive definite if x1 and x2 locate between +- sqrt(2/3). (For main function)")
def hessian_f(x):
    x1, x2 = x
    return np.array([[12*x1**2 - 8, 0], [0, 12*x2**2 - 8]])


def f_new(x):
    x1, x2 = x
    return np.exp(x1) * np.sin(x2) + np.cos(x1) * x2**2

def grad_f_new(x):
    x1, x2 = x
    df_dx1 = np.exp(x1) * np.sin(x2) - np.sin(x1) * x2**2
    df_dx2 = np.exp(x1) * np.cos(x2) + 2 * x2 * np.cos(x1)
    return np.array([df_dx1, df_dx2])

def hessian_f_new(x):
    x1, x2 = x
    d2f_dx1 = np.exp(x1) * np.sin(x2) - np.cos(x1) * x2**2
    d2f_dx2 = -np.exp(x1) * np.sin(x2) + 2 * np.cos(x1)
    d2f_dx1x2 = np.exp(x1) * np.cos(x2) - 2 * x2 * np.sin(x1)
    return np.array([[d2f_dx1, d2f_dx1x2], [d2f_dx1x2, d2f_dx2]])


def backtracking_line_search(f, grad_f, x, p, alpha, rho=0.5, c=0.1):
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= rho
    return alpha


def genetic_algorithm_step_size(f, grad_f, x0, p, pop_size=20, generations=10, mutation_rate=0.1):
    population = np.random.uniform(0.1, 2.0, pop_size)  # Initial population of step sizes
    for generation in range(generations):
        costs = [f(x0 + alpha * p) for alpha in population]
        
        sorted_indices = np.argsort(costs)
        population = population[sorted_indices[:pop_size // 2]]
        
        children = []
        for i in range(pop_size // 2):
            for j in range(i + 1, pop_size // 2):
                child = (population[i] + population[j]) / 2
                if np.random.rand() < mutation_rate:
                    child += np.random.normal(0, 0.1)  # Mutation
                children.append(child)
        population = np.concatenate((population, children[:pop_size - len(population)]))
        
    best_alpha = population[np.argmin([f(x0 + alpha * p) for alpha in population])]
    return best_alpha


def modified_newton_method_with_genetic_step(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100):
    x = x0
    alphas = []
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hessian_f(x)

        # 1-b) Ensure Hessian is positive definite
        if np.linalg.eigvals(hess).min() <= 0:
            hess += np.eye(len(x)) * 1e-3  # Regularize Hessian

        p = -np.linalg.solve(hess, grad)

        # Use genetic algorithm to find initial step size
        alpha = genetic_algorithm_step_size(f, grad_f, x, p)
        alpha = backtracking_line_search(f, grad_f, x, p, alpha=alpha)

        alphas.append(alpha)
        x = x + alpha * p

        if np.linalg.norm(grad) < tol:
            break

    return {
        "Final point": x,
        "Iterations": i + 1,
        "Final cost": f(x),
        "Step sizes": alphas,
        "Distance to optimal": np.linalg.norm(x)
    }


x0 = np.array([2.0, 1.0])

result = modified_newton_method_with_genetic_step(f, grad_f, hessian_f, x0)
print("Results:", result)

print('\n')
result_new = modified_newton_method_with_genetic_step(f_new, grad_f_new, hessian_f_new, x0)
print("Results for new function:", result_new)

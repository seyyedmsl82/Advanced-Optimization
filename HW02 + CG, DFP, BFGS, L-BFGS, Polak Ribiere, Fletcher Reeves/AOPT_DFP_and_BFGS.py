import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def backtracking_line_search(func, grad_func, xk, 
                            pk, alpha=1.0, rho=0.8, c=1e-4):
    while func(xk + alpha * pk) > func(xk) + c * alpha * np.dot(
                                                grad_func(xk), pk):
        alpha *= rho
    return alpha

# BFGS implementation
def bfgs(func, grad_func, x0, max_iter=100, tol=1e-5):
    n = len(x0)
    Hk = np.eye(n)
    xk = x0
    hist_bfgs = {"x": [], "f": [], "alpha": []}
    for _ in range(max_iter):
        grad = grad_func(xk)
        if np.linalg.norm(grad) < tol:
            break
        pk = -np.dot(Hk, grad)
        alpha = backtracking_line_search(func, grad_func, xk, pk)
        sk = alpha * pk
        xk_next = xk + sk
        yk = grad_func(xk_next) - grad
        rho_k = 1.0 / np.dot(yk, sk)
        I = np.eye(n)
        Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (
            I - rho_k * np.outer(yk, sk)
            ) + rho_k * np.outer(sk, sk)

        hist_bfgs["x"].append(xk_next)
        hist_bfgs["f"].append(func(xk_next))
        hist_bfgs["alpha"].append(alpha)
        xk = xk_next
    return xk, hist_bfgs

# DFP implementation
def dfp(func, grad_func, x0, max_iter=100, tol=1e-5):
    n = len(x0)
    Hk = np.eye(n)
    xk = x0
    hist_dfp = {"x": [], "f": [], "alpha": []}
    for _ in range(max_iter):
        grad = grad_func(xk)
        if np.linalg.norm(grad) < tol:
            break
        pk = -np.dot(Hk, grad)
        alpha = backtracking_line_search(func, grad_func, xk, pk)
        sk = alpha * pk
        xk_next = xk + sk
        yk = grad_func(xk_next) - grad
        rho_k = 1.0 / np.dot(yk, sk)
        Hk = Hk + rho_k * np.outer(sk, sk) - np.outer(np.dot(Hk, yk), 
            np.dot(Hk, yk)) / np.dot(yk, np.dot(Hk, yk))
            
        hist_dfp["x"].append(xk_next)
        hist_dfp["f"].append(func(xk_next))
        hist_dfp["alpha"].append(alpha)
        xk = xk_next
    return xk, hist_dfp

# Plot results
def plot_results(hist_bfgs, iterations_bfgs, hist_dfp, 
                    iterations_dfp, optimum):     
                    
    plt.figure()
    plt.plot(range(iterations_bfgs), hist_bfgs["f"], label="BFGS")
    plt.plot(range(iterations_dfp), hist_dfp["f"], label="DFP")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title("Function Value per Iteration")
    plt.legend()
    # plt.show()

    dist_bfgs = [np.linalg.norm(np.array(x) - optimum) for x in hist_bfgs["x"]]
    dist_dfp = [np.linalg.norm(np.array(x) - optimum) for x in hist_dfp["x"]]
    
    plt.figure()
    plt.plot(range(iterations_bfgs), dist_bfgs, label="BFGS")
    plt.plot(range(iterations_dfp), dist_dfp, label="DFP")
    plt.xlabel("Iteration")
    plt.ylabel("Distance to Optimum")
    plt.title("Distance to Optimum per Iteration")
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(range(iterations_bfgs), hist_bfgs["alpha"], label="BFGS")
    plt.plot(range(iterations_dfp), hist_dfp["alpha"], label="DFP")
    plt.xlabel("Iteration")
    plt.ylabel("Step Size (alpha)")
    plt.title("Step Size per Iteration")
    plt.legend()
    plt.show()

x0 = np.array([-1.2, 1])

# Run BFGS
x_bfgs, hist_bfgs = bfgs(rosenbrock, rosenbrock_grad, x0)

# Run DFP
x_dfp, hist_dfp = dfp(rosenbrock, rosenbrock_grad, x0)

iterations_bfgs = len(hist_bfgs["f"])
iterations_dfp = len(hist_dfp["f"])
optimum = np.array([1, 1])

plot_results(hist_bfgs, iterations_bfgs, hist_dfp, iterations_dfp, optimum)

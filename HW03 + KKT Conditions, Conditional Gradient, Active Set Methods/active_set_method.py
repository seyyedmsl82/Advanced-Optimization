import numpy as np
import matplotlib.pyplot as plt

def solve_qp_active_set(G, c, A, b, x0):
    x = x0.copy()
    W = []  # Active set
    tol = 1e-10
    x_history = [x.copy()]

    # Determine initial active set and ensure feasibility
    for i, (a, b_i) in enumerate(zip(A, b)):
        if np.abs(a @ x - b_i) < tol:
            W.append(i)
        elif a @ x > b_i + tol:
            correction = (a @ x - b_i) / (a @ a)
            x = x - correction * a

    print(f"Initial point: {x0}, Active set: {W}")

    # Iterative process
    for _ in range(1000):
        try:
            # Solve for search direction considering active constraints
            W_mat = A[W] if W else np.zeros((0, len(x)))
            if W_mat.size > 0:
                KKT_mat = np.block([
                    [G, -W_mat.T],
                    [W_mat, np.zeros((len(W), len(W)))]
                ])
                KKT_rhs = np.block([-G @ x - c, np.zeros(len(W))])
                solution = np.linalg.solve(KKT_mat, KKT_rhs)
                pk = solution[:len(x)]
            else:
                pk = -np.linalg.pinv(G) @ (G @ x + c)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered.")
            break

        # Check optimality condition
        if np.linalg.norm(pk) < tol:
            lambdas = np.linalg.lstsq(-A[W].T, G @ x + c, rcond=None)[0] \
                if W else np.array([])
                
            if all(l >= 0 for l in lambdas):
                print(f"Optimal solution found: {x}, with {_} iterations.")
                return x_history
            else:
                min_lambda_index = np.argmin(lambdas)
                W.pop(min_lambda_index)
                continue

        # Compute step length ensuring feasibility
        alpha_k = 1
        for i, (a, b_i) in enumerate(zip(A, b)):
            if a @ pk > tol and i not in W:
                alpha_k = min(alpha_k, (b_i - a @ x) / (a @ pk))

        # Take the step
        x = x + alpha_k * pk
        x_history.append(x.copy())

        # Update active set
        for i, (a, b_i) in enumerate(zip(A, b)):
            if np.abs(a @ x - b_i) < tol and i not in W:
                W.append(i)

    print(f"Max iterations reached, solution: {x}")
    return x_history

G = np.array([[2, -2],
              [-2, 4]])
c = np.array([-2, -6])
A = np.array([[0.5, 0.5],
              [-1, 2],
              [-1, 0],
              [0, -1]])
b = np.array([1, 2, 0, 0])

initial_points = [
    np.array([0.5, 0.5]),
    np.array([2, 0]),
    np.array([1, 1])
]

def visualize_results(G, c, A, b, initial_points):
    fig, ax = plt.subplots(figsize=(10, 10))
    x_range = np.linspace(-1, 7, 100)
    for a, b_i in zip(A, b):
        if a[1]!=0:
            y_range = np.where(a[1] != 0, (b_i - a[0] * x_range) / a[1]+0.01, np.inf)
        elif a[1]==0:
            y_range = np.where(a[1] == 0, (b_i - a[0] * x_range) / 0.01, np.inf)
        ax.plot(x_range, y_range, label=f"${a[0]}x_1 + {a[1]}x_2 \leq {b_i}$")
        ax.fill_between(x_range, y_range, 7, alpha=0.2)

    X, Y = np.meshgrid(np.linspace(-1, 7, 100), np.linspace(-1, 7, 100))
    Z = G[0, 0]*X**2 + 2*G[0, 1]*X*Y + G[1, 1]*Y**2 + c[0]*X + c[1]*Y
    ax.contour(X, Y, Z, levels=20, cmap='coolwarm')

    for point in initial_points:
        x_history = solve_qp_active_set(G, c, A, b, point)
        x_history = np.array(x_history)
        ax.plot(x_history[:, 0], x_history[:, 1], marker='o', label=f"Path from {point}")

    ax.set_xlim([-1, 7])
    ax.set_ylim([-1, 7])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Quadratic Programming - Active Set Method")
    ax.legend()
    plt.show()

visualize_results(G, c, A, b, initial_points)

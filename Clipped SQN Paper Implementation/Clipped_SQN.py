
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

rho = []
s = []
y_prime = []

def h_beta_c(lambda_m, lambda_M, beta, L1, c):
    return lambda_m - (beta * lambda_M**2 / 4) * (2 + 3 * L1 * c)

def generate_synthetic_data(n_samples, d):
    """Generates synthetic dataset following the paper's description."""
    A = np.random.uniform(0, 1, (n_samples, d)) * (np.random.rand(n_samples, d) < 0.1)  # Sparse feature matrix
    U = np.random.uniform(-1, 1, (n_samples, d))
    b = np.sign(np.sum(U * A, axis=1))  # Labels
    return A, b

def robust_linear_regression_loss(x, A, b):
    """Non-convex robust linear regression loss function."""
    residuals = b - np.dot(A, x)
    return np.mean(np.log(0.5 * residuals**2 + 1))

def logistic_regression_loss(x, A, b):
    """Non-convex logistic regression loss function."""
    logits = np.dot(A, x)
    return -np.mean(b * np.log(sigmoid(logits)+1e-5) + (1 - b) * np.log(sigmoid(-logits)+1e-5))

def robust_linear_regression_gradient(x, A, b, batch_size):
    """Gradient of the non-convex robust linear regression loss function."""
    indices = np.random.choice(len(A), batch_size, replace=False)
    A_batch, b_batch = A[indices], b[indices]
    
    residuals = b_batch - np.dot(A_batch, x)
    grad = -np.dot(A_batch.T, residuals / (residuals**2 / 2 + 1)) / batch_size
    return grad

def logistic_regression_gradient(x, A, b, batch_size):
    """Gradient of the non-convex logistic regression loss function."""
    indices = np.random.choice(len(A), batch_size, replace=False)
    A_batch, b_batch = A[indices], b[indices]
    
    logits = np.dot(A_batch, x)
    probs = sigmoid(logits)
    grad = np.dot(A_batch.T, probs - b_batch) / batch_size
    return grad

def plot_results(loss_histories, labels):
    """Plots the training error curves for different methods."""
    plt.figure(figsize=(8, 5))
    for loss, label in zip(loss_histories, labels):
        plt.plot(loss, label=label)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Error for Different Optimization Methods")
    plt.show()

def compute_lambda_m(delta, kappa, p, q, gamma_0):
    """
    Compute the value of lambda_m as defined in the provided formula.
    """
    term_1 = (kappa**2 * p) / (q * gamma_0**4)
    term_2 = (1 / delta) + ((delta * gamma_0 + kappa**2) / gamma_0**3)
    term_3 = (p + 2 * q * gamma_0**4) / (2 * p * gamma_0)
    
    # Combine the terms inside the parentheses
    lambda_m = (delta + term_1 * (term_2 + term_3))**-1
    return lambda_m

def compute_lambda_M(delta, gamma_0, kappa, q, p):
    """
    Compute the value of lambda_M using the provided formula.
    """
    # Compute 'a'
    a = 1 + (2 / (delta * gamma_0 * kappa**2 * q)) + (1 / (delta * gamma_0 * kappa**2 * q))**2
    
    # Compute lambda_M using the formula
    lambda_M = (1 / (delta * gamma_0**2 * kappa**2 * q)) * ((a**p - 1) / (a - 1))
    
    return lambda_M

def clipped_stochastic_quasi_newton(x_0, grad_func, num_iterations, p, delta, kappa, 
                                    epsilon, beta, batch_size_S1, batch_size_S2, A, b, loss_func):
    """
    Implements the Clipped Stochastic Quasi-Newton method using the adaptive L-BFGS method.
    """
    x_k = x_0
    grad_k = grad_func(x_k, A, b, batch_size_S1)
    grad_k_prev = grad_k  # Initialize the previous gradient for first iteration
    loss_history = []

    gamma_0 = 0.9
    gamma_1 = 0.5
    L0 = np.sqrt(2*(gamma_0**2 + gamma_1**2))
    L1 = gamma_1 * np.sqrt(2)
    step_size = 1
    q = 10
    r = 5

    for k in range(num_iterations):
        if k % r == 0:
            grad_k = grad_func(x_k, A, b, batch_size_S1)
        else:
            grad_k_prev = grad_k
            grad_k = grad_k_prev + grad_func(x_k, A, b, batch_size_S2) - grad_func((x_k + step_size)/grad_k, A, b, batch_size_S2)
        
        PI = gamma_0 * (1 + (np.exp(gamma_1/L0))) + (gamma_1**2) * np.dot(grad_k_prev.T, grad_k)
        w_k = kappa**2 / PI**2

        q_k = q * PI**4
        while q_k > 1 or q_k < 0:
            q = q / 2
            q_k = q * PI**4

        lambda_m = compute_lambda_m(delta, kappa, p, q, gamma_0)
        lambda_M = compute_lambda_M(delta, gamma_0, kappa, q, p)

        h_beta = h_beta_c(lambda_m, lambda_M, beta, L1, c)
        step_size = min(
            h_beta / (2 * L0 * lambda_M**2), 
            h_beta * epsilon / (L0 * lambda_M**2 * np.linalg.norm(grad_k) + 1e-5), 
            h_beta * epsilon / (L1 * lambda_M**2 * np.linalg.norm(grad_k)**2 + 1e-5)
        )

        update_direction = stochastic_adaptive_lbfgs(
            x_k, (x_k + step_size)/grad_k, grad_k, grad_k_prev, memory_size, delta, q, kappa, k, w_k
        )
        
        x_k = x_k + step_size * update_direction
        loss_history.append(loss_func(x_k, A, b))
    
    return x_k, loss_history


def stochastic_adaptive_lbfgs(x_k, x_k_minus_1, grad_k, grad_k_minus_1, memory_size, delta, q, kappa, k, w_k_minus_1):
    """
    Implements the Stochastic Adaptive L-BFGS method to generate the Hessian inverse approximation.
    
    Parameters:
        x_k (numpy array): Current model parameter.
        x_k_minus_1 (numpy array): Previous model parameter.
        grad_k (numpy array): Gradient at x_k.
        grad_k_minus_1 (numpy array): Gradient at x_k_minus_1.
        memory_size (int): Memory size for L-BFGS updates.
        delta (float): Design parameter for stability.
        q (float): Design parameter to control curvature.
        kappa (float): Additional design parameter affecting bounds.
        w_k_minus_1 (float): Design parameter controlling scaling of the update.
        
    Returns:
        numpy array: Approximation of the Hessian inverse applied to the gradient.
    """
    global rho, s, y_prime

    # Compute s_k and y_k
    s_k = x_k - x_k_minus_1
    s.append(s_k)
    y_k = grad_k - grad_k_minus_1
    
    # Compute initial Hessian approximation H_k,0
    y_norm = np.dot(y_k.T, y_k)
    s_y_dot = np.dot(s_k.T, y_k)

    c_k = max(delta, w_k_minus_1 * y_norm / s_y_dot)
    H_k_0 = np.eye(len(x_k)) / c_k

    # Compute theta_k based on curvature conditions
    mu_k = np.dot(s_k.T, np.dot(H_k_0, s_k))
    if s_y_dot < q * mu_k:
        theta_k = ((1 - q) * mu_k) / (mu_k - s_y_dot)
    else:
        theta_k = 1
    
    # Compute y'_k
    y_prime_k = w_k_minus_1 * (theta_k * y_k + (1 - theta_k) * np.dot(H_k_0, s_k))
    y_prime.append(y_prime_k)

    rho_k = [1 / np.dot(s_k.T, y_prime_k)]
    rho.append(rho_k)

    
    # L-BFGS updates using memory
    u = grad_k.copy()
    nu = []
    
    # Minimum between memory size and the lenght of previous steps
    P = min(memory_size, k)

    # First loop for correction
    for i in range(P):
        nu_i = rho[k-i-1][0] * np.dot(u.T, s[k-i-1])
        nu.append(nu_i)
        u = u - nu_i * y_prime[k-i-1]
    
    # Initial Hessian application
    alpha = np.dot(1/c_k, u)
    
    # Second loop for applying scaling
    for i in range(P):
        beta = rho[k-P+i] * np.dot(alpha[i].T, y_prime[k-P+i])
        alpha = alpha + (nu[P-i-1] - beta) * s[k-P+i]
    
    return alpha  # -H_k * grad_k


def mini_batch_sgd(x_0, grad_func, num_iterations, batch_size, learning_rate, A, b, loss_func):
    """
    Implements the standard mini-batch SGD method.
    """
    x_k = x_0
    loss_history = []
    
    for k in range(num_iterations):
        # Get a mini-batch of samples
        grad_k = grad_func(x_k, A, b, batch_size)
        
        # Update rule: x_{k+1} = x_k - eta * grad_k
        x_k = x_k - learning_rate * grad_k
        loss_history.append(loss_func(x_k, A, b))
    
    return x_k, loss_history


# Experiment parameters
dim = 100
num_samples = 5000
num_iterations = 1500
memory_size = 5
delta = 10
kappa = 10
epsilon = 1e-6
beta = 0.01
c = 0.02
batch_size_S1 = 2000
batch_size_S2 = 100

# Generate dataset
A, b = generate_synthetic_data(num_samples, dim)

# Initialize x_0
x_0 = np.random.uniform(-1, 1, dim)

x_final_rlr, loss_rlr = clipped_stochastic_quasi_newton(x_0, robust_linear_regression_gradient, num_iterations, memory_size, delta, kappa, 
                                                         epsilon, beta, batch_size_S1, batch_size_S2, A, b, robust_linear_regression_loss)

x_final_lr, loss_lr = clipped_stochastic_quasi_newton(x_0, logistic_regression_gradient, num_iterations, memory_size, delta, kappa, 
                                                      epsilon, beta, batch_size_S1, batch_size_S2, A, b, logistic_regression_loss)

# Run Mini-batch SGD
learning_rate_sgd = 1e-3  # You can adjust this based on your experiments
x_final_sgd_rlr, loss_sgd_rlr = mini_batch_sgd(x_0, robust_linear_regression_gradient, num_iterations, 500, learning_rate_sgd, A, b, robust_linear_regression_loss)
x_final_sgd_lr, loss_sgd_lr = mini_batch_sgd(x_0, robust_linear_regression_gradient, num_iterations, 500, learning_rate_sgd, A, b, logistic_regression_loss)

# Plot results
plot_results([loss_rlr, loss_sgd_rlr], ["Robust Linear Regression: Clipped Stochastic Quasi-Newton", "Robust Linear Regression: Mini-batch SGD"])
plot_results([loss_lr, loss_sgd_lr], ["Logistic Regression: Clipped Stochastic Quasi-Newton", "Logistic Regression: Mini-batch SGD"])

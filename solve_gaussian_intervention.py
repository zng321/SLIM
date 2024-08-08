import numpy as np
import cvxpy as cp
from scipy.stats import norm
from sklearn.covariance import empirical_covariance
from scipy.linalg import sqrtm, inv

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def solve(mu_hat, sigma_hat, theta, theta0, alpha=0.1, verbose=False):
    # Init
    assert is_pos_def(sigma_hat)
    sigma_hat_sqrt = sqrtm(sigma_hat)
    d = sigma_hat.shape[0]

    # Variables
    mu = cp.Variable((d, 1))
    S = cp.Variable((d, d), PSD=True)
    t = cp.Variable((1, 1))

    # Constraints
    constraints = []

    # \theta_0 + \theta^\top \mu + \Phi^{-1}(1-\alpha) t \le 0
    constraints += [theta0 + cp.transpose(theta) @ mu + norm.ppf(1 - alpha) * t <= 0]

    # [[tI, S \theta] [\theta^\top S, t]] >> 0
    constraints += [cp.bmat([[t * np.eye(d), S @ theta], [cp.transpose(theta) @ S, t]]) >> 0]

    # \mu \in \R^d, S \in \PSD^d, t \in \R_+
    constraints += [t >= 0]

    # Objective and solve
    objective = cp.Minimize(cp.norm(mu - mu_hat[..., np.newaxis]) ** 2 + cp.norm(S - sigma_hat_sqrt, "fro") ** 2)
    p = cp.Problem(objective, constraints)

    result = p.solve(solver=cp.MOSEK, verbose=verbose)
    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return mu.value, S.value

def compute_A_opt(covsa, cov_opt):
    covsa_inv_sqrt = inv(sqrtm(covsa))
    covsa_sqrt = sqrtm(covsa)
    intermediate = covsa_sqrt @ cov_opt @ covsa_sqrt
    intermediate_sqrt = sqrtm(intermediate)
    A_opt = covsa_inv_sqrt @ intermediate_sqrt @ covsa_inv_sqrt
    return A_opt

if __name__ == "__main__":
    # Create data
    n = 10000
    D = 4096
    np.random.seed(3243)
    mu = np.random.normal(1, 2, (D, 1))
    mu_hat = np.random.normal(4, 1, (D, 1))

    A = np.random.normal(1, 2, (n, D))
    Sigma_hat = empirical_covariance(A)
    Sigma_hat_sqrt = np.linalg.cholesky(Sigma_hat)
    mu_hat = A.sum(0) / n
    assert is_pos_def(Sigma_hat)
    
    # Define the variables
    theta = np.random.normal(size=(D, 1))
    theta0 = 1.0
    alpha = 0.1
    breakpoint()
    output = solve(mu_hat, Sigma_hat, theta, theta0, alpha, verbose=True)

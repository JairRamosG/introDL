import numpy as np

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            # Predicción actual
            h = 1 / (1 + np.exp(-(x @ self.theta)))

            # Gradiente
            grad = (x.T @ (h - y)) / m

            # Hessiano
            S = np.diag(h * (1 - h))
            H = (x.T @ S @ x) / m

            # Actualización Newton-Raphson
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break

            self.theta -= delta

            # Criterio de convergencia
            if np.linalg.norm(delta, ord=1) < self.eps:
                break            
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(x @ self.theta)))
        # *** END CODE HERE ***
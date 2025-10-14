import numpy as np

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        # Calcular parámetros básicos
        phi = np.mean(y)
        mu_0 = np.mean(x[y == 0], axis=0)
        mu_1 = np.mean(x[y == 1], axis=0)

        # Matriz de covarianza compartida
        Sigma = ((x[y == 0] - mu_0).T @ (x[y == 0] - mu_0) +
                 (x[y == 1] - mu_1).T @ (x[y == 1] - mu_1)) / m

        # Inversa de Sigma
        Sigma_inv = np.linalg.inv(Sigma)

        # Parámetros lineales y sesgo (bias)
        theta = Sigma_inv @ (mu_1 - mu_0)
        theta_0 = (0.5 * (mu_0 @ Sigma_inv @ mu_0
                        - mu_1 @ Sigma_inv @ mu_1)
                + np.log(phi / (1 - phi)))

        # Concatenar intercepto + pesos
        self.theta = np.concatenate(([theta_0], theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE
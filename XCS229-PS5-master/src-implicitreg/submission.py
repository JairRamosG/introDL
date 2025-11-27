import argparse
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import util

def generate_plot(betas, X, Y, X_val, Y_val, save_path):
    """Generate a scatter plot of validation error vs. norm

    Args:
        betas: list of numpy arrays, indicating different solutions
        X, Y: training dataset 
        X_val, Y_val: validataion dataset
        save_path: path to save the plot
    """
    # check if the validationerror is zero
    for b in betas:
        assert(np.allclose(X.dot(b), Y))

    # compute the norm and the validation error of all the solutions in list beta
    val_err = []
    norms = []
    for i in range(len(betas)):
        val_err.append(np.mean((X_val.dot(betas[i]) - Y_val) ** 2))
        norms.append(np.linalg.norm(betas[i]))

    # plot the validation error against norm of the solution
    util.plot_points(norms, val_err, save_path)

def get_minimum_norm_solution(X, Y):

    """Generate the minimum norm solution

    Args:
        X, Y: features and labels of a given dataset 
        
    Returns:
        rho: the minimum norm solution
    """

    rho = None

    # *** START CODE HERE ***
    rho = (np.linalg.pinv(X) @ Y).flatten()
    # *** END CODE HERE ***

    return rho

def get_different_n_solutions(beta_0, ns, n = 3):

    """Generate different norm solutions

    Args:
        beta_0: minimum norm solution
        ns: orthonormal basis for the null space of all the inputs in the training dataset
        n: number of solutions to be generated
        
    Returns:
        betas: a list with `n` **different** solutions
    """

    solutions = []

    # *** START CODE HERE ***
    for i in range(n):
        null_vector = ns[i % ns.shape[0]] 
        solutions.append(beta_0 + (i + 1) * null_vector)
    # *** END CODE HERE ***

    return solutions

def linear_model_main():
    save_path_linear = "implicitreg_linear"
    
    train_path = 'ir1_train.csv'
    valid_path = 'ir1_valid.csv'
    X, Y = util.load_dataset(train_path)
    X_val, Y_val = util.load_dataset(valid_path)
    
    beta_0 = get_minimum_norm_solution(X, Y)
    
    # ns[i] is orthogonal to all the inputs in the training dataset
    # to help you understand the starter code, check the dimension 
    # of ns before you use it
    ns = null_space(X).T

    # get 3 different solutions and generate a scatter plot
    beta = [beta_0] + get_different_n_solutions(beta_0, ns)
    generate_plot(beta, X, Y, X_val, Y_val, save_path_linear)
    


# quadratic parameterized model
class QP:
    def __init__(self, dim, beta = None):
        self.dim = dim
        if beta is None:
            self.theta = np.ones(dim)
            self.phi = np.ones(dim)
        else:
            self.theta = beta.copy()
            self.phi = beta.copy()
            
    def train_GD(self, X, Y, eta = 8e-2, max_step = 1000, 
                 verbose = False, X_val = None, Y_val = None):
        """Train the QP model using gradient descent

        Args:
            X: shape (n, d) matrix representing the input
            Y: shape (n) vector representing the label
            eta: learning rate
            max_step: maximum training steps
            verbose: return training/validation logs
            X_val: validation/validation input
            Y_val: validation/validation output
        """
        if verbose:
            log_steps = []
            log_vals = []
            log_trains = []

        for t in range(max_step):
            # Compute the gradient of self.theta, self.phi using data X, Y
            # and update self.theta, self.phi
            # *** START CODE HERE ***
            # Predicción cuadrática
            y_pred = X @ self.theta + (X**2) @ self.phi

            # Error
            error = y_pred - Y

            # Gradientes
            grad_theta = 2 * X.T @ error / X.shape[0]
            grad_phi = 2 * (X**2).T @ error / X.shape[0]

            # Actualización
            self.theta -= eta * grad_theta
            self.phi   -= eta * grad_phi
                        
            # *** END CODE HERE ***
            if verbose:
                log_steps.append(t)
                log_vals.append(self.validation(X_val, Y_val))
                log_trains.append(self.validation(X, Y))
        if verbose:
            return (log_steps, log_trains, log_vals)
            
    def train_SGD(self, X, Y, eta = 8e-2, max_step = 1000, batch_size = 1, 
                  verbose = False, X_val = None, Y_val = None):
        """Train the QP model using stochastic gradient descent

        Args:
            X: shape (n, d) matrix representing the input
            Y: shape (n) vector representing the label
            eta: learning rate
            max_step: maximum training steps
            batch_size: batch size of the SGD algorithm
            verbose: return training/validation logs
            X_val: validation/validation input
            Y_val: validation/validation output
        """
        if verbose:
            log_steps = []
            log_vals = []
            log_trains = []

        np.random.seed(0)
        n = X.shape[0]
        idx = np.arange(n)

        for t in range(max_step):
            idx_batch = np.random.choice(idx, size = batch_size, replace = False)
            X_batch = X[idx_batch]
            Y_batch = Y[idx_batch]

            # Compute the gradient of self.theta, self.phi using data X_batch, Y_batch
            # and update self.theta, self.phi
            # *** START CODE HERE ***
            # Predicción en batch
            y_pred = X_batch @ self.theta + (X_batch**2) @ self.phi

            # Error
            error = y_pred - Y_batch

            # Gradientes
            grad_theta = 2 * X_batch.T @ error / batch_size
            grad_phi = 2 * (X_batch**2).T @ error / batch_size

            # Actualización
            self.theta -= eta * grad_theta
            self.phi   -= eta * grad_phi
            # *** END CODE HERE ***
            if verbose:
                log_steps.append(t)
                log_vals.append(self.validation(X_val, Y_val))
                log_trains.append(self.validation(X, Y))
        if verbose:
            return (log_steps, log_trains, log_vals)
    
    def predict(self, X):
        return X.dot(self.theta ** 2 - self.phi ** 2)
    
    def gradient(self, X, Y):
        """Return the gradient w.r.t. theta and phi
        """
        # *** START CODE HERE ***
        y_pred = X @ self.theta + (X**2) @ self.phi
        error = y_pred - Y
        grad_theta = 2 * X.T @ error / X.shape[0]
        grad_phi = 2 * (X**2).T @ error / X.shape[0]
        return grad_theta, grad_phi
        # *** END CODE HERE ***
    
    def validation(self, X, Y):
        return np.average(0.25 * (self.predict(X) - Y) ** 2)

def QP_model_initialization(train_path, valid_path):
    X, Y = util.load_dataset(train_path)
    X_val, Y_val = util.load_dataset(valid_path)
    d = X.shape[1]
    
    save_path = "implicitreg_quadratic_initialization"

    # Use gradient descent to train a quadratically parameterized 
    # model with different initialization. Plot the curves of validation
    # error against the number of gradient steps in a single figure.
    log = []
    labels = []
    alphas = [0.1, 0.03, 0.01]
    for i in range(len(alphas)):
        alpha = alphas[i]
        model = QP(d, np.ones(d) * alpha)
        log.append(model.train_GD(X, Y, verbose = True, X_val = X_val, Y_val = Y_val))
        labels.append("init. = {}".format(alphas[i]))
        print("initialization: {:.3f}".format(alpha), "final validation error: ", model.validation(X_val, Y_val))
    util.plot_training_and_validation_curves(log, save_path, label = labels)

def QP_model_batchsize(train_path, valid_path):
    X, Y = util.load_dataset(train_path)
    X_val, Y_val = util.load_dataset(valid_path)
    d = X.shape[1]
    
    save_path = "implicitreg_quadratic_batchsize"

    # Use SGD to train a quadratically parameterized model with
    # different batchsize. Plot the curves of validation
    # error against the number of gradient steps in a single figure.
    log = []
    labels = []
    bs = [1, 5, 40]
    for i in range(len(bs)):
        model = QP(d, np.ones(d) * 0.1)
        log.append(model.train_SGD(X, Y, eta = 0.08, batch_size = bs[i], verbose = True, 
                        X_val = X_val, Y_val = Y_val))
        labels.append("batch size = {}".format(bs[i]))
        print("batchsize: ", bs[i], "final validation error: ", model.validation(X_val, Y_val))
    util.plot_training_and_validation_curves(log, save_path, label = labels)

def implicitreg_main():
    train_path = 'ir2_train.csv'
    valid_path = 'ir2_valid.csv'
    QP_model_initialization(train_path, valid_path)
    QP_model_batchsize(train_path, valid_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['linear', 'implicitreg'], default='linear')

    args = parser.parse_args()

    if args.type == 'linear':
        linear_model_main()
    elif args.type == 'implicitreg':
        implicitreg_main()
    
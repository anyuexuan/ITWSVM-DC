import numpy as np
import cvxpy as cp
from Kernel import Kernel
import time
from collections import defaultdict


class BinaryITWSVM_DC:
    def __init__(self, kernel='sigmoid', alpha=2, C1=1.0, C2=1.0, gamma=0.1, q=0.1, degree=3):
        self.kernel = kernel.lower()
        self.alpha = alpha
        self.C1, self.C2 = C1, C2
        self.x_train = None
        self.gamma, self.q, self.degree = gamma, q, degree
        self.beta1, self.beta2 = None, None
        self.b1, self.b2 = None, None
        self.pos_d_beta_history, self.neg_d_beta_history = [], []

    def fit(self, x_train, y_train, iter_num=300, fit_on_K=False, solver=cp.MOSEK):
        self.pos_d_beta_history, self.neg_d_beta_history = [], []
        x_train, y_train = np.array(x_train) + 0., np.array(y_train) + 0.
        y_train -= np.min(y_train)
        self.x_train = x_train
        m = x_train.shape[0]
        index = np.arange(y_train.shape[0])[y_train == 0]
        index_ = list(set(range(self.x_train.shape[0])) - set(index))
        self.beta1 = np.random.uniform(-1., 1., [m, 1])
        self.beta2 = np.random.uniform(-1., 1., [m, 1])
        self.b1 = np.random.uniform(-1., 1., [1, 1])
        self.b2 = np.random.uniform(-1., 1., [1, 1])
        if fit_on_K:
            K = x_train.copy()
            A, B = x_train[index], x_train[index_]
        else:
            K = Kernel(self.x_train, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                       degree=self.degree)
            A = Kernel(self.x_train[index], self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                       degree=self.degree)
            B = Kernel(self.x_train[index_], self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                       degree=self.degree)
        K = (K + K.T) * 0.5
        E, _ = np.linalg.eig(K)
        E = np.real(E)
        rho2 = np.max(E) + 1e-4

        beta_previous, b_previous = self.beta1, self.b1

        def f(x):
            return 0.5 * (self.alpha * x.T.dot(K).dot(x)) + 0.5 * np.linalg.norm(
                A.dot(self.beta1) + self.b1) ** 2 + self.C1 * np.sum(np.maximum(0, 1 + (B.dot(self.beta1) + self.b1)))

        for i in range(iter_num):
            theta = 0.5 * self.alpha * self.beta1.T.dot(rho2 * np.eye(m) - K)
            beta, b = cp.Variable((m, 1)), cp.Variable((1, 1))
            loss = cp.quad_form(beta, 0.5 * self.alpha * rho2 * np.eye(m)) + 0.5 * cp.norm(
                A @ beta + b) ** 2 + 0.5 * self.C1 * cp.sum((cp.maximum(0, 1 + (B @ beta + b))) ** 2) - theta @ beta
            cp.Problem(cp.Minimize(loss)).solve(solver=solver)
            self.beta1, self.b1 = beta.value, b.value

            if self.beta1 is None:
                self.beta1 = beta_previous
                self.b1 = b_previous
                break
            d_beta = self.beta1 - beta_previous
            self.pos_d_beta_history.append(np.sum(d_beta ** 2))

            if np.sum(d_beta ** 2) < 1e-3:
                break
            vt = 3
            while f(self.beta1 + vt * d_beta) > f(self.beta1) - 0.4 * vt * np.sum(d_beta ** 2):
                vt = 0.5 * vt
            self.beta1 = self.beta1 + vt * d_beta
            beta_previous, b_previous = self.beta1, self.b1

        beta_previous, b_previous = self.beta1, self.b1

        def f(x):
            return 0.5 * (self.alpha * x.T.dot(K).dot(x)) + 0.5 * np.linalg.norm(
                B.dot(self.beta2) + self.b2) ** 2 + self.C2 * np.sum(np.maximum(0, 1 - (A.dot(self.beta2) + self.b2)))

        for i in range(iter_num):
            theta = 0.5 * self.alpha * self.beta2.T.dot(rho2 * np.eye(m) - K)
            beta, b = cp.Variable((m, 1)), cp.Variable((1, 1))
            loss = cp.quad_form(beta, 0.5 * self.alpha * rho2 * np.eye(m)) + 0.5 * cp.norm(
                B @ beta + b) ** 2 + 0.5 * self.C2 * cp.sum((cp.maximum(0, 1 - (A @ beta + b))) ** 2) - theta @ beta
            cp.Problem(cp.Minimize(loss)).solve(solver=solver)
            self.beta2, self.b2 = beta.value, b.value

            if self.beta2 is None:
                self.beta2 = beta_previous
                self.b2 = b_previous
                break
            d_beta = self.beta2 - beta_previous
            self.neg_d_beta_history.append(np.sum(d_beta ** 2))

            if np.sum(d_beta ** 2) < 1e-3:
                break
            vt = 3
            while f(self.beta2 + vt * d_beta) > f(self.beta2) - 0.4 * vt * np.sum(d_beta ** 2):
                vt = 0.5 * vt
            self.beta2 = self.beta2 + vt * d_beta
            beta_previous, b_previous = self.beta2, self.b2

    def predict(self, x, test_on_K=False):
        x = np.array(x)
        if test_on_K:
            K = x
        else:
            K = Kernel(x, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                       degree=self.degree)
        pos_out = np.abs(K.dot(self.beta1) + self.b1)
        neg_out = np.abs(K.dot(self.beta2) + self.b2)
        return (np.sign(pos_out - neg_out).squeeze() + 1) / 2

    def score(self, x_test, y_test, test_on_K=False):
        x_test, y_test = np.array(x_test) + 0., np.array(y_test) + 0.
        y_test -= np.min(y_test)
        return np.mean(np.equal(self.predict(x_test, test_on_K), y_test))

    def save(self, filepath):
        np.savez(filepath, kernel=self.kernel, alpha=self.alpha, C1=self.C1, C2=self.C2, x_train=self.x_train,
                 gamma=self.gamma, q=self.q, degree=self.degree, beta1=self.beta1, beta2=self.beta2, b1=self.b1,
                 b2=self.b2)

    def load(self, filepath):
        if '.npz' not in filepath:
            filepath = filepath + '.npz'
        params = np.load(filepath)
        self.kernel = params['kernel']
        self.alpha = params['alpha']
        self.C1 = params['C1']
        self.C2 = params['C2']
        self.x_train = params['x_train']
        self.gamma, self.q, self.degree = params['gamma'], params['q'], params['degree']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.b1 = params['b1']
        self.b2 = params['b2']


class ITWSVM_DC():
    def __init__(self, kernel='rbf', multi_method='ovr', alpha=2, C=1.0, gamma=0.1, q=0., degree=3):
        self.kernel = kernel.lower()
        self.multi_method = multi_method.lower()
        self.alpha = alpha
        self.C = C
        self.x_train = None
        self.N_CLASSES = None
        self.gamma, self.q, self.degree = gamma, q, degree
        self.model = defaultdict(list)

    def fit(self, x_train, y_train, iter_num=300, fit_on_K=False, solver=cp.MOSEK):
        x_train, y_train = np.array(x_train) + 0., np.array(y_train) + 0.
        y_train -= np.min(y_train)
        m = x_train.shape[0]
        self.N_CLASSES = int(np.max(y_train) + 1)
        self.x_train = x_train
        self.model = defaultdict(list)
        t = time.time()
        if self.multi_method == 'ovr':
            if fit_on_K:
                K = x_train
            else:
                K = Kernel(self.x_train, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                           degree=self.degree)
            K = (K + K.T) * 0.5
            E, _ = np.linalg.eig(K)
            E = np.real(E)
            rho2 = np.max(E) + 1e-4
            for i in range(self.N_CLASSES):
                index = np.arange(y_train.shape[0])[y_train == i]
                index_ = list(set(range(self.x_train.shape[0])) - set(index))
                if fit_on_K:
                    A, B = x_train[index], x_train[index_]
                else:
                    A = Kernel(self.x_train[index], self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                               degree=self.degree)
                    B = Kernel(self.x_train[index_], self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                               degree=self.degree)
                beta_new, b_new = np.random.uniform(-1., 1., [m, 1]), np.random.uniform(-1., 1., [1, 1])
                beta_previous, b_previous = beta_new.copy(), b_new.copy()
                for _ in range(iter_num):
                    theta = 0.5 * self.alpha * beta_new.T.dot(rho2 * np.eye(m) - K)

                    def f(x):
                        return 0.5 * (self.alpha * x.T.dot(K).dot(x)) + 0.5 * np.linalg.norm(
                            A.dot(beta_new) + b_new) ** 2 + self.C * np.sum(
                            np.maximum(0, 1 - (B.dot(beta_new) + b_new)))

                    beta, b = cp.Variable((m, 1)), cp.Variable((1, 1))
                    loss = cp.quad_form(beta, 0.5 * self.alpha * rho2 * np.eye(m)) + 0.5 * cp.norm(
                        A @ beta + b) ** 2 + 0.5 * self.C * cp.sum(
                        (cp.maximum(0, 1 - (B @ beta + b))) ** 2) - theta @ beta
                    cp.Problem(cp.Minimize(loss)).solve(solver=solver)
                    beta_new, b_new = beta.value, b.value

                    if beta_new is None:
                        beta_new = beta_previous
                        b_new = b_previous
                        break
                    d_beta = beta_new - beta_previous

                    if np.sum(d_beta ** 2) < 1e-3:
                        break
                    vt = 3
                    while f(beta_new + vt * d_beta) > f(beta_new) - 0.4 * vt * np.sum(d_beta ** 2):
                        vt = 0.5 * vt
                    beta_new = beta_new + vt * d_beta
                    beta_previous, b_previous = beta_new, b_new
                self.model[i] = [beta_new, b_new]
        elif self.multi_method == 'ovo':
            for i in range(self.N_CLASSES):
                index = np.arange(y_train.shape[0])[y_train == i]
                if len(index) == 0:
                    continue
                for j in range(self.N_CLASSES):
                    if j >= i:
                        continue
                    index_ = np.arange(y_train.shape[0])[y_train == j]
                    if len(index_) == 0:
                        continue
                    X = x_train[np.concatenate([index, index_])]
                    Y = np.concatenate([np.zeros(len(index)), np.ones(len(index_))])

                    model = BinaryITWSVM_DC(kernel=self.kernel, alpha=self.alpha, C1=self.C, C2=self.C, gamma=self.gamma,
                                           q=self.q, degree=self.degree)
                    model.fit(X, Y, fit_on_K=False, solver=solver)
                    self.model[(i, j)] = model
        else:
            print('Multiple Classification Method Error！')
            return None

    def predict(self, x_test, test_on_K=False):
        x_test = np.array(x_test)
        if self.multi_method == 'ovr':
            if test_on_K:
                K = x_test
            else:
                K = Kernel(x_test, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q, degree=self.degree)
            model_out = np.full([x_test.shape[0], self.N_CLASSES], np.inf)
            for i in self.model.keys():
                beta, b = self.model[i]
                model_out[:, i] = np.abs(K.dot(beta) + b).squeeze()
            prediction = np.squeeze(np.argmin(model_out, 1))
            return prediction
        elif self.multi_method == 'ovo':
            model_out = np.zeros([x_test.shape[0], self.N_CLASSES])
            for (i, j) in self.model.keys():
                model = self.model[(i, j)]
                out = model.predict(x_test)
                model_out[:, i] += (out == 0)
                model_out[:, j] += (out == 1)
            prediction = np.argmax(model_out, axis=1)
            return prediction
        else:
            print('Multiple Classification Method Error！')
            return None

    def score(self, x_test, y_test, test_on_K=False):
        x_test, y_test = np.array(x_test) + 0., np.array(y_test) + 0.
        y_test -= np.min(y_test)
        return np.mean(self.predict(x_test, test_on_K=test_on_K) == y_test)

    def save(self, filepath):
        np.savez(filepath, kernel=self.kernel, multi_method=self.multi_method, alpha=self.alpha, C=self.C,
                 x_train=self.x_train, N_CLASSES=self.N_CLASSES, gamma=self.gamma, q=self.q, degree=self.degree,
                 model=[self.model])

    def load(self, filepath):
        if '.npz' not in filepath:
            filepath = filepath + '.npz'
        params = np.load(filepath, allow_pickle=True)
        self.kernel = params['kernel']
        self.multi_method = str(params['multi_method'])
        self.alpha = params['alpha']
        self.C = params['C']
        self.x_train = params['x_train']
        self.N_CLASSES = params['N_CLASSES']
        self.gamma, self.q, self.degree = params['gamma'], params['q'], params['degree']
        self.model = params['model'][0]

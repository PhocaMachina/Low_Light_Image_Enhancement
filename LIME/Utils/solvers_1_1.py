import numpy as np
import cv2
from LIME.Utils.gradient import gen_D, gen_D_truncated, get_grad_toeplitz, get_grad_toeplitz_transpose, get_magnitude, \
    gen_D_conv, get_grad_conv, gen_grad_kernel
from LIME.Utils.weight_gen import gen_spacial_affinity_kernel, gen_weight
from LIME.Utils.np_add_broadcast import padded_add
from numpy.linalg import norm
# linear algebra for sped_up solver
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


class exact_solver():

    def __init__(self, map: np.ndarray, iterations: int, mu0: float, rho: float, alpha: float = 0.15, *args, **kwargs):
        self.iter = iterations  # max iter
        self.map = map  # init map
        self.rho = rho
        self.alpha = alpha
        self.m, self.n = self.map.shape
        self.converge = 0
        self.curr_iter = 0

        self.I = np.ones((self.m, self.n))
        self.G = np.zeros((2 * self.m, self.n))
        self.Z = np.zeros((2 * self.m, self.n))
        self.kernel = gen_spacial_affinity_kernel(spatial_sigma=2, size=5)
        self.W_v = gen_weight(map, 0, self.kernel)
        self.W_h = gen_weight(map, 1, self.kernel)

        self.mu = mu0

    def t_sub(self):
        print('running T sub programme')
        V = self.G - (self.Z / self.mu)

        # since in the paper the D_T*V operator is actually reshape(D_T*vec(V)),
        # and D_T can be written as [D_v_T, D_h_T], bear in mind that V contains both horizontal and vertical entries
        # thus the D_T*V can be written as [D_v_T, D_h_T] [v_v, v_h]_T = D_v_T*v_v + D_h_T*v_h. the following  code
        # is an implementation of this formula
        D_v_T = gen_D_conv(self.m, 0, trans=1)
        D_h_T = gen_D_conv(self.n, 1, trans=1)
        V_h = V[0:self.m, 0:self.n]
        V_v = V[self.m:2 * self.m, 0:self.n]
        mu_D_T_V = self.mu * (np.dot(V_h, D_h_T) + np.dot(D_v_T, V_v))
        # the FFT(D_d) is simply application of the rule: circular convolution in 2_D space is equivalent to multiplication in the frequency domain (2_D_DFT)
        # and since my multiplication with D is equivalent to using the convolve function with grad_kernels, the FFT part is simply translating the kernels
        # into the frequency domain with correct padding (correct dimensions)
        d_h = np.zeros((self.m, self.n))
        d_v = np.zeros((self.m, self.n))

        d_h[1, 1] = -1
        d_h[1, 0] = 1
        d_v[1, 1] = -1
        d_v[0, 1] = 1
        F_Dh = np.multiply(np.fft.fftshift(np.fft.fft2(d_h)).conj(), np.fft.fftshift(np.fft.fft2(d_h)))
        F_Dv = np.multiply(np.fft.fftshift(np.fft.fft2(d_v)).conj(), np.fft.fftshift(np.fft.fft2(d_v)))
        # calculate the numerator
        top = 2 * self.map + mu_D_T_V
        numerator = np.fft.fftshift(np.fft.fft2(top))
        # calculate the denominator

        denominator = 2 * self.I + self.mu * (F_Dh + F_Dv)
        # update T and calculate T_grad
        inside = np.divide(numerator, denominator)
        self.T = np.fft.ifft2(np.fft.ifftshift(inside))
        D_h = gen_D_conv(self.n, x=1)
        D_v = gen_D_conv(self.m, x=0)
        self.T_grad = np.vstack((np.dot(self.T, D_h), np.dot(D_v, self.T)))

    def shrinkage(self, threshold: np.ndarray, input: np.ndarray):
        return np.multiply(np.sign(input), np.maximum(np.absolute(input) - threshold, np.zeros_like(input)))

    def g_sub(self):
        print('G')
        threshold = (self.alpha * np.vstack((self.W_h, self.W_v))) / self.mu
        input = self.T_grad + (self.Z / self.mu)
        self.G = self.shrinkage(threshold=threshold, input=input)

    def z_sub(self):
        print('Z')
        self.Z = self.Z + self.mu * (self.T_grad - self.G)
        self.mu = self.mu * self.rho

    def step(self):
        print('Iterating: ' + str(self.curr_iter))
        self.curr_iter += 1
        self.t_sub()
        self.g_sub()
        self.z_sub()

        self.delta = norm(self.T_grad - self.G) / norm(self.map)
        print(self.delta)

    def iterate(self, k: int):
        for i in range(k):
            if self.curr_iter < self.iter and self.converge == 0:
                self.step()


class sped_up_solver():
    def __init__(self, map: np.ndarray, gamma: float = 0.8, alpha: float = 0.15, epsilon: float = 1e-3, *args,
                 **kwargs):
        self.map = map
        self.lamb = alpha
        self.eps = epsilon
        self.gamma = gamma
        self.map_refined = np.zeros_like(map)
        self.map_refined_gamma = np.zeros_like(map)

        self.kernel = gen_spacial_affinity_kernel()
        self.W_v = gen_weight(map, 0, self.kernel)
        self.W_h = gen_weight(map, 1, self.kernel)

    def get_five_points(self, p: int, m: int, n: int):

        i, j = p // n, p % n
        d = {}
        if i - 1 >= 0:
            d[(i - 1) * n + j] = (i - 1, j, 0)
        if i + 1 < m:
            d[(i + 1) * n + j] = (i + 1, j, 0)
        if j - 1 >= 0:
            d[i * n + j - 1] = (i, j - 1, 1)
        if j + 1 < n:
            d[i * n + j + 1] = (i, j + 1, 1)
        return d

    def get_refined_map(self):

        m, n = self.map.shape
        map_vec = self.map.copy().flatten()
        # generate the five-point spatially inhomogeneous Laplacian matrix with a dictionary and then use csr_matrix to form L_g
        row, column, data = [], [], []
        for p in range(n * m):
            diag = 0
            for q, (k, l, x) in self.get_five_points(p, m, n).items():
                weight = self.W_h[k, l] if x else self.W_v[k, l]
                row.append(p)
                column.append(q)
                data.append(-weight)
                diag += weight
            row.append(p)
            column.append(p)
            data.append(diag)
        F = csr_matrix((data, (row, column)), shape=(m * n, m * n))

        # solve the linear system using spsolve
        Id = diags([np.ones(m * n)], [0])
        A = Id + self.lamb * F
        self.map_refined = spsolve(csr_matrix(A), map_vec, permc_spec=None, use_umfpack=True).reshape((m, n))

        # gamma correction
        self.map_refined_gamma = np.clip(self.map_refined, self.eps, 1) ** self.gamma


if __name__ == '__main__':
    solver = exact_solver(np.ones((10, 12)), 60, 0.05, 1.1)
    solver.iterate(12)

import numpy as np
import cv2
from LIME.Utils.gradient import gen_D, gen_D_truncated, get_grad_toeplitz, get_grad_toeplitz_transpose, get_magnitude, \
    gen_D_conv, get_grad_conv, gen_grad_kernel
from LIME.Utils.weight_gen import gen_spacial_affinity_kernel, gen_weight
from LIME.Utils.np_add_broadcast import padded_add
# linear algebra for sped_up solver
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


class exact_solver():

    def __init__(self, map: np.ndarray, iterations: int, mu0: float, rho: float, alpha: float = 0.15, mode: str = 'cat',
                 *args, **kwargs):
        self.iter = iterations  # max iter
        self.map = map  # init map
        self.mu0 = mu0
        self.rho = rho
        self.alpha = alpha
        self.mode = mode
        self.m, self.n = self.map.shape

        # opt init
        if self.mode == 'cat':
            self.Z = np.zeros((2 * self.m, self.n))
            self.Zero = np.zeros((2 * self.m, self.n))
        elif self.mode == 'mag':
            self.Z = np.zeros((self.m, self.n))
            self.Zero = np.zeros((self.m, self.n))
        else:
            print('crap!')

        self.I = np.ones((self.m, self.n))
        self.G = self.Z
        self.T = map
        self.mu = self.mu0
        self.kernel = gen_spacial_affinity_kernel()
        self.W_v = gen_weight(map, 0, self.kernel)
        self.W_h = gen_weight(map, 1, self.kernel)
        self.W = np.vstack((self.W_v, self.W_h))

        self.iteration_num = 0

    def shrink(self, threshold: np.ndarray, input:np.ndarray):
        # print(threshold.shape)
        # print(threshold)
        # print(np.abs(input) - threshold)
        return np.multiply(np.sign(input), np.maximum(np.abs(input) - threshold, self.Zero)) # sgn(x).*max{abs(x)-epsilon,0}

    def t_sub(self):
        print('running T sub programme')
        U = self.G - self.Z / self.mu
        U_v = U[0:self.m, 0:self.n]
        U_h = U[self.m:2 * self.m, 0:self.n]
        D_trans = gen_D_conv(self.m, 2, trans=1)
        D_trans_v = D_trans[0:self.m, 0:self.m]
        D_trans_h = D_trans[self.m:2 * self.m, 0:self.m]

        # since in the paper the D_T*V operator is actually reshape(D_T*vec(V)),
        # and D_T can be written as [D_v_T, D_h_T], bear in mind that V contains both horizontal and vertical entries
        # thus the D_T*V can be written as [D_v_T, D_h_T] [v_v, v_h]_T = D_v_T*v_v + D_h_T*v_h. the following  code
        # is an implementation of this formula

        D_trans_U = np.dot(D_trans_v, U_v) + np.dot(D_trans_h, U_h)
        # the FFT(D_d) is simply application of the rule: circular convolution in 2_D space is equivalent to multiplication in the frequency domain (2_D_DFT)
        # and since my multiplication with D is equivalent to using the convolve function with grad_kernels, the FFT part is simply translating the kernels
        # into the frequency domain with correct padding (correct dimensions)
        fft_v = np.fft.fftshift(np.fft.fft2(gen_grad_kernel(0), (self.m, self.n)))
        fft_h = np.fft.fftshift(np.fft.fft2(gen_grad_kernel(1), (self.m, self.n)))
        fft_conjugate_sum = np.multiply(np.conj(fft_v), fft_v) + np.multiply(np.conj(fft_h), fft_h)

        # calculate the numerator
        numerator = np.fft.fftshift(np.fft.fft2(2 * self.map + self.mu*D_trans_U))

        # calculate the denominator
        denominator = 2*self.I + self.mu*fft_conjugate_sum

        #update T and calculate T_grad
        self.T = np.fft.ifft2(np.fft.ifftshift(numerator / denominator))
        D_v = gen_D_conv(self.m, x=0)
        D_h = gen_D_conv(self.m, x=1)
        self.grad_T = np.vstack((np.dot(D_v, self.T), np.dot(D_h, self.T)))

    def g_sub(self):
        print('running G sub programme')
        if self.mode == 'mag':
            print('on magnitude_mode')

        if self.mode == 'cat':
            print('on concatenate_mode')
            X = self.alpha * self.W / self.mu
            Y = self.Z / self.mu

            # update G
            self.G = self.shrink(threshold=X, input=self.grad_T + Y)

    def z_sub(self):
        print('running Z sub programme')
        if self.mode == 'mag':
            print('on magnitude_mode')

        if self.mode == 'cat':
            print('on concatenate_mode')
            self.Z = self.Z + self.mu * (self.grad_T - self.G)
            self.mu = self.mu * self.rho

    def step(self):
        print('Iteration ' + str(self.iteration_num + 1))
        if self.iteration_num <= self.iter:
            self.t_sub()
            self.g_sub()
            self.z_sub()
            self.iteration_num += 1
            # print(self.T)
            print(self.G)
            # print(self.Z)
        else:
            print('reached max number of iterations')

    def iterate(self, num_iterations: int):
        if self.iteration_num + num_iterations > self.iter:
            print('exceed iteration limit')
        else:
            for i in range(num_iterations):
                self.step()


class sped_up_solver():
    def __init__(self, map: np.ndarray, gamma: float = 0.8, alpha: float = 0.15, epsilon:float = 1e-3, *args, **kwargs):
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
    solver = sped_up_solver(np.ones((12, 8)))
    solver.get_refined_map()
    print(solver.map_refined_gamma)

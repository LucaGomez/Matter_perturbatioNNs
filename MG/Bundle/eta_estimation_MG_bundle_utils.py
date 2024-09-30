import torch
import numpy as np
from copy import deepcopy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def _chebyshev_second(a, b, n):
    """
    Compute the Chebyshev nodes on the interval [a, b].
    """
    nodes = np.cos(np.arange(n) / float(n - 1) * np.pi)
    nodes = ((a + b) + (-b + a) * nodes) / 2
    return nodes


class EtaEstimation:
    def __init__(self, v_sol, v_index, Om_m_0, g_a, res_fun, f_fun, t_min, t_max, N_for_int=int(1e3), j_max=6):
        """
        Initialize the eta estimation class.
        """
        self.v_sol = v_sol
        self.v_index = v_index
        self.Om_m_0 = Om_m_0
        self.g_a = g_a
        self.res = res_fun
        self.f = f_fun
        self.j_max = j_max
        self.t_min, self.t_max = t_min, t_max
        self.ts_for_int = _chebyshev_second(t_min, t_max, N_for_int)
        self.ts_for_int_diff = torch.tensor(deepcopy(self.ts_for_int)).requires_grad_().reshape(-1, 1)
        self.eta_list = [0] * (j_max + 1)
        #self.multiple_dependent_variables = len(np.array(self.v_sol(self.ts_for_int[0:1], to_numpy=True)).shape) > 1
        self.multiple_dependent_variables = True
        
    def C(self, t):
        """
        Compute the C function.
        """
        
        Om_t = self.Om_m_0*np.ones_like(t)
        ga_t = self.g_a*np.ones_like(t)
        v_sol_val = self.v_sol(t, Om_t, ga_t, to_numpy=True)
        if self.multiple_dependent_variables:
            return 2 * v_sol_val[self.v_index] + self.f(t, self.Om_m_0)
        else:
            return 2 * v_sol_val + self.f(t)

    def integrator(self, integrand, ts):
        """
        Perform integration using the cumulative trapezoidal rule.
        """
        return cumulative_trapezoid(integrand, x=ts, initial=0)

    def q_fun(self, ts):
        """
        Compute the q function by integrating C over time.
        """
        return self.integrator(self.C(ts), ts)

    def compute_integrand(self, factor_in, J, eta_list):
        out = 0
        for j1 in range(J):
            j2 = (J - 1) - j1
            out += eta_list[j1] * eta_list[j2]
        return out*factor_in

    def compute_etas(self):
        """
        Compute the eta values.
        """
        ts = self.ts_for_int
        ts_diff = self.ts_for_int_diff

        self.qs = self.q_fun(ts)
        
        Om_t = self.Om_m_0*torch.ones_like(ts_diff)
        ga_t = self.g_a*torch.ones_like(ts_diff)

        if self.multiple_dependent_variables:
            ress = self.res(*self.v_sol(ts_diff, Om_t, ga_t), ts_diff, self.Om_m_0, self.g_a)[self.v_index].reshape(1, -1).detach().numpy()[0]
        else:
            ress = self.res(self.v_sol(ts_diff), ts_diff)[self.v_index].reshape(1, -1).detach().numpy()[0]

        factor_in = np.exp(self.qs)
        factor_out = np.exp(-self.qs)
        self.eta_list[0] = -factor_out * self.integrator(ress * factor_in, ts)

        for J in range(1, self.j_max + 1):
            aux = self.compute_integrand(factor_in, J, self.eta_list)
            self.eta_list[J] = -self.integrator(aux, ts) * factor_out

    def make_eta_hat(self, order=None, loose_bound=False, tight_bound=False):
        """
        Construct eta_hat by interpolating the eta values.
        """
        order = self.j_max if order is None else order
        eta = 0
        for j in range(order + 1):
            eta_j = self.eta_list[j]
            if loose_bound:
                eta_j = np.abs(eta_j)
            elif tight_bound and j > tight_bound[1]:
                eta_j = np.abs(eta_j)

            eta += eta_j

            if tight_bound and j == tight_bound[0]:
                eta = np.abs(eta)

        return interp1d(self.ts_for_int, eta)

import torch
import numpy as np
from typing import Callable
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def chebyshev_second(a: float, b: float, n: int) -> np.ndarray:
    """
    Compute the Chebyshev nodes on the interval [a, b].

    Args:
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        n (int): Number of nodes.

    Returns:
        np.ndarray: Chebyshev nodes.
    """
    nodes = np.cos(np.arange(n) / (n - 1) * np.pi)
    return ((a + b) + (-b + a) * nodes) / 2


class EtaEstimation_MG:
    def __init__(
            self,
            v_sol: Callable,
            Om_m_0: float,
            g_a: float,
            v_index: int,
            res_fun: Callable,
            B_fun: Callable,
            C_fun: Callable,
            t_min: float,
            t_max: float,
            N_for_int: int,
            j_max: int
    ):
        """
        Initialize the eta estimation class.

        Args:
            v_sol (Callable): Solution function.
            v_index (int): Index of the variable of interest.
            res_fun (Callable): Residual function.
            B_fun (Callable): B function.
            C_fun (Callable): C function.
            t_min (float): Minimum time.
            t_max (float): Maximum time.
            N_for_int (int, optional): Number of points for integration. Defaults to 1e5.
            j_max (int, optional): Maximum order for eta estimation. Defaults to 6.
        """
        self.v_sol = v_sol
        self.Om_m_0 = Om_m_0
        self.g_a = g_a
        self.v_index = v_index
        self.r = res_fun
        self.B = B_fun
        self.C = C_fun
        self.j_max = j_max
        self.t_min, self.t_max = t_min, t_max
        self.ts_for_int = chebyshev_second(t_min, t_max, N_for_int)
        self.ts = np.linspace(t_min, t_max, N_for_int)
        self.ts_diff = torch.tensor(self.ts, requires_grad=True).reshape(-1, 1)
        self.ts_for_int_diff = torch.tensor(self.ts_for_int, requires_grad=True).reshape(-1, 1)
        self.eta_list = [0] * (j_max + 1)
        #self.multiple_dependent_variables = len(np.atleast_1d(self.v_sol(self.ts_for_int[0:1], to_numpy=True))) > 1
        self.multiple_dependent_variables = True
    def F_1(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the C function.

        Args:
            t (np.ndarray): Time points.

        Returns:
            np.ndarray: Computed C function values.
        """
        Om = self.Om_m_0*np.ones_like(t)
        ga = self.g_a*np.ones_like(t)
        v_sol_val = self.v_sol(t, Om, ga, to_numpy=True)
        if self.multiple_dependent_variables:
            return 2 * v_sol_val[self.v_index] + self.B(t,self.Om_m_0)
        else:
            return 2 * v_sol_val + self.B(t,self.Om_m_0)

    @staticmethod
    def integrator(integrand: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """
        Perform integration using the cumulative trapezoidal rule.

        Args:
            integrand (np.ndarray): Function to integrate.
            ts (np.ndarray): Time points.

        Returns:
            np.ndarray: Integrated values.
        """
        return cumulative_trapezoid(integrand, x=ts, initial=0)

    def compute_etas(self):
        """
        Compute the eta values.
        """
        ts = self.ts_for_int
        ts_diff = self.ts_for_int_diff
        Om_diff = self.Om_m_0*torch.ones_like(ts_diff)
        ga_diff = self.g_a*torch.ones_like(ts_diff)

        self.qs = self.integrator(self.F_1(ts), ts)
        self.qs_down = self.integrator(np.minimum(self.F_1(ts), 0), ts)

        if self.multiple_dependent_variables:
            rs = self.r(*self.v_sol(ts_diff,Om_diff,ga_diff), ts_diff, self.Om_m_0, self.g_a)[self.v_index].reshape(1, -1).detach().numpy()[0]
        else:
            rs = self.r(self.v_sol(ts_diff,Om_diff,ga_diff), ts_diff, self.Om_m_0, self.g_a)[self.v_index].reshape(1, -1).detach().numpy()[0]

        factor_in = np.exp(self.qs)
        factor_out = np.exp(-self.qs)
        factor_down_in = np.exp(self.qs_down)
        factor_down_out = np.exp(-self.qs_down)

        self.R = np.max(self.integrator(np.abs(rs) * factor_down_in, ts))
        self.K = np.max(np.abs(self.C(ts)) * factor_down_out)
        self.eta_list[0] = -factor_out * self.integrator(rs * factor_in, ts)

        for J in range(1, self.j_max + 1):
            aux = factor_in * sum(self.eta_list[j1] * self.eta_list[J - 1 - j1] for j1 in range(J))
            self.eta_list[J] = -self.integrator(aux, ts) * factor_out

    def make_eta_hat(self, order: int) -> Callable:
        """
        Construct eta_hat by interpolating the eta values.

        Args:
            order (int, optional): Order of eta estimation. Defaults to None (uses j_max).
        Returns:
            Callable: Interpolated eta_hat function.
        """
        order = self.j_max if order is None else order
        eta = sum(self.eta_list[j] for j in range(order + 1))

        return interp1d(self.ts_for_int, eta)

    def make_eta_bound(self, J: int) -> Callable:
        """
        Compute the eta bound.

        Args:
            J (int, optional): Order for eta bound computation. Defaults to None (uses j_max).

        Returns:
            Callable: Interpolated eta bound function.
        """
        R, K = self.R, self.K
        ts, t_0 = self.ts_for_int, self.t_min

        condition = R * K * (ts - t_0)
        if np.max(condition) >= 1:
            raise ValueError('The condition R*K*(t-t_0) < 1 is not satisfied.')

        J = self.j_max if J is None else J
        eta = sum(self.eta_list[:J + 1])

        limit = (R * (condition ** (J + 1)) * np.exp(-self.qs_down)) / (1 - condition)
        bound = np.abs(eta) + limit
        #print('term for Y = '+str(np.max(limit)))
        return interp1d(self.ts_for_int, bound)
    
    def make_eta_bound_X(self, j_max: int):
        """
        Compute the X bound.

        Args:
            J (int, optional): Order for bound computation.

        Returns:
            Numpy array: X bound computed in the values ts.
        """
        Om_m_0 = self.Om_m_0
        g_a = self.g_a
        R = self.R
        ts = self.ts_for_int
        ts_diff = self.ts_for_int_diff
        eta = sum(self.eta_list[:j_max + 1])
        Om = Om_m_0*torch.ones_like(ts_diff)
        ga = g_a*torch.ones_like(ts_diff)
        rs = self.r(*self.v_sol(ts_diff,Om,ga), ts_diff, Om_m_0, g_a)[0].reshape(1, -1).detach().numpy()[0]
        term1 = np.abs(self.integrator(eta-rs,ts))
        term2 = -R*(ts-ts[0])*(R*(ts-ts[0]))**j_max*np.log(np.ones(len(ts))-R*(ts-ts[0]))
        bound = term1+term2
        #print('term for X = '+str(np.max(term2)))
        return interp1d(self.ts_for_int, bound)

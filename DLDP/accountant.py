import numpy as np
import math
import scipy.stats

class MomentsAccountant:
    """矩母统计

    利用矩母函数估计隐私损失

    详见论文：Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]//Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016: 308-318.

    论文连接：https://arxiv.org/pdf/1607.00133.pdf%20.

    Examples
    --------
    >>> import MomentsAccount
    >>> accountant = MomentsAccount()
    >>> epsilon, delta = accountant.get_privacy_spent(4, 0.01, 10000, 1e-5)
    """

    def __init__(self, moment_orders=32):
        self.moment_orders = moment_orders

    def compute_moment(self, sigma, q, lmbd):
        lmbd_int = int(math.ceil(lmbd))
        if lmbd_int == 0:
            return 1.0

        a_lambda_first_term_exact = 0
        a_lambda_second_term_exact = 0
        for i in range(lmbd_int + 1):
            coef_i = scipy.special.binom(lmbd_int, i) * (q ** i) * (1 - q) ** (lmbd - i)
            s1, s2 = 0, 0
            s1 = coef_i * np.exp((i * i - i) / (2.0 * (sigma ** 2)))
            s2 = coef_i * np.exp((i * i + i) / (2.0 * (sigma ** 2)))
            a_lambda_first_term_exact += s1
            a_lambda_second_term_exact += s2

        a_lambda_exact = ((1.0 - q) * a_lambda_first_term_exact +
                          q * a_lambda_second_term_exact)

        return a_lambda_exact

    def compute_log_moment(self, sigma, q, steps):
        log_moments = []

        for lmbd in range(self.moment_orders + 1):
            log_moment = 0
            moment = self.compute_moment(sigma, q, lmbd)
            log_moment += np.log(moment) * steps
            log_moments.append((lmbd, log_moment))
        return log_moments

    def _compute_eps(self, log_moments, delta):
        min_eps = float("inf")

        for moment_order, log_moment in log_moments:
            if moment_order == 0:
                continue
            if math.isinf(log_moment) or math.isnan(log_moment):
                print("The %d-th order is inf or Nan\n" % moment_order)
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
            
        return min_eps

    def get_privacy_spent(self, sigma, q, steps, target_delta):
        log_moments = self.compute_log_moment(sigma, q, steps)

        return self._compute_eps(log_moments, target_delta), target_delta
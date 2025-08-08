# epsilon_tracker.py

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


class DynamicEpsilonTracker:
    def __init__(self, *, delta, sample_rate, noise_multiplier):
        self.delta = delta
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.rdp_total = None
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def step(self, current_C):
        # sensitivity = current_C
        sensitivity = max(current_C, 1e-6)
        effective_noise_multiplier = self.noise_multiplier

        print(f"[ε Tracker] C_t = {current_C:.6f}, effective σ = {effective_noise_multiplier:.6f}")

        rdp = compute_rdp(
            q=self.sample_rate,
            noise_multiplier=effective_noise_multiplier,
            steps=1,
            orders=self.orders,
        )

        if self.rdp_total is None:
            self.rdp_total = rdp
        else:
            self.rdp_total = [a + b for a, b in zip(self.rdp_total, rdp)]
        
        print(f"[ε Tracker] rdp_total[0] = {self.rdp_total[0]:.6f}")  # 添加调试信息



    def get_epsilon(self):
        if self.rdp_total is None:
            return 0.0
        eps, _ = get_privacy_spent(orders=self.orders, rdp=self.rdp_total, delta=self.delta)
        return eps
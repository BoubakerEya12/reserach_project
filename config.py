# sim/config.py
from dataclasses import dataclass

@dataclass
class SystemConfig:
    """
    Simulation parameters for one 'slot'.
    AWGN model: per-RB noise power sigma^2 = N0 * B_RB (Watts).
    """
    # Antennas / users / RBs
    M: int = 8           # BS antennas
    K: int = 16          # users
    N_RB: int = 8        # resource blocks in frequency
    U_max: int = 2       # max users per RB (<= M for ZF to work)
    R_max: int = 1       # enforce <=1 RB per user per slot

    # OFDM + power
    B_RB: float = 180e3  # Hz, RB bandwidth
    N0: float = 1e-20    # W/Hz (one-sided AWGN PSD)
    P_tot: float = 10.0  # W, total BS power budget per slot
    P_RB_max: float = 2.0  # W, per-RB cap

    # CSI quality (eq. (csit-model) in your LaTeX)
    eta: float = 0.8     # correlation between hat(h) and true effective channel

    # Reproducibility
    seed: int = 42       # set once, pass rng around for consistent draws

    # Pathloss/shadowing model (3GPP TR 38.901 UMa)
    pathloss_model: str = "3GPP_38.901_UMa"
    fc_GHz: float = 3.5
    h_bs: float = 25.0
    h_ut: float = 1.5
    los_mode: str = "prob"   # "prob" | "always_los" | "always_nlos"
    cell_radius_m: float = 250.0

    def noise_power(self) -> float:
        """Per-RB AWGN noise power σ² = N0 * B_RB (Watts)."""
        return float(self.N0 * self.B_RB)

    @property
    def sigma2(self) -> float:
        """Alias: cfg.sigma2 gives the same as cfg.noise_power()."""
        return self.noise_power()

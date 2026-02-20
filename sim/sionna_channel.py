# sim/sionna_channel.py
# ---------------------------------------------------------------------
# 3GPP TR 38.901 UMa channel generation with Sionna (single-cell DL)
#
# Outputs (RB-level, matching your paper notation):
#   h_rb_true : [B, K, N_RB, Nt] complex64
#   h_rb_hat  : [B, K, N_RB, Nt] complex64   (imperfect CSIT via eta-model)
#
# Notes:
# - We generate OFDM frequency response on N_sc = 12*N_RB subcarriers,
#   then average each group of 12 subcarriers to obtain RB-level channels.
# - normalize=False to preserve 3GPP path loss + shadowing in amplitudes.
# ---------------------------------------------------------------------

import inspect
import tensorflow as tf
from sionna.channel.tr38901 import UMa, AntennaArray
from sionna.channel import gen_single_sector_topology, subcarrier_frequencies, cir_to_ofdm_channel


def setup_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    return gpus


def complex_normal_tf(shape, dtype=tf.complex64):
    """e ~ CN(0,1) per complex entry."""
    re = tf.random.normal(shape, dtype=tf.float32)
    im = tf.random.normal(shape, dtype=tf.float32)
    z = tf.complex(re, im) / tf.cast(tf.sqrt(2.0), tf.complex64)
    return tf.cast(z, dtype)


def _build_uma(cfg):
    """
    Build UMa channel object for the Sionna version on Narval:
      UMa(carrier_frequency, o2i_model, ut_array, bs_array, direction, ...)
    """
    fc = tf.constant(float(cfg.fc_GHz) * 1e9, tf.float32)

    # BS array: 1 x Nt (ULA)
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=int(cfg.M),
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=fc,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
    )

    # UE array: 1 x 1
    ut_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=fc,
    )

    # Required by your Sionna build
    o2i_model = getattr(cfg, "o2i_model", "low")        # must be "low" or "high"
    direction = getattr(cfg, "direction", "downlink")   # "downlink" is typical

    channel = UMa(
        carrier_frequency=fc,
        o2i_model=o2i_model,
        ut_array=ut_array,
        bs_array=bs_array,
        direction=direction,
    )
    return channel, fc


def _make_topology(cfg, batch_size, fc):
    """
    Robust wrapper for gen_single_sector_topology across Sionna versions.
    Your Narval build does NOT accept carrier_frequency/ut_height/bs_height/cell_radius
    (at least not with these names), so we only pass what exists.
    """
    sig = inspect.signature(gen_single_sector_topology)
    params = sig.parameters

    kwargs = {}

    # Common/core args (supported in most versions)
    if "batch_size" in params:
        kwargs["batch_size"] = int(batch_size)
    if "num_ut" in params:
        kwargs["num_ut"] = int(cfg.K)
    if "scenario" in params:
        # Some versions accept "uma" or "UMa"; keep lower-case (Sionna usually handles it)
        kwargs["scenario"] = "uma"

    # Optional args (only added if your version supports them)
    if "carrier_frequency" in params:
        kwargs["carrier_frequency"] = fc

    if "ut_height" in params:
        kwargs["ut_height"] = float(cfg.h_ut)
    if "min_ut_height" in params:
        kwargs["min_ut_height"] = float(cfg.h_ut)
    if "max_ut_height" in params:
        kwargs["max_ut_height"] = float(cfg.h_ut)
    if "bs_height" in params:
        kwargs["bs_height"] = float(cfg.h_bs)

    # radius naming differs across versions
    if "cell_radius" in params:
        kwargs["cell_radius"] = float(cfg.cell_radius_m)
    if "cell_radius_m" in params:
        kwargs["cell_radius_m"] = float(cfg.cell_radius_m)
    if "max_ut_distance" in params:
        kwargs["max_ut_distance"] = float(cfg.cell_radius_m)
    if "min_ut_distance" in params:
        kwargs["min_ut_distance"] = 10.0
    if "isd" in params:
        kwargs["isd"] = float(2.0 * cfg.cell_radius_m)
    if "min_bs_ut_dist" in params:
        kwargs["min_bs_ut_dist"] = 10.0

    # If your older version uses different names, we can add them here later
    return gen_single_sector_topology(**kwargs)


def _squeeze_to_bkntsc(h_freq):
    """
    Convert Sionna output to [B, K, Nt, N_sc] by squeezing singleton axes in the middle.
    """
    h = h_freq
    while len(h.shape) > 4:
        squeezed = False
        for ax in range(2, len(h.shape) - 1):
            if h.shape[ax] == 1:
                h = tf.squeeze(h, axis=ax)
                squeezed = True
                break
        if not squeezed:
            break
    return h  # expected [B, K, Nt, N_sc]


def generate_h_rb_true(cfg, batch_size: int):
    """
    Generate RB-level true channels from Sionna UMa.
    Returns:
      h_rb_true : [B, K, N_RB, Nt] complex64
    """
    channel, fc = _build_uma(cfg)

    topo = _make_topology(cfg, batch_size, fc)

    N_RB = int(cfg.N_RB)
    N_sc = 12 * N_RB
    delta_f = tf.constant(float(getattr(cfg, "delta_f_hz", 15e3)), tf.float32)
    sampling_frequency = tf.constant(
        float(getattr(cfg, "sampling_frequency_hz", float(N_sc) * float(delta_f))),
        tf.float32,
    )

    # Older Sionna: set topology first, then call with (num_time_samples, sampling_frequency)
    if hasattr(channel, "set_topology"):
        try:
            if isinstance(topo, (tuple, list)):
                channel.set_topology(*topo)
            elif isinstance(topo, dict):
                channel.set_topology(**topo)
        except TypeError:
            # If topology doesn't match, we'll try the call-based API below
            pass

    call_sig = inspect.signature(channel.__call__)
    if "sampling_frequency" in call_sig.parameters and "num_time_samples" in call_sig.parameters:
        a, tau = channel(1, sampling_frequency)
    elif "sampling_frequency" in call_sig.parameters:
        a, tau = channel(topo, sampling_frequency=sampling_frequency)
    else:
        a, tau = channel(topo)

    freqs = subcarrier_frequencies(N_sc, delta_f)  # [N_sc]
    h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)

    h_bkntsc = tf.cast(_squeeze_to_bkntsc(h_freq), tf.complex64)

    # Expect [B, K, Nt, N_sc]
    if len(h_bkntsc.shape) != 4:
        raise ValueError(
            f"Unexpected channel tensor rank={len(h_bkntsc.shape)} shape={h_bkntsc.shape}. "
            f"Tip: print h_freq.shape to adapt squeezing for your Sionna version."
        )

    # RB averaging: reshape subcarriers into [N_RB, 12] and mean over 12
    B = tf.shape(h_bkntsc)[0]
    K = tf.shape(h_bkntsc)[1]
    Nt = tf.shape(h_bkntsc)[2]

    h = tf.reshape(h_bkntsc, [B, K, Nt, N_RB, 12])  # [B,K,Nt,N_RB,12]
    h_rb = tf.reduce_mean(h, axis=-1)              # [B,K,Nt,N_RB]
    h_rb = tf.transpose(h_rb, perm=[0, 1, 3, 2])   # [B,K,N_RB,Nt]
    return h_rb


@tf.function
def make_imperfect_csit(h_true, eta: float):
    """
    h_hat = eta*h_true + sqrt(1-eta^2)*e, e~CN(0,1)
    Input/Output: same shape as h_true (here [B,K,N_RB,Nt]).
    """
    eta = tf.cast(eta, tf.float32)
    scale_e = tf.sqrt(tf.maximum(1.0 - eta * eta, 0.0))
    e = complex_normal_tf(tf.shape(h_true), dtype=tf.complex64)
    return tf.cast(eta, tf.complex64) * h_true + tf.cast(scale_e, tf.complex64) * e


def generate_channels(cfg, batch_size: int):
    h_rb_true = generate_h_rb_true(cfg, batch_size)
    h_rb_hat = make_imperfect_csit(h_rb_true, float(cfg.eta))
    return h_rb_true, h_rb_hat


def debug_run(cfg, batch_size: int = 4):
    gpus = setup_gpu_memory_growth()
    print("GPUs detected:", [g.name for g in gpus])

    h_true, h_hat = generate_channels(cfg, batch_size)

    print("h_rb_true dtype:", h_true.dtype, "shape:", h_true.shape)
    print("h_rb_hat  dtype:", h_hat.dtype,  "shape:", h_hat.shape)

    p = tf.reduce_mean(tf.abs(h_true) ** 2)
    print("E[|h_rb_true|^2] ~", float(p.numpy()))


if __name__ == "__main__":
    try:
        from config import SystemConfig
    except Exception:
        from sim.config import SystemConfig

    cfg = SystemConfig()
    debug_run(cfg, batch_size=8)

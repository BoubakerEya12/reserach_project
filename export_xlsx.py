import numpy as np
import pandas as pd

IN_NPZ  = "data_test.npz"          # <-- change si besoin
OUT_XLS = "dataset_export.xlsx"    # fichier Excel de sortie

# ---------- charger le dataset ----------
d = np.load(IN_NPZ, allow_pickle=True)

H_true = d["H_true"]   # [S,K,N_RB,M] (gros, on ne l'exporte pas par défaut)
H_hat  = d["H_hat"]    # idem
Z      = d["Z"]        # [S,N_RB,K]
alpha  = d["alpha"]    # [S,N_RB]
phi    = d["phi"]      # [S,N_RB,K]
P      = d["powers"]   # [S,N_RB,K]
SINR   = d["sinr"]     # [S,N_RB,K]
sigma2 = float(d["sigma2"])

S, N_RB, K = Z.shape
# M = H_true.shape[3]  # si tu veux vérifier

# ---------- feuille "rb_user": table longue (sample, RB, user, Z, phi, power, SINR) ----------
rows = []
for s in range(S):
    # vecteurs indices pour RB et user
    n_idx, u_idx = np.meshgrid(np.arange(N_RB), np.arange(K), indexing="ij")  # shape [N_RB,K] chacun
    rows.append(pd.DataFrame({
        "sample": s,
        "RB": n_idx.ravel(),
        "user": u_idx.ravel(),
        "Z": Z[s].ravel().astype(float),
        "phi": phi[s].ravel().astype(float),
        "power_W": P[s].ravel().astype(float),
        "SINR_linear": SINR[s].ravel().astype(float),
    }))
df_rb_user = pd.concat(rows, ignore_index=True)
# Ajoute SINR en dB (masque les zéros pour éviter -inf)
with np.errstate(divide="ignore"):
    df_rb_user["SINR_dB"] = 10*np.log10(df_rb_user["SINR_linear"].replace(0.0, np.nan))

# ---------- feuille "alpha": alpha par RB ----------
rows_alpha = []
for s in range(S):
    rows_alpha.append(pd.DataFrame({
        "sample": s,
        "RB": np.arange(N_RB),
        "alpha": alpha[s].astype(float),
    }))
df_alpha = pd.concat(rows_alpha, ignore_index=True)

# ---------- feuille "summary": une ligne par sample ----------
summary_rows = []
for s in range(S):
    Zs = Z[s]            # [N_RB,K]
    Ps = P[s]            # [N_RB,K]
    SINRs = SINR[s]      # [N_RB,K]
    alphas = alpha[s]    # [N_RB]

    active_rb_mask = Ps.sum(axis=1) > 0.0       # RB actif si puissance totale > 0
    num_active_rb  = int(active_rb_mask.sum())
    total_pow_used = float(Ps.sum())            # somme totale (doit ≤ P_tot)
    users_per_rb   = Zs.sum(axis=1)             # nb users programmés par RB
    users_total    = int(Zs.sum())              # total users-RB actifs (≃ U_max * #RB actifs)

    nonzero_sinr = SINRs[SINRs > 0]
    if nonzero_sinr.size > 0:
        sinr_db = 10*np.log10(nonzero_sinr)
        sinr_min = float(np.min(sinr_db))
        sinr_med = float(np.median(sinr_db))
        sinr_max = float(np.max(sinr_db))
    else:
        sinr_min = sinr_med = sinr_max = float("nan")

    summary_rows.append(dict(
        sample=s,
        active_RBs=num_active_rb,
        total_power_W=total_pow_used,
        users_total=users_total,
        avg_users_per_active_RB=float(users_per_rb[active_rb_mask].mean()) if num_active_rb>0 else 0.0,
        alpha_mean=float(np.mean(alphas)),
        alpha_max=float(np.max(alphas)),
        sinr_min_dB=sinr_min,
        sinr_median_dB=sinr_med,
        sinr_max_dB=sinr_max,
        sigma2_W=sigma2,
    ))
df_summary = pd.DataFrame(summary_rows)

# ---------- écrire l'Excel ----------
with pd.ExcelWriter(OUT_XLS, engine="openpyxl") as writer:
    df_summary.to_excel(writer, index=False, sheet_name="summary")
    df_alpha.to_excel(writer,   index=False, sheet_name="alpha")
    # Si le tableau est grand, Excel supporte ~1,048,576 lignes max :
    # pour S=5, N_RB=8, K=16 on a 640 lignes → OK.
    df_rb_user.to_excel(writer, index=False, sheet_name="rb_user")

print(f"Excel écrit -> {OUT_XLS}")
print("Feuilles : summary, alpha, rb_user")

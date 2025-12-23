import random
import numpy as np
import matplotlib.pyplot as plt
import CodeMonteCarlo
from cross_sections import load_janis_sigma_csv, make_loglog_interp



def hist_counts(energies, bins):
    energies = list(energies)
    if len(energies) == 0:
        centers = np.sqrt(bins[:-1] * bins[1:])
        return centers, np.zeros(len(bins) - 1), bins

    if isinstance(energies[0], (tuple, list)) and len(energies[0]) >= 2:
        E = np.asarray([p[0] for p in energies], dtype=float)
        W = np.asarray([p[1] for p in energies], dtype=float)
        counts, edges = np.histogram(E, bins=bins, weights=W)
    else:
        E = np.asarray(energies, dtype=float)
        counts, edges = np.histogram(E, bins=bins)

    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, counts, edges


def reset_module_stats():
    # important: stats est global et cumule sinon
    CodeMonteCarlo.stats = {
        "enter_pe": 0,
        "exit_pe_left": 0,
        "exit_pe_right": 0,
        "enter_pb": 0,
        "exit_pb_left": 0,
        "exit_pb_right": 0,
    }

def smooth_nan(y, w=7):
    """Moyenne glissante centrée, ignore les NaN."""
    y = np.asarray(y, float)
    k = np.ones(w, dtype=float)

    num = np.convolve(np.nan_to_num(y, nan=0.0), k, mode="same")
    den = np.convolve(np.isfinite(y).astype(float), k, mode="same")
    out = num / np.maximum(den, 1e-12)
    out[den == 0] = np.nan
    return out

def interp_loglog(x_src, y_src, x_new):
    """
    Interpolation log-log de y(x) sur x_new.
    - ignore y<=0
    - extrapolation: clamp aux bords
    """
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_new = np.asarray(x_new, float)

    mask = (x_src > 0) & np.isfinite(x_src) & (y_src > 0) & np.isfinite(y_src)
    if np.sum(mask) < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    xs = x_src[mask]
    ys = y_src[mask]

    idx = np.argsort(xs)
    xs = xs[idx]
    ys = ys[idx]

    logx = np.log(xs)
    logy = np.log(ys)

    logx_new = np.log(np.clip(x_new, xs[0], xs[-1]))
    y_new = np.exp(np.interp(logx_new, logx, logy))
    return y_new


def run_once(N, seed, sample_on):
    CodeMonteCarlo.SAMPLE_ON = sample_on

    random.seed(seed)
    np.random.seed(seed)

    reset_module_stats()

    (hitsG, hitsM, passed,
     abs_fe, abs_pe, abs_pb, abs_au,
     energies_created, energies_G, energies_M) = CodeMonteCarlo.hit_count_GandM_detector(N)

    return {
        "hitsG": hitsG,
        "hitsM": hitsM,
        "passed": passed,
        "energies_G": energies_G,
        "energies_M": energies_M
    }
def au_number_density():
    NA = 6.02214076e23
    rho_Au = 19320.0          # kg/m^3
    A_Au   = 196.96657e-3     # kg/mol
    return rho_Au * NA / A_Au  # atoms/m^3

def load_janis_sigma_total_au():
    # total = capture (MT=102) + scatter elastic (MT=2)
    E_cap, sig_cap = load_janis_sigma_csv("captureAU.csv")
    E_scat, sig_scat = load_janis_sigma_csv("scatterAU.csv")

    f_cap = make_loglog_interp(E_cap, sig_cap)
    f_scat = make_loglog_interp(E_scat, sig_scat)

    def sigma_total_barns(E_eV: float) -> float:
        return f_cap(E_eV) + f_scat(E_eV)

    return sigma_total_barns


def interval_sigma_from_transmission(centers, I0, I1, edges_interval, gold_thick_m,
                                    min_sum_I0=50.0):
    """
    Calcule sigma_micro (barns) intervalle par intervalle via :
      T = sum(I1)/sum(I0)
      Sigma = -ln(T)/t
      sigma = Sigma/N_Au / 1e-28
    Retourne (E_mid, sigma_micro_barns, T, sumI0, sumI1)
    """
    centers = np.asarray(centers, float)
    I0 = np.asarray(I0, float)
    I1 = np.asarray(I1, float)

    N_Au = au_number_density()
    out_E = []
    out_sig = []
    out_T = []
    out_s0 = []
    out_s1 = []

    for a, b in zip(edges_interval[:-1], edges_interval[1:]):
        m = (centers >= a) & (centers < b) & np.isfinite(I0) & np.isfinite(I1) & (I0 >= 0) & (I1 >= 0)
        s0 = float(np.sum(I0[m]))
        s1 = float(np.sum(I1[m]))

        if s0 < min_sum_I0 or s1 <= 0:
            out_E.append(np.sqrt(a * b))
            out_sig.append(np.nan)
            out_T.append(np.nan)
            out_s0.append(s0)
            out_s1.append(s1)
            continue

        T = s1 / s0
        Sigma = -np.log(T) / gold_thick_m                 # [m^-1]
        sigma_barns = (Sigma / N_Au) / 1e-28              # barns

        out_E.append(np.sqrt(a * b))
        out_sig.append(sigma_barns)
        out_T.append(T)
        out_s0.append(s0)
        out_s1.append(s1)

    return np.array(out_E), np.array(out_sig), np.array(out_T), np.array(out_s0), np.array(out_s1)
def interval_sigma_janis_flux_weighted(centers, I0, edges_interval, sigma_janis_total):
    """
    Calcule la moyenne JANIS "vue" par ton spectre incident I0:
      <sigma> = sum(I0 * sigma(E))/sum(I0) sur l'intervalle
    """
    centers = np.asarray(centers, float)
    I0 = np.asarray(I0, float)

    out_E = []
    out_sig = []

    for a, b in zip(edges_interval[:-1], edges_interval[1:]):
        m = (centers >= a) & (centers < b) & np.isfinite(I0) & (I0 > 0)
        w = I0[m]
        if w.size < 2:
            out_E.append(np.sqrt(a * b))
            out_sig.append(np.nan)
            continue

        sig_vals = np.array([sigma_janis_total(float(e)) for e in centers[m]], dtype=float)
        sig_mean = float(np.sum(w * sig_vals) / np.sum(w))  # barns

        out_E.append(np.sqrt(a * b))
        out_sig.append(sig_mean)

    return np.array(out_E), np.array(out_sig)

def interval_mean_from_samples(centers, values, edges_interval, weights=None, min_points=2):
    """
    Moyenne par intervalle de 'values' échantillonnées sur 'centers'.
    Exemple: values = sigma(E) calculée à chaque bin center.
    - weights: si None -> moyenne simple, sinon moyenne pondérée.
    Retour: (E_mid, mean_val, npts)
    """
    centers = np.asarray(centers, float)
    values  = np.asarray(values, float)

    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, float)

    out_E, out_mean, out_n = [], [], []

    for a, b in zip(edges_interval[:-1], edges_interval[1:]):
        m = (centers >= a) & (centers < b) & np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
        n = int(np.count_nonzero(m))

        out_E.append(np.sqrt(a * b))
        out_n.append(n)

        if n < min_points:
            out_mean.append(np.nan)
            continue

        w = weights[m]
        v = values[m]
        ws = float(np.sum(w))
        out_mean.append(float(np.sum(w * v) / ws) if ws > 0 else np.nan)

    return np.array(out_E), np.array(out_mean), np.array(out_n)


def interval_mean_janis_unweighted(centers, edges_interval, sigma_janis_total):
    """
    Moyenne JANIS par intervalle SANS pondération flux:
      <sigma> = moyenne sur les centers qui tombent dans l'intervalle.
    (Si tu veux absolument l'intégrale continue, on peut faire une version trapz/logspace,
     mais ici c'est cohérent avec 'même morceaux' que tes bins.)
    """
    centers = np.asarray(centers, float)

    sig_on_centers = np.array([sigma_janis_total(float(e)) for e in centers], dtype=float)
    E_mid, sig_mean, npts = interval_mean_from_samples(
        centers, sig_on_centers, edges_interval, weights=None, min_points=2
    )
    return E_mid, sig_mean, npts



def compare_with_without_sample(
    N=200000,
    seed=42,
    nbins=300,
    do_sigma=True,
    # zone où tu veux moyenner (tu peux changer)

    n_intervals=10
):
    # mêmes bins pour les 2 runs (IMPORTANT)
    bins = np.logspace(np.log10(CodeMonteCarlo.E_MIN), np.log10(CodeMonteCarlo.E_MAX), nbins)

    out0 = run_once(N, seed, sample_on=False)
    out1 = run_once(N, seed, sample_on=True)

    centers, I0, edges = hist_counts(out0["energies_G"], bins)
    _,       I1, _     = hist_counts(out1["energies_G"], bins)

    dlnE = np.log(edges[1:] / edges[:-1])
    I0n = I0 / dlnE
    I1n = I1 / dlnE

    print("Total I0 (somme bins) =", float(np.sum(I0)))
    print("Total I1 (somme bins) =", float(np.sum(I1)))

    # --- plot spectre ---
    plt.figure(figsize=(10, 5))
    plt.xscale("log"); plt.yscale("log")
    m0 = I0n > 0
    m1 = I1n > 0
    plt.plot(centers[m0], I0n[m0], 'o-', markersize=2, label="without gold")
    plt.plot(centers[m1], I1n[m1], 'o-', markersize=2, label="with gold")
    plt.axvline(CodeMonteCarlo.E1); plt.axvline(CodeMonteCarlo.E2)
    plt.xlabel("E (eV)")
    plt.ylabel("counts / ln(E)")
    plt.title("Energy spectrum at detector G")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not do_sigma:
        return

    # --- intervalles d'énergie pour moyenner ---
    Emin = bins[0]
    Emax = bins[-1]

    edges_interval = np.logspace(
        np.log10(Emin),
        np.log10(Emax),
        n_intervals + 1
    )

    # --- sigma MC moyenne par intervalle via transmission ---
    t = float(CodeMonteCarlo.gold_thick)
    E_mid, sig_mc, T, s0, s1 = interval_sigma_from_transmission(
        centers, I0, I1, edges_interval, gold_thick_m=t,
        min_sum_I0=80.0  # tu peux monter si tu veux plus stable
    )

    # --- sigma JANIS moyenne (pondérée par le flux incident I0) ---
    sigma_janis_total = load_janis_sigma_total_au()
    E_mid2, sig_janis = interval_sigma_janis_flux_weighted(
        centers, I0, edges_interval, sigma_janis_total
    )
    # --- JANIS moyenne simple sur les mêmes intervalles (pas pondérée par I0) ---
    E_mid3, sig_janis_simple, npts_j = interval_mean_janis_unweighted(
        centers, edges_interval, sigma_janis_total
    )

    # --- plot comparaison: 1 point par intervalle + ligne ---
    plt.figure(figsize=(10, 5))
    plt.xscale("log")
    plt.yscale("log")

    ok_mc = np.isfinite(sig_mc) & (sig_mc > 0)
    ok_j  = np.isfinite(sig_janis) & (sig_janis > 0)

    # (optionnel) garder seulement les intervalles où les deux existent
    ok = ok_mc & ok_j

    plt.plot(E_mid[ok],  sig_mc[ok],  'o-', markersize=5, label="MC (transmission)")
    plt.plot(E_mid2[ok], sig_janis[ok], 's-', markersize=5, label="JANIS (flux-weighted)")

    plt.xlabel("E (eV) (interval geometric mean)")
    plt.ylabel("sigma_t (barns)")
    plt.title(f"Au microscopic total cross section (interval-averaged), {n_intervals} intervals")

    # petite box infos (IMPORTANT: avant show)
    txt = (f"N={N}\n"
           f"intervals={n_intervals} (log)\n"
           f"t={t:.2e} m")
    plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes,
             va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_with_without_sample(
        N=100000,
        seed=42,
        nbins=300,
        do_sigma=True,
        n_intervals=15
    )
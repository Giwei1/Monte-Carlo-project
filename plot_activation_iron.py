import random
import numpy as np
import matplotlib.pyplot as plt

# 1) Import explicite du Monte Carlo "normal"
import CodeMonteCarlo

# 2) Import du patch (il va modifier certaines fonctions de CodeMonteCarlo)
import CodeMonteCarlo_PEprotect as PF


def _activation_profile_from_abs_fe(abs_fe, nbins):
    if len(abs_fe) == 0:
        return None

    # abs_fe peut être [(x,y), ...] ou [(x,y,w), ...]
    x_abs = np.array([p[0] for p in abs_fe], dtype=float)

    # poids : si pas présent, on met 1.0
    if len(abs_fe[0]) >= 3:
        w_abs = np.array([p[2] for p in abs_fe], dtype=float)
    else:
        w_abs = np.ones_like(x_abs)

    depth_m = x_abs - CodeMonteCarlo.x_H
    iron_thickness_m = CodeMonteCarlo.x_J - CodeMonteCarlo.x_H

    bins = np.linspace(0.0, iron_thickness_m, nbins + 1)

    # ✅ Histogramme pondéré : activation = somme des poids par bin
    wsum, edges = np.histogram(depth_m, bins=bins, weights=w_abs)

    centers_m = 0.5 * (edges[:-1] + edges[1:])
    widths_m = np.diff(edges)

    # "activation par cm" = (somme des poids dans le bin) / (épaisseur du bin en cm)
    activation_per_cm = wsum / (widths_m * 100.0)  # m -> cm

    return centers_m * 100.0, activation_per_cm


def run_once(N, seed, nbins, pe_front):
    # Or ON (comme ton script)
    CodeMonteCarlo.SAMPLE_ON = True

    # Active/désactive la protection
    PF.PE_FRONT_ON = pe_front

    # Seed identique pour réduire le bruit de comparaison
    random.seed(seed)
    np.random.seed(seed)

    (hits_G, hits_M, passed,
     abs_fe, abs_pe, abs_pb, abs_au,
     energies_created, energies_G, energies_M) = CodeMonteCarlo.hit_count_GandM_detector(N)

    prof = _activation_profile_from_abs_fe(abs_fe, nbins)

    return {
        "pe_front": pe_front,
        "hits_G": hits_G,
        "hits_M": hits_M,
        "abs_fe": abs_fe,
        "abs_au": abs_au,
        "profile": prof
    }




def compare_activation_iron(N=300000, seed=42, nbins=80):
    out_no  = run_once(N, seed, nbins, pe_front=False)
    out_yes = run_once(N, seed, nbins, pe_front=True)

    def total_weighted_abs(abs_list):
        if len(abs_list) == 0:
            return 0.0
        if len(abs_list[0]) >= 3:
            return float(np.sum([p[2] for p in abs_list]))
        return float(len(abs_list))

    abs_no = total_weighted_abs(out_no["abs_fe"])
    abs_yes = total_weighted_abs(out_yes["abs_fe"])

    print("\n========== COMPARAISON ACTIVATION FER ==========")
    print(f"N={N}, seed={seed}, nbins={nbins}")

    print("\n--- Sans PE-front ---")
    print("Absorptions Fe :", abs_no)
    print("Absorptions Au :", len(out_no["abs_au"]))
    print("Hits G :", out_no["hits_G"], "Hits M :", out_no["hits_M"])

    print("\n--- Avec PE-front (5 cm avant H) ---")
    print("Absorptions Fe :", abs_yes)
    print("Absorptions Au :", len(out_yes["abs_au"]))
    print("Hits G :", out_yes["hits_G"], "Hits M :", out_yes["hits_M"])

    if abs_no > 0:
        ratio = abs_yes / abs_no
        diff_pct = 100.0 * (abs_yes - abs_no) / abs_no  # variation (peut être négative)
        red_pct = 100.0 * (1.0 - ratio)  # réduction (positive si ça diminue)

        print("\n--- Impact activation totale fer ---")
        print(f"Ratio (avec/sans) = {ratio:.4f}")
        print(f"Variation (%)      = {diff_pct:.2f}%")
        print(f"Réduction (%)      = {red_pct:.2f}%")
    else:
        ratio = np.nan
        diff_pct = np.nan
        red_pct = np.nan
        print("\n(abs_no = 0) Ratio impossible.")

    # Plot comparatif
    plt.figure(figsize=(10, 5))

    if out_no["profile"] is not None:
        x_no, a_no = out_no["profile"]
        plt.plot(x_no, a_no, label="Without PE-front")

    """if out_yes["profile"] is not None:
        x_yes, a_yes = out_yes["profile"]
        plt.plot(x_yes, a_yes, label="With PE-front (5 cm before iron)")"""

    plt.xlabel("Depth in the iron (cm)")
    plt.ylabel("Absorptions per cm")
    plt.title("Activation of iron without PE layer")
    plt.legend()
    # --- petite box avec la diminution ---
    """if np.isfinite(red_pct):
        txt = (f"Activation Fe (total)\n"
               f"with/without = {ratio:.3f}\n"
               f"Reduction = {red_pct:.2f}%")"""
    """else:
        txt = "Activation Fe (total)\nabs_no = 0"""""

    """plt.text(
        0.02, 0.98, "Activation Fe (total)\nabs_no = 0",
        transform=plt.gca().transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )"""
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_activation_iron(N=200000, seed=42, nbins=80)

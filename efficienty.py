import random
import math
import numpy as np
from cross_sections import macro_xs
import matplotlib.pyplot as plt
import time

"""
pour le test jai enlevé l'angle de depart qui est random dans le 2pi
et jai enlevé les slits, cest pour eviter troup d'iterations pour le test
changer les valeurs d'absorption et scatter

changer le poids des energies
"""

# constants
EPS = 1e-12
NUDGE = 1e-10
BARN = 1e-28

# geometrie du moderateur (m)
mod_x_min = 0.0
mod_x_max = 0.05
mod_y_min = -0.05
mod_y_max = 0.05

# --- Splitting (juste après les fentes) ---
USE_SPLITTING = False
SPLIT_N = 1000
# --- Splitting dans le plomb après 1ère interaction ---
USE_LEAD_SPLIT = True
LEAD_SPLIT_N = 50

# parametre energetique (anciens, gardés)
Tt_eV = 0.025
alpha = 0.95
Tf_eV = 0.805e6
Ek_eV = 0.78e6
E1 = 0.1
E2 = 8.0e5
E_MIN = 1e-4
E_MAX = 1e8

# slits opening (m)
slits = [
    (1.05, -0.075 / 2, 0.075 / 2),
    (2.05, -0.05 / 2, 0.05 / 2),
    (3.55, -0.01, 0.01),
    (4.55, -0.01, 0.01)
]

stats = {
    "enter_pe": 0,
    "exit_pe_left": 0,
    "exit_pe_right": 0,
    "enter_pb": 0,
    "exit_pb_left": 0,
    "exit_pb_right": 0,
}

# detecteur position G (m)
detector_y_G = [
    (-0.02, -0.01),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.01, 0.02)
]

# détecteur M
detector_M_x = 8.0615
detector_M_y = (-0.005, 0.005)

# limites globales
global_y_min = -0.15
global_y_max = 0.15

# positions (m)
x_F = 6.05
x_G = 6.05 + 36.25 / 100
x_H = x_G + 0.20
x_J = x_H + 1.0215
x_K = x_J + 0.05
x_L = x_K + 0.1775
x_M = detector_M_x

# régions
REGION_VACUUM = "vacuum"
REGION_GOLD = "gold"
REGION_IRON = "iron"
REGION_PE = "pe"
REGION_LEAD = "lead"
REGION_OUTSIDE = "outside"

# or
gold_thick = 1e-4
gold_ymin = -0.02
gold_ymax = 0.02

# activer l'or
SAMPLE_ON = True

SIGMA_c = {
    REGION_GOLD: 1.5,
    REGION_IRON: 1.0,
    REGION_PE: 0.5,
    REGION_LEAD: 2.0
}

P_ABS = {
    REGION_GOLD: 0.7,
    REGION_IRON: 0.3,
    REGION_PE: 0.1,
    REGION_LEAD: 0.5
}

# ============================================================
# PARAMÈTRES DU SPECTRE
# ============================================================
Tterm = 0.025
Tfiss = 0.805e6
Ekin_fast = 0.78e6
leakF = 0.95

E1 = 0.1
E2 = 800e3

E_MIN = 1e-4
E_MAX = 1e8

cnst = 1.05
A = (1.0 / E2**leakF) * np.exp(E2 / Tfiss)


def pdf_spectrum(E):
    if E <= E1:
        return 2 * np.pi / (np.pi * Tterm) ** (3 / 2) * np.sqrt(E) * np.exp(-E / Tterm)
    elif E <= E2:
        return 1.0 / E**leakF
    else:
        return A * np.exp(-E / Tfiss) * np.sinh(np.sqrt(4 * Ekin_fast * E / Tfiss**2))


E_star = 0.5 * Tterm
M1 = pdf_spectrum(E_star) if E_MIN <= E_star <= E1 else max(pdf_spectrum(E_MIN), pdf_spectrum(E1))

E_grid_high = np.logspace(np.log10(E2), np.log10(E_MAX), 8000)
M3 = float(np.max([pdf_spectrum(float(e)) for e in E_grid_high]))


def majorant(E):
    if E <= E1:
        return M1
    elif E <= E2:
        return 1.0 / E**leakF
    else:
        return M3


a = leakF
I0 = E_MIN * M1
I1 = -I0 + E1 * M1 - (E1 ** (1 - a)) / (1 - a)
I2 = I1 + (E2 ** (1 - a)) / (1 - a) - E2 * M3
I3 = I2 + E_MAX * M3


def MAJ(E):
    if E <= E1:
        return (-I0 + E * M1) / I3
    elif E <= E2:
        return (I1 + E ** (1 - a) / (1 - a)) / I3
    else:
        return (I2 + E * M3) / I3


II1 = MAJ(E1)
II2 = MAJ(E2)


def iMAJ(x):
    if x <= II1:
        return (x * I3 + I0) / M1
    elif x <= II2:
        return ((x * I3 - I1) * (1 - a)) ** (1 / (1 - a))
    else:
        return (x * I3 - I2) / M3


def sample_energy_rejection():
    while True:
        u = np.random.rand()
        E = iMAJ(u)

        if E < E_MIN or E > E_MAX:
            continue

        f = pdf_spectrum(E)
        g = majorant(E)

        if np.random.rand() < f / (cnst * g):
            return E


def plot_energy_comparison(energies, ngrid=2000, max_points=8000):
    energies = np.array(energies, dtype=float)
    energies = energies[np.isfinite(energies)]
    energies = energies[(energies >= E_MIN) & (energies <= E_MAX)]
    if energies.size == 0:
        print("Aucune énergie valide à tracer.")
        return

    if energies.size > max_points:
        idx = np.random.choice(energies.size, size=max_points, replace=False)
        E_pts = energies[idx]
    else:
        E_pts = energies

    E_grid = np.logspace(np.log10(E_MIN), np.log10(E_MAX), int(ngrid))
    f_vals = np.array([pdf_spectrum(float(e)) for e in E_grid], dtype=float)
    g_vals = np.array([majorant(float(e)) for e in E_grid], dtype=float)

    y_pts = np.array([pdf_spectrum(float(e)) for e in E_pts], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(E_grid, f_vals, "k-", lw=2, label="f(E) ")
    plt.plot(E_grid, g_vals, "r--", lw=2, label="g(E) majorant")
    plt.scatter(E_pts, y_pts, marker="x", c="orange", s=20, linewidths=0.8, alpha=0.9, label="energy of neutrons")
    plt.axvline(E1, color="gray", lw=1)
    plt.axvline(E2, color="gray", lw=1)
    plt.xlabel("Energy E (eV)")
    plt.ylabel("pdf(E) ")
    plt.title("Energy neutron source with rejection sampling : f(E), g(E) + points")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


class Neutron:
    def __init__(self):
        self.current_x = random.uniform(mod_x_min, mod_x_max)
        self.current_y = random.uniform(mod_y_min, mod_y_max)

        theta = 0  # random.uniform(0, 2*math.pi)
        self.current_dx = math.cos(theta)
        self.current_dy = math.sin(theta)
        self.current_angle = theta

        self.energy = sample_energy_rejection()
        self.w = 1.0
        self.split_count = 0
        self.lead_split_done = False
        self.alive = True

    def y_at_x_plane(self, x_plane):
        dx = self.current_dx
        dy = self.current_dy
        x0 = self.current_x
        y0 = self.current_y

        if abs(dx) < EPS:
            return None
        t = (x_plane - x0) / dx
        if t <= 0:
            return None
        return y0 + t * dy

    def clone(self):
        n = Neutron.__new__(Neutron)
        n.current_x = self.current_x
        n.current_y = self.current_y
        n.current_dx = self.current_dx
        n.current_dy = self.current_dy
        n.current_angle = self.current_angle
        n.energy = self.energy
        n.alive = self.alive
        n.w = self.w
        n.split_count = self.split_count
        n.lead_split_done = self.lead_split_done
        return n


def detector_efficiency(E):
    if E < 0.125:
        return 0.90
    elif E <= 10.0:
        return 0.10
    else:
        return 0.001


def split_neutron(neutron, nsplit):
    if nsplit <= 1:
        return [neutron]
    w_child = neutron.w / nsplit
    children = []
    for _ in range(nsplit):
        c = neutron.clone()
        c.w = w_child
        children.append(c)
    return children


def sample_isotropic_direction():
    theta = random.uniform(0.0, 2.0 * math.pi)
    return math.cos(theta), math.sin(theta), theta


# ====== MODE TEST : overrides de sections efficaces ======
USE_XS_OVERRIDE = True
XS_OVERRIDE = {
    # REGION_GOLD: {"Sigma_s": 0.0, "Sigma_a": 5e6},
    # "pe": {"Sigma_s": 5.0, "Sigma_a": 2.0},
    # "lead": {"Sigma_s": 1.0, "Sigma_a": 0.5},
}


def transport_region(neutron, x_left, x_right, region_name):
    Sigma_s, Sigma_a, Sigma_t, Pabs = macro_xs(region_name, neutron.energy)

    if USE_XS_OVERRIDE and region_name in XS_OVERRIDE:
        Sigma_s = XS_OVERRIDE[region_name]["Sigma_s"]
        Sigma_a = XS_OVERRIDE[region_name]["Sigma_a"]
        Sigma_t = Sigma_s + Sigma_a
        Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0

    if Sigma_t <= 0.0:
        Sigma_t = 0.0

    while neutron.alive:
        xi = random.random()
        l_collision = float("inf") if Sigma_t == 0.0 else -math.log(xi) / Sigma_t

        dx = neutron.current_dx
        dy = neutron.current_dy

        if dx > 0:
            dist_boundary = (x_right - neutron.current_x) / dx
            boundary_side = "right"
        elif dx < 0:
            dist_boundary = (x_left - neutron.current_x) / dx
            boundary_side = "left"
        else:
            dist_boundary = float("inf")
            boundary_side = None

        if region_name == REGION_GOLD:
            if dy > 0:
                dist_boundary_y = (gold_ymax - neutron.current_y) / dy
                if dist_boundary_y < dist_boundary:
                    dist_boundary = dist_boundary_y
                    boundary_side = "top"
            elif dy < 0:
                dist_boundary_y = (gold_ymin - neutron.current_y) / dy
                if dist_boundary_y < dist_boundary:
                    dist_boundary = dist_boundary_y
                    boundary_side = "bottom"

        if l_collision < dist_boundary:
            neutron.current_x += dx * l_collision
            neutron.current_y += dy * l_collision

            if not (global_y_min < neutron.current_y < global_y_max):
                neutron.alive = False
                return "lost", None, None

            if random.random() < Pabs:
                neutron.alive = False
                return "absorbed", neutron.current_x, neutron.current_y
            else:
                new_dx, new_dy, new_angle = sample_isotropic_direction()
                neutron.current_dx = new_dx
                neutron.current_dy = new_dy
                neutron.current_angle = new_angle
                continue
        else:
            neutron.current_x += dx * dist_boundary
            neutron.current_y += dy * dist_boundary

            if not (global_y_min < neutron.current_y < global_y_max):
                neutron.alive = False
                return "lost", None, None

            return boundary_side, None, None


def passes_slits(neutron):
    for (x_slit, y_min, y_max) in slits:
        y_at_slit = neutron.y_at_x_plane(x_slit)
        if y_at_slit is None:
            return False
        if not (y_min < y_at_slit < y_max):
            return False
    return True


def get_region(x, y):
    if y < global_y_min or y > global_y_max:
        return REGION_OUTSIDE

    if x < x_F:
        return REGION_VACUUM

    elif x_F <= x <= x_F + gold_thick and gold_ymin < y < gold_ymax:
        return REGION_GOLD if SAMPLE_ON else REGION_VACUUM

    elif x_F + gold_thick < x < x_G:
        return REGION_VACUUM

    elif x_H <= x < x_J:
        return REGION_IRON

    if x_J <= x < x_K:
        return REGION_PE

    if x_K <= x < x_L:
        return REGION_LEAD

    if x >= x_L:
        return REGION_VACUUM

    return REGION_OUTSIDE


def nudge(neutron):
    neutron.current_x += neutron.current_dx * NUDGE
    neutron.current_y += neutron.current_dy * NUDGE


def inside_gold(y):
    return gold_ymin < y < gold_ymax


def in_global_y(y):
    return global_y_min < y < global_y_max


def detector_hit_G_from_y(y):
    for i, (ymin, ymax) in enumerate(detector_y_G):
        if ymin < y < ymax:
            return i
    return None


def try_score_detector_G(neutron, hit_per_detector, energies_G=None):
    yG = neutron.y_at_x_plane(x_G)
    if yG is None:
        return 0.0
    idx = detector_hit_G_from_y(yG)
    if idx is None:
        return 0.0

    eps = detector_efficiency(neutron.energy)
    contrib = neutron.w * eps

    hit_per_detector[idx] += contrib
    if energies_G is not None:
        energies_G.append((neutron.energy, contrib))

    return contrib


def try_score_detector_M_weight(neutron, energies_M=None):
    yM = neutron.y_at_x_plane(detector_M_x)
    if yM is None:
        return 0.0
    if not (detector_M_y[0] < yM < detector_M_y[1]):
        return 0.0

    eps = 1.0  # tu avais mis eps=1 pour tester
    w_det = neutron.w * eps

    if energies_M is not None:
        energies_M.append((neutron.energy, w_det))

    return w_det


def enter_iron(neutron, hit_per_detector, absorptions_iron, energies_G=None):
    """
    Retourne (status, scoreG_added)
    status in {"dead","continue"}
    """
    res, x_abs, y_abs = transport_region(neutron, x_H, x_J, REGION_IRON)

    if res == "absorbed":
        absorptions_iron.append((x_abs, y_abs, neutron.w))
        return "dead", 0.0
    if res == "lost":
        return "dead", 0.0

    if res == "left":
        g = try_score_detector_G(neutron, hit_per_detector, energies_G)

        y_gold = neutron.y_at_x_plane(x_F + gold_thick)
        if y_gold is not None and inside_gold(y_gold):
            neutron.current_x = x_F + gold_thick
            neutron.current_y = y_gold
            nudge(neutron)
            return "continue", g

        return "dead", g

    if res == "right":
        nudge(neutron)
        return "continue", 0.0

    return "dead", 0.0


def enter_pe(neutron, absorptions_pe):
    stats["enter_pe"] += 1
    res, x_abs, y_abs = transport_region(neutron, x_J, x_K, REGION_PE)

    if res == "absorbed":
        absorptions_pe.append((x_abs, y_abs, neutron.w))
        return "dead"
    if res == "lost":
        return "dead"

    if res == "left":
        stats["exit_pe_left"] += 1
    elif res == "right":
        stats["exit_pe_right"] += 1
    nudge(neutron)
    return "continue"


def enter_lead(neutron, absorptions_lead):
    stats["enter_pb"] += 1
    res, x_abs, y_abs = transport_region(neutron, x_K, x_L, REGION_LEAD)

    if res == "absorbed":
        absorptions_lead.append((x_abs, y_abs, neutron.w))
        return "dead"
    if res == "lost":
        return "dead"

    if res == "left":
        stats["exit_pb_left"] += 1
    elif res == "right":
        stats["exit_pb_right"] += 1

    nudge(neutron)
    return "continue"


def enter_gold(neutron, hit_per_detector, absorptions_gold, energies_G=None):
    """
    Retourne (status, scoreG_added)
    """
    res, x_abs, y_abs = transport_region(neutron, x_F, x_F + gold_thick, REGION_GOLD)

    if res == "absorbed":
        absorptions_gold.append((x_abs, y_abs, neutron.w))
        return "dead", 0.0
    if res == "lost":
        return "dead", 0.0

    if res == "left":
        return "dead", 0.0

    if res in ("top", "bottom"):
        y_iron = neutron.y_at_x_plane(x_H)
        if y_iron is not None and in_global_y(y_iron):
            neutron.current_x = x_H
            neutron.current_y = y_iron
            nudge(neutron)
            return "continue", 0.0
        return "dead", 0.0

    if res == "right":
        g = try_score_detector_G(neutron, hit_per_detector, energies_G)

        y_iron = neutron.y_at_x_plane(x_H)
        if y_iron is not None and in_global_y(y_iron):
            neutron.current_x = x_H
            neutron.current_y = y_iron
            nudge(neutron)
            return "continue", g

        return "dead", g

    return "dead", 0.0


def initial_injection(neutron, hit_per_detector, energies_G=None):
    """
    Retourne (ok, scoreG_added)
    """
    y_gold_at_xF = neutron.y_at_x_plane(x_F)
    score_G = 0.0

    if SAMPLE_ON and (y_gold_at_xF is not None) and inside_gold(y_gold_at_xF):
        neutron.current_x = x_F
        neutron.current_y = y_gold_at_xF
        nudge(neutron)
        return True, 0.0

    score_G += try_score_detector_G(neutron, hit_per_detector, energies_G)

    y_iron = neutron.y_at_x_plane(x_H)
    if (y_iron is not None) and in_global_y(y_iron):
        neutron.current_x = x_H
        neutron.current_y = y_iron
        nudge(neutron)
        return True, score_G

    return False, score_G


def propagate_to_M_from_xL(neutron, energies_M=None):
    if neutron.current_dx <= 0:
        return 0.0
    return try_score_detector_M_weight(neutron, energies_M)


def run_one_neutron(neutron,
                    hit_per_detector,
                    absorptions_iron,
                    absorptions_pe,
                    absorptions_lead,
                    absorptions_gold,
                    energies_M=None,
                    energies_G=None):
    """
    Retourne (scoreG, scoreM)
    """
    score_G_total = 0.0

    ok, scoreG0 = initial_injection(neutron, hit_per_detector, energies_G)
    score_G_total += scoreG0
    if not ok:
        return 0.0, 0.0

    while neutron.alive:
        x, y = neutron.current_x, neutron.current_y
        reg = get_region(x, y)

        if (x_H - 1e-9) <= x <= (x_J + 1e-9) and reg == REGION_IRON:
            status, gadd = enter_iron(neutron, hit_per_detector, absorptions_iron, energies_G)
            score_G_total += gadd
            if status == "dead":
                return score_G_total, 0.0
            continue

        if reg == REGION_PE:
            status = enter_pe(neutron, absorptions_pe)
            if status == "dead":
                return score_G_total, 0.0
            continue

        if reg == REGION_LEAD:
            status = enter_lead(neutron, absorptions_lead)
            if status == "dead":
                return score_G_total, 0.0
            continue

        if reg == REGION_GOLD:
            status, gadd = enter_gold(neutron, hit_per_detector, absorptions_gold, energies_G)
            score_G_total += gadd
            if status == "dead":
                return score_G_total, 0.0
            continue

        if x >= x_L - 1e-9 and reg == REGION_VACUUM:
            score_M = propagate_to_M_from_xL(neutron, energies_M)
            return score_G_total, score_M

        neutron.alive = False
        return score_G_total, 0.0

    return score_G_total, 0.0


def hit_count_GandM_detector(num_neutrons):
    created = 0
    hit_per_detector = [0.0, 0.0, 0.0, 0.0]
    total_passing_slits = 0

    hit_M_total = 0.0

    scores_M = []
    scores_G = []

    absorptions_iron = []
    absorptions_pe = []
    absorptions_lead = []
    absorptions_gold = []

    energies_created = []
    energies_G = []
    energies_M = []

    for _ in range(num_neutrons):
        neutron = Neutron()
        created += 1
        energies_created.append(neutron.energy)

        if not passes_slits(neutron):
            neutron.alive = False
            scores_M.append(0.0)
            scores_G.append(0.0)
            continue

        total_passing_slits += 1

        neutrons_to_run = [neutron]
        if USE_SPLITTING:
            neutrons_to_run = split_neutron(neutron, SPLIT_N)

        source_score_M = 0.0
        source_score_G = 0.0

        for ntr in neutrons_to_run:
            g, m = run_one_neutron(
                ntr,
                hit_per_detector,
                absorptions_iron,
                absorptions_pe,
                absorptions_lead,
                absorptions_gold,
                energies_M,
                energies_G
            )
            source_score_G += g
            source_score_M += m

        scores_G.append(source_score_G)
        scores_M.append(source_score_M)
        hit_M_total += source_score_M

    print("Neutrons créés :", created)

    return (hit_per_detector, hit_M_total, total_passing_slits,
            absorptions_iron, absorptions_pe, absorptions_lead, absorptions_gold,
            energies_created, energies_G, energies_M,
            scores_G, scores_M, created)


def mc_stats(scores):
    S = np.asarray(scores, dtype=float)
    N = S.size
    if N < 2:
        return float(S.mean()), 0.0, 0.0, 0.0

    sumS = float(np.sum(S))
    sumS2 = float(np.sum(S * S))

    mean = sumS / N
    var_mean = (sumS2 - (sumS * sumS) / N) / (N * (N - 1))
    std_mean = math.sqrt(max(var_mean, 0.0))
    rel_err = std_mean / abs(mean) if mean != 0 else float("inf")

    return mean, var_mean, std_mean, rel_err


if __name__ == "__main__":
    N = 1_000_000

    t0 = time.perf_counter()
    (hits_G, hit_M_total, passed,
     abs_fe, abs_pe, abs_pb, abs_au,
     energies_created, energies_G, energies_M,
     scores_G, scores_M, created) = hit_count_GandM_detector(N)
    t1 = time.perf_counter()

    total_time = t1 - t0
    time_per_neutron = total_time / created

    # stats MC
    mean_M, var_mean_M, std_mean_M, rel_err_M = mc_stats(scores_M)
    mean_G, var_mean_G, std_mean_G, rel_err_G = mc_stats(scores_G)

    # efficiency MC
    mc_eff_M = 1.0 / (var_mean_M * total_time) if (var_mean_M > 0 and total_time > 0) else 0.0
    mc_eff_G = 1.0 / (var_mean_G * total_time) if (var_mean_G > 0 and total_time > 0) else 0.0

    print("\n--- MONTE CARLO EFFICIENCY ---")
    print("MC efficiency (M) = 1/(sigma^2 * tau) =", mc_eff_M)
    print("MC efficiency (G) = 1/(sigma^2 * tau) =", mc_eff_G)

    print("\n--- RESULTS ---")
    print("Neutrons passés par les fentes :", passed)
    print("Hits détecteur G (4 fenêtres):", hits_G)
    print("Hits détecteur M (total):", hit_M_total)

    print("\n--- MC ESTIMATOR (per source neutron) ---")
    print("Mean G =", mean_G, "   Std(mean G) =", std_mean_G, "   Rel error G =", rel_err_G)
    print("Mean M =", mean_M, "   Std(mean M) =", std_mean_M, "   Rel error M =", rel_err_M)

    print("\n--- CPU TIMING ---")
    print("Total CPU time (s):", total_time)
    print("CPU time per neutron (s/neutron):", time_per_neutron)

    print("\n--- ABSORPTIONS (weights) ---")
    print("Absorptions Fe :", sum(w for x, y, w in abs_fe))
    print("Absorptions PE :", sum(w for x, y, w in abs_pe))
    print("Absorptions Pb :", sum(w for x, y, w in abs_pb))
    print("Absorptions Au :", sum(w for x, y, w in abs_au))

    print("\n--- STATS ---")
    print("STATS:", stats)
    print("Pe traversés (sortie droite):", stats["exit_pe_right"])
    print("Pb traversés (sortie droite):", stats["exit_pb_right"])

import os
import numpy as np

# =========================
# Constantes + conversions
# =========================
BARN = 1e-28                 # 1 barn = 1e-28 m^2
NA   = 6.02214076e23         # Avogadro [mol^-1]


# ============================================================
# 1) Lecture CSV JANIS (énergie ; sigma) + interpolation log-log
# ============================================================

def _data_path(filename: str) -> str:
    """
    Assure qu'on lit les CSV dans le même dossier que cross_sections.py,
    même si tu lances main.py depuis un autre répertoire.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)


def load_janis_sigma_csv(filename: str):
    """
    Lit un export JANIS typique avec séparateur ';' :
      Incident energy ; Cross section
      1.00000E-05 ; 12.345
      ...

    Retourne (E_eV, sigma_barns) triés par énergie croissante.
    """
    path = _data_path(filename)

    E_list = []
    S_list = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Ignore headers : on ne garde que les lignes commençant par un nombre / '.' / '-'
            if line[0] not in "0123456789.-":
                continue

            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 2:
                continue

            # Au cas où JANIS met des virgules
            E_str = parts[0].replace(",", ".")
            S_str = parts[1].replace(",", ".")

            try:
                E = float(E_str)  # eV
                S = float(S_str)  # barns
            except ValueError:
                continue

            if E > 0 and S >= 0:
                E_list.append(E)
                S_list.append(S)

    if len(E_list) < 2:
        raise ValueError(
            f"Pas assez de données lues depuis {filename}. "
            f"Vérifie que le CSV est bien un export JANIS 'E ; sigma'."
        )

    E = np.array(E_list, dtype=float)
    S = np.array(S_list, dtype=float)

    idx = np.argsort(E)
    return E[idx], S[idx]


def make_loglog_interp(E_grid, y_grid):
    """
    Interpolation stable : log(y) en fonction de log(E) + clamp.
    Très important pour des cross sections (variations sur plusieurs ordres de grandeur).
    """
    E_grid = np.asarray(E_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    # éviter log(0)
    y_grid = np.maximum(y_grid, 1e-300)

    logE = np.log(E_grid)
    logY = np.log(y_grid)

    Emin = float(E_grid[0])
    Emax = float(E_grid[-1])

    def f(E):
        E = float(E)
        if E <= Emin:
            return float(np.exp(logY[0]))
        if E >= Emax:
            return float(np.exp(logY[-1]))
        return float(np.exp(np.interp(np.log(E), logE, logY)))

    return f


# =========================
# 2) Micro -> Macro
# =========================

def atomic_density(rho_kg_m3: float, A_kg_mol: float) -> float:
    """
    N = rho * NA / A
    """
    return (rho_kg_m3 * NA) / A_kg_mol


def sigma_micro_to_macro(N_atom_m3: float, sigma_barns: float) -> float:
    """
    Sigma_macro [m^-1] = N [m^-3] * sigma [m^2]
    sigma[m^2] = sigma_barns * 1e-28
    """
    return N_atom_m3 * sigma_barns * BARN


# ============================================================
# 3) Définition des matériaux (densités + tables JANIS)
# ============================================================

# --- Fer ---
RHO_FE = 7874.0           # kg/m^3
A_FE   = 55.845e-3        # kg/mol
N_FE   = atomic_density(RHO_FE, A_FE)

_E_cap_fe, _sig_cap_fe = load_janis_sigma_csv("captureIron.csv")   # MT=102
_E_scat_fe, _sig_scat_fe = load_janis_sigma_csv("scatterIron.csv") # MT=2

_sigma_cap_fe = make_loglog_interp(_E_cap_fe, _sig_cap_fe)
_sigma_scat_fe = make_loglog_interp(_E_scat_fe, _sig_scat_fe)


def macro_xs_iron(E_eV: float):
    """
    Fer:
      Sigma_s = diffusion élastique (MT=2)
      Sigma_a = capture (MT=102)
    """
    sig_s = _sigma_scat_fe(E_eV)
    sig_a = _sigma_cap_fe(E_eV)

    Sigma_s = sigma_micro_to_macro(N_FE, sig_s)
    Sigma_a = sigma_micro_to_macro(N_FE, sig_a)

    Sigma_t = Sigma_s + Sigma_a
    Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0
    return Sigma_s, Sigma_a, Sigma_t, Pabs


# --- Or (Au-197 ~ or naturel) ---
RHO_AU = 19320.0          # kg/m^3
A_AU   = 196.96657e-3     # kg/mol
N_AU   = atomic_density(RHO_AU, A_AU)

_E_cap_au, _sig_cap_au = load_janis_sigma_csv("captureAU.csv")     # MT=102
_E_scat_au, _sig_scat_au = load_janis_sigma_csv("scatterAU.csv")   # MT=2

_sigma_cap_au = make_loglog_interp(_E_cap_au, _sig_cap_au)
_sigma_scat_au = make_loglog_interp(_E_scat_au, _sig_scat_au)


def macro_xs_gold(E_eV: float):
    """
    Or:
      La résonance ~5 eV est contenue directement dans le CSV captureAU.csv
      donc l'interpolation la reproduit automatiquement.
    """
    sig_s = _sigma_scat_au(E_eV)
    sig_a = _sigma_cap_au(E_eV)

    Sigma_s = sigma_micro_to_macro(N_AU, sig_s)
    Sigma_a = sigma_micro_to_macro(N_AU, sig_a)

    Sigma_t = Sigma_s + Sigma_a
    Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0
    return Sigma_s, Sigma_a, Sigma_t, Pabs


# --- Plomb (Pb) ---
RHO_PB = 11340.0          # kg/m^3
A_PB   = 207.2e-3         # kg/mol
N_PB   = atomic_density(RHO_PB, A_PB)

_E_cap_pb, _sig_cap_pb = load_janis_sigma_csv("capturePb.csv")     # MT=102
_E_scat_pb, _sig_scat_pb = load_janis_sigma_csv("scatterPb.csv")   # MT=2

_sigma_cap_pb = make_loglog_interp(_E_cap_pb, _sig_cap_pb)
_sigma_scat_pb = make_loglog_interp(_E_scat_pb, _sig_scat_pb)


def macro_xs_lead(E_eV: float):
    """
    Plomb:
      Absorption faible, diffusion modérée.
    """
    sig_s = _sigma_scat_pb(E_eV)
    sig_a = _sigma_cap_pb(E_eV)

    Sigma_s = sigma_micro_to_macro(N_PB, sig_s)
    Sigma_a = sigma_micro_to_macro(N_PB, sig_a)

    Sigma_t = Sigma_s + Sigma_a
    Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0
    return Sigma_s, Sigma_a, Sigma_t, Pabs


# --- PE boré 5% bore naturel ---
# Hypothèse "raisonnable": 5% en masse de bore naturel, 95% en masse de PE (CH2)
# Densité globale approx (tu peux ajuster si ton énoncé donne une valeur)
RHO_PEB = 1000.0

W_B  = 0.05
W_PE = 0.95

# Masses molaires
A_H   = 1.008e-3
A_C   = 12.011e-3
A_BN  = 10.81e-3       # bore naturel (moyenne isotopique)
A_CH2 = 14.0e-3        # CH2

# Nombre de "molécules" CH2 par m3 (via fraction massique de PE)
N_CH2 = (W_PE * RHO_PEB * NA) / A_CH2

# Atomes par m3 dans la partie PE
N_H = 2.0 * N_CH2
N_C = 1.0 * N_CH2

# Bore naturel (5% massique)
N_B = (W_B * RHO_PEB * NA) / A_BN

# CSV JANIS (H, C scattering; B capture)
_E_scat_H, _sig_scat_H = load_janis_sigma_csv("scatterH.csv")      # MT=2
_E_scat_C, _sig_scat_C = load_janis_sigma_csv("scatterC.csv")      # MT=2
_E_cap_B,  _sig_cap_B  = load_janis_sigma_csv("captureBo.csv")     # MT=102 (B natural)
_E_scat_B, _sig_scat_B = load_janis_sigma_csv("scatterBo.csv")  # MT=2 (B natural)
_E_cap_H, _sig_cap_H = load_janis_sigma_csv("captureH.csv")        # MT=102
_E_cap_C, _sig_cap_C = load_janis_sigma_csv("captureC.csv")        # MT=102


_sigma_scat_B = make_loglog_interp(_E_scat_B, _sig_scat_B)
_sigma_scat_H = make_loglog_interp(_E_scat_H, _sig_scat_H)
_sigma_scat_C = make_loglog_interp(_E_scat_C, _sig_scat_C)
_sigma_cap_B  = make_loglog_interp(_E_cap_B,  _sig_cap_B)
_sigma_cap_H = make_loglog_interp(_E_cap_H, _sig_cap_H)
_sigma_cap_C = make_loglog_interp(_E_cap_C, _sig_cap_C)

def macro_xs_pe_boron(E_eV: float):
    """
    PE boré:
      Diffusion = H + C
      Absorption = B + H + C
    """
    # --- diffusion ---
    Sigma_s = (
        sigma_micro_to_macro(N_H, _sigma_scat_H(E_eV)) +
        sigma_micro_to_macro(N_C, _sigma_scat_C(E_eV))+
        sigma_micro_to_macro(N_B, _sigma_scat_B(E_eV))
    )

    # --- absorption (AJOUT H et C) ---
    Sigma_a = (
        sigma_micro_to_macro(N_B, _sigma_cap_B(E_eV)) +
        sigma_micro_to_macro(N_H, _sigma_cap_H(E_eV)) +
        sigma_micro_to_macro(N_C, _sigma_cap_C(E_eV))
    )

    Sigma_t = Sigma_s + Sigma_a
    Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0
    return Sigma_s, Sigma_a, Sigma_t, Pabs

# ============================================================
# 4) Une fonction UNIQUE que ton Monte Carlo peut appeler
# ============================================================

def macro_xs(region_name: str, E_eV: float):
    """
    Retourne (Sigma_s, Sigma_a, Sigma_t, Pabs) pour un matériau donné.
    region_name doit être une des chaînes que tu utilises dans main.py :
      "iron", "gold", "lead", "pe"
    """
    if region_name == "iron":
        return macro_xs_iron(E_eV)
    if region_name == "gold":
        return macro_xs_gold(E_eV)
    if region_name == "lead":
        return macro_xs_lead(E_eV)
    if region_name == "pe":
        return macro_xs_pe_boron(E_eV)

    # Pour vacuum/outside : pas d'interaction
    return 0.0, 0.0, 0.0, 0.0

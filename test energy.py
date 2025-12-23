import random
import math
import numpy as np
from cross_sections import macro_xs
import matplotlib.pyplot as plt


"""
pour le test jai enlevé l'angle de depart qui est random dans le 2pi
et jai enlevé les slits, cest pour eviter troup d'iterations pour le test
changer les valeurs d'absorption et scatter

changer le poids des energies
"""



#constants
EPS = 1e-12
NUDGE = 1e-10
BARN = 1e-28

#geometrie du moderateur (m)
mod_x_min = 0.0
mod_x_max = 0.05
mod_y_min = -0.05
mod_y_max = 0.05


# --- Splitting (juste après les fentes) ---
USE_SPLITTING = False     # <- tu mets False quand tu veux désactiver
SPLIT_N = 1000             # <- nombre de copies (2,5,10,...)
# --- Splitting dans le plomb après 1ère interaction ---
USE_LEAD_SPLIT = True
LEAD_SPLIT_N = 50   # ex: 20, 50, 100 (évite 1000 sinon explosion)


#parametre energetique
Tt_eV = 0.025              # eV (température thermique)
alpha = 0.95               # "leakage factor" utilisé comme exposant du 1/E^alpha
Tf_eV = 0.805e6            # 0.805 MeV -> eV
Ek_eV = 0.78e6             # 0.78 MeV  -> eV

E1 = 0.1                   # jonction Maxwell -> 1/E à 0.1 eV
E2 = 8.0e5                 # jonction 1/E -> Watt à 800 keV = 8e5 eV

E_MIN = 1e-4
E_MAX = 1e8


#slits opening (m)
slits = [
    (1.05, -0.075/2, 0.075/2),  # (x position, y bottom, y top
    (2.05, -0.05/2, 0.05/2),
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
stats.update({
    "lead_split_events": 0,          # nb de fois où on split dans le plomb
    "lead_split_children": 0,        # nb total de clones créés
    "lead_split_parent_w": 0.0,      # somme des poids des parents splittés
    "lead_split_children_w": 0.0,    # somme des poids des enfants (doit = parent_w)
})

#detecteur position G (m)

detector_y_G = [
    (-0.02, -0.01),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.01, 0.02)
]

# détecteur M
detector_M_x = 8.0615
detector_M_y = (-0.005, 0.005)#cest 0.005 normalement

# limites globales
global_y_min = -0.15
global_y_max = 0.15


# positions (m)
x_F = 6.05
x_G = 6.05 + 36.25/100
x_H = x_G + 0.20
x_J = x_H + 1.0215
x_K = x_J + 0.05
x_L = x_K + 0.1775
x_M = detector_M_x

#régions
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

#activer l'or
SAMPLE_ON = True


SIGMA_c = {
    REGION_GOLD: 1.5,
    REGION_IRON: 1.0,
    REGION_PE:   0.5,
    REGION_LEAD: 2.0
}

# Probabilité d’absorption lors d’une collision (le reste = diffusion)
# P_ABS = sigma_a / sigma_c(= sigma_s +sigma_a)
P_ABS = {
    REGION_GOLD: 0.7,
    REGION_IRON: 0.3,   # 30% absorption, 70% scattering
    REGION_PE:   0.1,
    REGION_LEAD: 0.5
}


#fonctions des distributions d'energie pour creation du neutron
# ============================================================
# PARAMÈTRES DU SPECTRE
# ============================================================
Tterm = 0.025        # eV
Tfiss = 0.805e6      # eV
Ekin_fast = 0.78e6   # eV
leakF = 0.95

E1 = 0.1             # join Maxwell → Fermi
E2 = 800e3           # join Fermi → Watt

E_MIN = 1e-4
E_MAX = 1e8

cnst = 1.05          # constante de sûreté rejection

A = (1.0 / E2**leakF) * np.exp(E2 / Tfiss)

# ============================================================
# PDF CIBLE (spectre joint MATLAB)
# ============================================================
def pdf_spectrum(E):
    if E <= E1:
        return 2*np.pi / (np.pi*Tterm)**(3/2) * np.sqrt(E) * np.exp(-E/Tterm)
    elif E <= E2:
        return 1.0 / E**leakF
    else:
        return A*np.exp(-E/Tfiss) * np.sinh(np.sqrt(4*Ekin_fast*E / Tfiss**2))


# ============================================================
# MAJORANT (comme dans le MATLAB)
# ============================================================
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


# ============================================================
# INTÉGRALE ANALYTIQUE DE LA MAJORANT (CDF)
# ============================================================
a = leakF

I0 = E_MIN * M1
I1 = -I0 + E1*M1 - (E1**(1-a))/(1-a)
I2 = I1 + (E2**(1-a))/(1-a) - E2*M3
I3 = I2 + E_MAX*M3   # normalisation


def MAJ(E):
    if E <= E1:
        return (-I0 + E*M1) / I3
    elif E <= E2:
        return (I1 + E**(1-a)/(1-a)) / I3
    else:
        return (I2 + E*M3) / I3


# ============================================================
# INVERSE DE LA CDF (iMAJ)
# ============================================================
II1 = MAJ(E1)
II2 = MAJ(E2)

def iMAJ(x):
    if x <= II1:
        return (x*I3 + I0) / M1
    elif x <= II2:
        return ((x*I3 - I1)*(1-a))**(1/(1-a))
    else:
        return (x*I3 - I2) / M3


# ============================================================
# REJECTION SAMPLING (MATLAB STYLE)
# ============================================================
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
# ============================================================

def plot_energy_comparison(energies, ngrid=2000, max_points=8000):
    energies = np.array(energies, dtype=float)
    energies = energies[np.isfinite(energies)]
    energies = energies[(energies >= E_MIN) & (energies <= E_MAX)]
    if energies.size == 0:
        print("Aucune énergie valide à tracer.")
        return

    # Sous-échantillonnage pour ne pas surcharger le plot
    if energies.size > max_points:
        idx = np.random.choice(energies.size, size=max_points, replace=False)
        E_pts = energies[idx]
    else:
        E_pts = energies

    # Grille pour les courbes
    E_grid = np.logspace(np.log10(E_MIN), np.log10(E_MAX), int(ngrid))
    f_vals = np.array([pdf_spectrum(float(e)) for e in E_grid], dtype=float)
    g_vals = np.array([majorant(float(e)) for e in E_grid], dtype=float)

    # IMPORTANT : pour vérifier que g>=f, on trace les valeurs BRUTES (pas normalisées)
    # mais comme c'est juste pour affichage, on peut aussi normaliser les deux
    # à la fin si tu veux comparer avec un histogramme.
    f_plot = f_vals
    g_plot = g_vals

    # Points: y = f(E_pts) (ça place les points sur la courbe cible)
    y_pts = np.array([pdf_spectrum(float(e)) for e in E_pts], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.xscale("log")
    plt.yscale("log")

    plt.plot(E_grid, f_plot, "k-", lw=2, label="f(E) ")
    plt.plot(E_grid, g_plot, "r--", lw=2, label="g(E) majorant")

    plt.scatter(
        E_pts, y_pts,
        marker="x",
        c="orange",
        s=20,
        linewidths=0.8,
        alpha=0.9,
        label="energy of neutrons"
    )
    plt.axvline(E1, color="gray", lw=1)
    plt.axvline(E2, color="gray", lw=1)

    plt.xlabel("Energy E (eV)")
    plt.ylabel("pdf(E) ")
    plt.title("Energy neutron source with rejection sampling : f(E), g(E) + points")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()






class Neutron():
    def __init__(self):
        #position actuelle
        self.current_x = random.uniform(mod_x_min, mod_x_max)
        self.current_y = random.uniform(mod_y_min, mod_y_max)

        #direction
        theta = 0#random.uniform(0, 2*math.pi)
        self.current_dx = math.cos(theta)
        self.current_dy = math.sin(theta)
        self.current_angle = theta

        #energie initiale (eV)
        self.energy = sample_energy_rejection()
        self.w = 1.0
        self.split_count = 0  # combien de splitting a déjà subi ce neutron
        self.lead_split_done = False  # pour ne split qu'une seule fois dans le plomb
        self.alive = True

    #fonction pour calculer l'intersection avec un plan x=constant quelconque
    def y_at_x_plane(self, x_plane):
        """
        Retourne y quand la trajectoire *actuelle* du neutron
        coupe le plan x = x_plane.

        - Si le neutron est parallèle à ce plan (dx = 0) → None
        - Si le plan se trouve "derrière" le neutron (par rapport à dx) → None
        """

        dx = self.current_dx
        dy = self.current_dy
        x0 = self.current_x
        y0 = self.current_y

        # Trajectoire parallèle à l'axe y → ne coupera jamais x = x_plane
        if abs(dx) < EPS:
            return None

        # paramètre de progression le long de la trajectoire
        t = (x_plane - x0) / dx

        # Si t <= 0, le plan est derrière le neutron dans sa direction de marche
        if t <= 0:
            return None

        # Sinon, on calcule y au point d'intersection
        y_plane = y0 + t * dy
        return y_plane

    def clone(self):
        n = Neutron.__new__(Neutron)   # évite de re-tirer énergie/position
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

#--------------------------------------
#fin de class
#--------------------------------------

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
    neutron.split_count += 1
    children = []
    for _ in range(nsplit):
        c = neutron.clone()
        c.w = w_child
        children.append(c)
    return children

def split_neutron_random_dir(neutron, nsplit):
    if nsplit <= 1:
        return [neutron]

    children = []
    w_child = neutron.w / nsplit
    stats["lead_split_events"] += 1
    stats["lead_split_parent_w"] += neutron.w

    for _ in range(nsplit):
        c = neutron.clone()
        c.w = w_child
        stats["lead_split_children"] += 1
        stats["lead_split_children_w"] += c.w

        # Chaque clone prend une direction isotrope ALÉATOIRE
        new_dx, new_dy, new_angle = sample_isotropic_direction()
        c.current_dx = new_dx
        c.current_dy = new_dy
        c.current_angle = new_angle

        # on compte ce splitting
        c.split_count = neutron.split_count + 1
        children.append(c)

    # le parent original ne sert plus (on le remplace par les clones)
    neutron.alive = False
    return children





#--------------------------------------
#fonction qui donne le free path en fonction de la region et qui donne une direction isotropique
#--------------------------------------


def sample_isotropic_direction():
    theta = random.uniform(0.0, 2.0 * math.pi)
    return math.cos(theta), math.sin(theta), theta



#--------------------------------------
#fonction qui donne le transport du neutron dans la region donnee
#--------------------------------------
# ====== MODE TEST : overrides de sections efficaces ======
USE_XS_OVERRIDE = True

XS_OVERRIDE = {
    #REGION_GOLD: {"Sigma_s": 0.0, "Sigma_a": 5e6},  # très absorbant (m^-1)
    #"pe":   {"Sigma_s": 5.0,  "Sigma_a": 2.0},   # en m^-1 (exagéré exprès)
    #"lead": {"Sigma_s": 1.0,  "Sigma_a": 0.5},   # en m^-1
}

def transport_region(neutron, x_left, x_right, region_name):
    """
    Transporte un neutron dans une région [x_left, x_right] remplie de `region_name`.

    Retour possible (à adapter à ton goût) :
      "absorbed"  : capturé dans la région
      "lost"      : sort des bornes globales en y
      "left"      : sort par x_left
      "right"     : sort par x_right
    """
    Sigma_s, Sigma_a, Sigma_t, Pabs = macro_xs(region_name, neutron.energy)
    # --- override test ---
    if USE_XS_OVERRIDE and region_name in XS_OVERRIDE:
        Sigma_s = XS_OVERRIDE[region_name]["Sigma_s"]
        Sigma_a = XS_OVERRIDE[region_name]["Sigma_a"]
        Sigma_t = Sigma_s + Sigma_a
        Pabs = Sigma_a / Sigma_t if Sigma_t > 0 else 0.0


    # sécurité : si Sigma_t==0 (vide) on évite division par 0
    if Sigma_t <= 0.0:
        # dans le vide : pas de collisions, donc on force "collision à l'infini"
        Sigma_t = 0.0

    while neutron.alive:

        # 1) distance libre avant collision dans ce matériau
        xi = random.random()
        # si Sigma_t == 0 -> pas de collision
        if Sigma_t == 0.0:
            l_collision = float("inf")
        else:
            l_collision = -math.log(xi) / Sigma_t

        dx = neutron.current_dx
        dy = neutron.current_dy


        # 2) distance jusqu'à la frontière en x, dans la direction actuelle

        if dx > 0:
            # on regarde la frontière de droite
            dist_boundary = (x_right - neutron.current_x) / dx     #cest la distance en 2D aucun soucis
            boundary_side = "right"


        elif dx < 0:
            # on regarde la frontière de gauche
            dist_boundary = (x_left - neutron.current_x) / dx
            boundary_side = "left"
        else:
            # dx = 0 → jamais de sortie en x, que des collisions
            dist_boundary = float("inf")
            boundary_side = None

        #si c'est de l'or on verif les boundaries du dessus aussi
        if region_name == REGION_GOLD:
            if dy > 0:
                y_top = gold_ymax
                dist_boundary_y = (y_top - neutron.current_y) / dy
                if dist_boundary is None or dist_boundary_y < dist_boundary:
                    dist_boundary = dist_boundary_y
                    boundary_side = "top"
            elif dy < 0:
                y_bottom = gold_ymin
                dist_boundary_y = (y_bottom - neutron.current_y) / dy
                if dist_boundary is None or dist_boundary_y < dist_boundary:
                    dist_boundary = dist_boundary_y
                    boundary_side = "bottom"

        # 3) Qui vient en premier ? collision ou frontière ?
        if l_collision < dist_boundary:
            # ---- Collision à l'intérieur de la région ----
            neutron.current_x += dx * l_collision
            neutron.current_y += dy * l_collision

            # check bornes en y
            if not (global_y_min < neutron.current_y < global_y_max):
                neutron.alive = False
                return "lost", None, None, None

            # absorption ou diffusion ?
            #random donne ente 0 et 1 et si plus petit que PABS alors on a absorption
            if random.random() < Pabs:
                neutron.alive = False
                return "absorbed", neutron.current_x, neutron.current_y, None

            else:
                # diffusion isotrope
                new_dx, new_dy, new_angle = sample_isotropic_direction()
                neutron.current_dx = new_dx
                neutron.current_dy = new_dy
                neutron.current_angle = new_angle

                # --- SPLIT dans le plomb après la 1ère interaction (collision) ---
                if (USE_LEAD_SPLIT
                        and region_name == REGION_LEAD
                        and (neutron.split_count <= 1)
                        and (not neutron.lead_split_done)):
                    neutron.lead_split_done = True

                    kids = split_neutron_random_dir(neutron, LEAD_SPLIT_N)

                    # On renvoie un "event" split + les clones
                    return "split", None, None, kids

                # sinon on continue normalement
                continue


        else:
            # ---- Le neutron atteint la frontière avant la collision ----
            neutron.current_x += dx * dist_boundary
            neutron.current_y += dy * dist_boundary

            # check bornes en y
            if not (global_y_min < neutron.current_y < global_y_max):
                neutron.alive = False
                return "lost", None, None, None

            return boundary_side, None, None,None








#--------------------------------------
#VERIFICATION DES FENTES: fonction qui verifie si le neutron passe a travers les fentes
#--------------------------------------

def passes_slits(neutron):
    for (x_slit, y_min, y_max) in slits:
        y_at_slit = neutron.y_at_x_plane(x_slit)

        #verification de si le neutron est dans la bonne direction
        if y_at_slit is None:
            return False
        #verification de si le neutron passes les fentes
        if not (y_min < y_at_slit < y_max):
            return False
    return True



# =============================
# REGION EN FONCTION DE x,y
# =============================

def get_region(x, y):
    """
    Retourne le type de région en fonction de (x,y).
    On se concentre sur la partie entre G et M (x >= x_G).
    """
    # si on sort des red dashed lines → perdu
    if y < global_y_min or y > global_y_max:
        return REGION_OUTSIDE

    # avant G, on ne s'en occupe pas ici
    if x < x_F:
        return REGION_VACUUM

    elif x_F <= x <= x_F+ gold_thick and gold_ymin<y<gold_ymax:
        return REGION_GOLD if SAMPLE_ON else REGION_VACUUM

    elif x_F+gold_thick < x < x_G:
        return REGION_VACUUM

    # J -> K : fer
    elif x_H <= x < x_J:
        return REGION_IRON

    # K -> L : PE boré
    if x_J <= x < x_K:
        return REGION_PE

    # L -> M : plomb
    if x_K <= x < x_L:
        return REGION_LEAD

    # au-delà de M, on laisse le test de M décider, sinon outside
    if x >= x_L:
        return REGION_VACUUM

    return REGION_OUTSIDE


def nudge(neutron):
    neutron.current_x += neutron.current_dx * NUDGE
    neutron.current_y += neutron.current_dy * NUDGE

def inside_gold(y):
    return (gold_ymin < y < gold_ymax)

def in_global_y(y):
    return (global_y_min < y < global_y_max)



def detector_hit_G_from_y(y):
    for i, (ymin, ymax) in enumerate(detector_y_G):
        if ymin < y < ymax:
            return i
    return None

#--------------------------------------
# Ici la fonction va calculer les hits dans le detecteur G et dans M
#--------------------------------------



def try_score_detector_G(neutron, hit_per_detector, energies_G=None):
    yG = neutron.y_at_x_plane(x_G)
    if yG is None:
        return
    idx = detector_hit_G_from_y(yG)
    if idx is None:
        return

    if random.random() < detector_efficiency(neutron.energy):
        hit_per_detector[idx] += neutron.w
        if energies_G is not None:
            energies_G.append((neutron.energy, neutron.w))



def try_score_detector_M(neutron, energies_M=None):
    yM = neutron.y_at_x_plane(detector_M_x)
    if yM is None:
        return False
    if not (detector_M_y[0] < yM < detector_M_y[1]):
        return False

   #ok = (random.random() < detector_efficiency(neutron.energy))
    ok = True
    if ok and energies_M is not None:
        energies_M.append((neutron.energy, neutron.w))
    return ok






#--------------------------------------
#fonctions de ce qui se passes aux frontieres avec les neutrons, car cela ne change rien au resultatfinal
#--------------------------------------



def enter_iron(neutron, hit_per_detector, absorptions_iron, energies_G=None):
    res, x_abs, y_abs, kids = transport_region(neutron, x_H, x_J, REGION_IRON)

    if res == "absorbed":
        absorptions_iron.append((x_abs, y_abs, neutron.w))
        return "dead", None

    if res == "lost":
        return "dead", None

    if res == "left":
        try_score_detector_G(neutron, hit_per_detector, energies_G)
        return "dead", None

    if res == "right":
        nudge(neutron)
        return "continue", None

    return "dead", None

def enter_pe(neutron, absorptions_pe):
    stats["enter_pe"] += 1
    res, x_abs, y_abs, kids = transport_region(neutron, x_J, x_K, REGION_PE)

    if res == "absorbed":
        absorptions_pe.append((x_abs, y_abs, neutron.w))
        return "dead", None

    if res == "lost":
        return "dead", None

    if res == "left":
        stats["exit_pe_left"] += 1
    elif res == "right":
        stats["exit_pe_right"] += 1

    nudge(neutron)
    return "continue", None


def enter_lead(neutron, absorptions_lead):
    stats["enter_pb"] += 1
    res, x_abs, y_abs, kids = transport_region(neutron, x_K, x_L, REGION_LEAD)

    if res == "absorbed":
        absorptions_lead.append((x_abs, y_abs, neutron.w))
        return "dead", None

    if res == "lost":
        return "dead", None

    if res == "split":
        return "split", kids

    if res == "left":
        stats["exit_pb_left"] += 1
    elif res == "right":
        stats["exit_pb_right"] += 1

    nudge(neutron)
    return "continue", None





def enter_gold(neutron, hit_per_detector, absorptions_gold, energies_G=None):
    res, x_abs, y_abs, kids = transport_region(
        neutron, x_F, x_F + gold_thick, REGION_GOLD
    )

    if res == "absorbed":
        absorptions_gold.append((x_abs, y_abs, neutron.w))
        return "dead", None

    if res == "lost":
        return "dead", None

    if res == "left":
        return "dead", None

    if res in ("top", "bottom"):
        y_iron = neutron.y_at_x_plane(x_H)
        if y_iron is not None and in_global_y(y_iron):
            neutron.current_x = x_H
            neutron.current_y = y_iron
            nudge(neutron)
            return "continue", None
        return "dead", None

    if res == "right":
        try_score_detector_G(neutron, hit_per_detector, energies_G)
        y_iron = neutron.y_at_x_plane(x_H)
        if y_iron is not None and in_global_y(y_iron):
            neutron.current_x = x_H
            neutron.current_y = y_iron
            nudge(neutron)
            return "continue", None
        return "dead", None

    return "dead", None



def initial_injection(neutron, hit_per_detector, energies_G=None):
    """
    Après les fentes, on injecte le neutron :
    - si SAMPLE_ON et la trajectoire coupe l'or à x_F -> on démarre dans l'or (x_F)
    - sinon (pas d'or OU il rate l'or) : on score le détecteur G au plan x_G,
      puis on place le neutron sur la face d'entrée du fer (x_H) si possible.
    """
    y_gold_at_xF = neutron.y_at_x_plane(x_F)

    # 1) Injection dans l'or seulement si l'or est activé
    if SAMPLE_ON and (y_gold_at_xF is not None) and inside_gold(y_gold_at_xF):
        neutron.current_x = x_F
        neutron.current_y = y_gold_at_xF
        nudge(neutron)
        return True

    # 2) Sinon : vide jusqu'au fer -> scorer G avant d'entrer dans le fer
    try_score_detector_G(neutron, hit_per_detector, energies_G)

    # 3) Puis on place le neutron sur la face d'entrée du fer
    y_iron = neutron.y_at_x_plane(x_H)
    if (y_iron is not None) and in_global_y(y_iron):
        neutron.current_x = x_H
        neutron.current_y = y_iron
        nudge(neutron)
        return True

    return False

def propagate_to_M_from_xL(neutron, energies_M=None):
    if neutron.current_dx <= 0:
        return False
    return try_score_detector_M(neutron, energies_M)




def run_one_neutron(neutron,
                    hit_per_detector,
                    absorptions_iron,
                    absorptions_pe,
                    absorptions_lead,
                    absorptions_gold,
                    energies_M=None,
                    energies_G=None):
    """
    Retourne un poids (float) = somme des poids qui hit M
    (car splitting => plusieurs contributions)
    """

    hitM_weight = 0.0

    # pile de neutrons à traiter (DFS)
    stack = [neutron]

    while stack:
        n = stack.pop()

        # injection (or ou fer) pour chaque neutron
        if not initial_injection(n, hit_per_detector, energies_G):
            continue

        while n.alive:
            x, y = n.current_x, n.current_y
            reg = get_region(x, y)

            if (x_H - 1e-9) <= x <= (x_J + 1e-9) and reg == REGION_IRON:
                status, kids = enter_iron(n, hit_per_detector, absorptions_iron, energies_G)
                if status == "dead":
                    break
                # pas de kids ici
                continue

            if reg == REGION_PE:
                status, kids = enter_pe(n, absorptions_pe)
                if status == "dead":
                    break
                continue

            if reg == REGION_LEAD:
                status, kids = enter_lead(n, absorptions_lead)

                if status == "split":
                    # on empile les clones et on arrête ce neutron (déjà tué dans split)
                    stack.extend(kids)
                    break

                if status == "dead":
                    break
                continue

            if reg == REGION_GOLD:
                status, kids = enter_gold(n, hit_per_detector, absorptions_gold, energies_G)
                if status == "dead":
                    break
                continue

            # vide après plomb : test M puis mort
            if x >= x_L - 1e-9 and reg == REGION_VACUUM:
                ok = propagate_to_M_from_xL(n, energies_M)
                if ok:
                    hitM_weight += n.w
                break

            n.alive = False
            break

    return hitM_weight


def hit_count_GandM_detector(num_neutrons):

    created = 0
    hit_per_detector = [0,0,0,0]
    total_passing_slits = 0
    hit_M = 0


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

        #verifie les slits

        """if not passes_slits(neutron):
            neutron.alive = False
            continue"""

        #le neutron passe les fentes
        total_passing_slits += 1

        #beginning of splitting
        neutrons_to_run = [neutron]
        if USE_SPLITTING:
            neutrons_to_run = split_neutron(neutron, SPLIT_N)

        for ntr in neutrons_to_run:
            w_hit = run_one_neutron(
                ntr,
                hit_per_detector,
                absorptions_iron,
                absorptions_pe,
                absorptions_lead,
                absorptions_gold,
                energies_M,
                energies_G
            )
            hit_M += w_hit

    print("Neutrons créés :", created)

    return hit_per_detector, hit_M, total_passing_slits, absorptions_iron, absorptions_pe, absorptions_lead, absorptions_gold, energies_created, energies_G, energies_M


if __name__ == "__main__":

    hits_G, hits_M, passed, abs_fe, abs_pe, abs_pb, abs_au, energies_created, energies_G, energies_M = hit_count_GandM_detector(
        100000)

    #plot_energy_comparison(energies_created,ngrid= 2000)

    print("Neutrons passés par les fentes :", passed)
    print("Hits détecteur G (4 fenêtres):", hits_G)
    print("Hits détecteur M :", hits_M)
    print("Absorptions Fe :", sum(w for x, y, w in abs_fe))
    print("Absorptions PE :", sum(w for x, y, w in abs_pe))
    print("Absorptions Pb :", sum(w for x, y, w in abs_pb))
    print("Absorptions Au :", sum(w for x, y, w in abs_au))

    print("STATS:", stats)
    print("Pe traversés (sortie droite):", stats["exit_pe_right"])
    print("Pb traversés (sortie droite):", stats["exit_pb_right"])
    print("\n--- SPLITTING LEAD DEBUG ---")
    print("Lead split events:", stats["lead_split_events"])
    print("Lead split children:", stats["lead_split_children"])
    if stats["lead_split_events"] > 0:
        ratio = stats["lead_split_children_w"] / max(stats["lead_split_parent_w"], 1e-30)
        print("Sum parent w:", stats["lead_split_parent_w"])
        print("Sum children w:", stats["lead_split_children_w"])
        print("children_w / parent_w =", ratio, "(doit être ~1.0)")
    else:
        print("Aucun splitting plomb déclenché (normal si peu de scatters dans Pb).")


import CodeMonteCarlo as MC

# --- paramètres PE-front ---
PE_FRONT_ON = True
PE_FRONT_THICK = 0.05
x_H_front_left = MC.x_H - PE_FRONT_THICK
x_H_front_right = MC.x_H
REGION_PE_FRONT = "pe_front"

# -----------------------------
# 1) Patch get_region
# -----------------------------
_original_get_region = MC.get_region

def get_region_pe_front(x, y):
    if PE_FRONT_ON and (x_H_front_left <= x < x_H_front_right):
        return REGION_PE_FRONT
    return _original_get_region(x, y)

MC.get_region = get_region_pe_front


# -----------------------------
# 2) Nouveau transport PE-front
# -----------------------------
def enter_pe_front(neutron, absorptions_pe_front):
    # même matériau que REGION_PE -> on réutilise les XS "pe"
    res, x_abs, y_abs = MC.transport_region(neutron, x_H_front_left, x_H_front_right, MC.REGION_PE)

    if res == "absorbed":
        absorptions_pe_front.append((x_abs, y_abs))
        return "dead"
    if res == "lost":
        return "dead"

    MC.nudge(neutron)
    return "continue"


# -----------------------------
# 3) Patch initial_injection
#    (cas "pas d'or": au lieu d'injecter à x_H, on injecte à x_H_front_left)
# -----------------------------
_original_initial_injection = MC.initial_injection

def initial_injection_pe_front(neutron, hit_per_detector, energies_G=None):
    """
    Copie de la logique de MC.initial_injection, mais:
    - si le neutron doit être placé sur la face d'entrée du fer,
      on le place sur la face d'entrée du PE-front (x_H_front_left).
    """
    y_gold_at_xF = neutron.y_at_x_plane(MC.x_F)

    # 1) Injection dans l'or (inchangé)
    if MC.SAMPLE_ON and (y_gold_at_xF is not None) and (MC.gold_ymin < y_gold_at_xF < MC.gold_ymax):
        neutron.current_x = MC.x_F
        neutron.current_y = y_gold_at_xF
        MC.nudge(neutron)
        return True

    # 2) Sinon : vide jusqu'au fer -> scorer G (inchangé)
    MC.try_score_detector_G(neutron, hit_per_detector, energies_G)

    # 3) Placement : au lieu de x_H, on vise x_H_front_left si PE-front ON
    x_target = x_H_front_left if PE_FRONT_ON else MC.x_H

    y_target = neutron.y_at_x_plane(x_target)
    if (y_target is not None) and (MC.global_y_min < y_target < MC.global_y_max):
        neutron.current_x = x_target
        neutron.current_y = y_target
        MC.nudge(neutron)
        return True

    return False

MC.initial_injection = initial_injection_pe_front


# -----------------------------
# 4) Patch enter_gold
#    (cas "sortie de l'or vers le fer": au lieu de placer à x_H, placer à x_H_front_left)
# -----------------------------
_original_enter_gold = MC.enter_gold

def enter_gold_pe_front(neutron, hit_per_detector, absorptions_gold, energies_G=None):
    res, x_abs, y_abs = MC.transport_region(neutron, MC.x_F, MC.x_F + MC.gold_thick, MC.REGION_GOLD)

    if res == "absorbed":
        absorptions_gold.append((x_abs, y_abs))
        return "dead"
    if res == "lost":
        return "dead"
    if res == "left":
        return "dead"

    if res in ("top", "bottom"):
        # sortie en y : vise la "barrière avant fer"
        x_target = x_H_front_left if PE_FRONT_ON else MC.x_H
        y_iron = neutron.y_at_x_plane(x_target)
        if y_iron is not None and (MC.global_y_min < y_iron < MC.global_y_max):
            neutron.current_x = x_target
            neutron.current_y = y_iron
            MC.nudge(neutron)
            return "continue"
        return "dead"

    if res == "right":
        # sortie côté droit de l'or : scorer G puis aller vers "avant fer"
        MC.try_score_detector_G(neutron, hit_per_detector, energies_G)

        x_target = x_H_front_left if PE_FRONT_ON else MC.x_H
        y_iron = neutron.y_at_x_plane(x_target)
        if y_iron is not None and (MC.global_y_min < y_iron < MC.global_y_max):
            neutron.current_x = x_target
            neutron.current_y = y_iron
            MC.nudge(neutron)
            return "continue"
        return "dead"

    return "dead"

MC.enter_gold = enter_gold_pe_front


# -----------------------------
# 5) Patch run_one_neutron
#    (juste ajouter le cas pe_front)
# -----------------------------
_original_run_one_neutron = MC.run_one_neutron

def run_one_neutron_pe_front(
    neutron, hit_per_detector,
    absorptions_iron, absorptions_pe, absorptions_lead, absorptions_gold,
    energies_M=None, energies_G=None,
    absorptions_pe_front=None
):
    if absorptions_pe_front is None:
        absorptions_pe_front = []

    # injection -> désormais injecte à x_H_front_left quand pas d'or
    if not MC.initial_injection(neutron, hit_per_detector, energies_G):
        return False

    while neutron.alive:
        x, y = neutron.current_x, neutron.current_y
        region = MC.get_region(x, y)

        if region == REGION_PE_FRONT:
            status = enter_pe_front(neutron, absorptions_pe_front)
            if status == "dead":
                return False
            continue

        # le reste inchangé
        if region == MC.REGION_IRON:
            status = MC.enter_iron(neutron, hit_per_detector, absorptions_iron, energies_G)
            if status == "dead":
                return False
            continue

        if region == MC.REGION_PE:
            status = MC.enter_pe(neutron, absorptions_pe)
            if status == "dead":
                return False
            continue

        if region == MC.REGION_LEAD:
            status = MC.enter_lead(neutron, absorptions_lead)
            if status == "dead":
                return False
            continue

        if region == MC.REGION_GOLD:
            status = MC.enter_gold(neutron, hit_per_detector, absorptions_gold, energies_G)
            if status == "dead":
                return False
            continue

        if x >= MC.x_L - 1e-9 and region == MC.REGION_VACUUM:
            return MC.propagate_to_M_from_xL(neutron, energies_M)

        neutron.alive = False
        return False

    return False

MC.run_one_neutron = run_one_neutron_pe_front



def run_PEfront_simulation(N=100000):
    created = 0
    hit_per_detector = [0, 0, 0, 0]
    hit_M = 0
    passed = 0

    absorptions_iron = []
    absorptions_pe = []
    absorptions_pe_front = []
    absorptions_lead = []
    absorptions_gold = []

    energies_created = []
    energies_G = []
    energies_M = []

    for _ in range(N):
        neutron = MC.Neutron()
        created += 1
        energies_created.append(neutron.energy)

        passed += 1  # slits désactivées comme dans ton code test

        hitM = MC.run_one_neutron(
            neutron,
            hit_per_detector,
            absorptions_iron,
            absorptions_pe,
            absorptions_lead,
            absorptions_gold,
            energies_M,
            energies_G,
            absorptions_pe_front
        )

        if hitM:
            hit_M += 1

    # ======================
    # PRINT DES RESULTATS
    # ======================
    print("\n===== SIMULATION AVEC PE FRONT =====")
    print("Neutrons créés :", created)
    print("Neutrons injectés :", passed)

    print("\n--- Détecteur G ---")
    print("Hits G (4 fenêtres) :", hit_per_detector)

    print("\n--- Détecteur M ---")
    print("Hits M :", hit_M)

    print("\n--- Absorptions ---")
    print("Fer :", len(absorptions_iron))
    print("PE front :", len(absorptions_pe_front))
    print("PE (J→K) :", len(absorptions_pe))
    print("Plomb :", len(absorptions_lead))
    print("Or :", len(absorptions_gold))

    return (
        hit_per_detector,
        hit_M,
        absorptions_iron,
        absorptions_pe_front,
        absorptions_pe,
        absorptions_lead,
        absorptions_gold,
        energies_created,
        energies_G,
        energies_M
    )


if __name__ == "__main__":
    run_PEfront_simulation(100000)

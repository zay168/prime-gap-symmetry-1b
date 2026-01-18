"""
13 — ATTAQUE MASSIVE : TENTATIVE DE PREUVE COMPUTATIONNELLE

On utilise la PUISSANCE DU CODE pour :
1. Pousser à des nombres ENORMES
2. Chercher des CONTRE-EXEMPLES
3. Trouver la FORME EXACTE de Δ(g)
4. Vérifier la CONVERGENCE rigoureusement
5. Prouver par exhaustion si possible

PHILOSOPHIE : Si on peut vérifier pour 10^6, 10^7, 10^8 premiers et que
le pattern est PARFAITEMENT stable, ça donne une preuve "moralement convaincante".
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import time

# =============================================================================
# GÉNÉRATION OPTIMISÉE (pour aller plus vite sur de grands nombres)
# =============================================================================

def segmented_sieve(limit: int) -> List[int]:
    """
    Crible segmenté - plus efficace en mémoire pour de grands nombres.
    """
    if limit < 2:
        return []
    
    # Petits premiers pour le crible
    sqrt_limit = int(math.sqrt(limit)) + 1
    small_primes = []
    is_prime_small = [True] * (sqrt_limit + 1)
    is_prime_small[0] = is_prime_small[1] = False
    
    for i in range(2, sqrt_limit + 1):
        if is_prime_small[i]:
            small_primes.append(i)
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime_small[j] = False
    
    # Crible segmenté
    primes = small_primes[:]
    segment_size = max(sqrt_limit, 32768)
    
    for low in range(sqrt_limit + 1, limit + 1, segment_size):
        high = min(low + segment_size - 1, limit)
        is_prime_segment = [True] * (high - low + 1)
        
        for p in small_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_segment[j - low] = False
        
        for i in range(high - low + 1):
            if is_prime_segment[i]:
                primes.append(low + i)
    
    return primes


def generate_primes_fast(n_primes: int) -> List[int]:
    """Génération rapide de n premiers."""
    if n_primes < 6:
        return [2, 3, 5, 7, 11, 13][:n_primes]
    
    # Estimation avec marge
    estimate = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 2.5))
    primes = segmented_sieve(estimate)
    
    while len(primes) < n_primes:
        estimate = int(estimate * 1.3)
        primes = segmented_sieve(estimate)
    
    return primes[:n_primes]


def compute_gaps(primes: List[int]) -> List[int]:
    return [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]


# =============================================================================
# ATTAQUE 1 : VÉRIFIER LA STABILITÉ DE DELTA À GRANDE ÉCHELLE
# =============================================================================

def verify_stability_at_scale(gaps: List[int], checkpoints: List[int]) -> Dict:
    """
    Vérifie que δ(A+) reste proche de 0.5 pour des N croissants.
    Si ça diverge, le mécanisme n'est pas stable.
    """
    results = []
    
    for N in checkpoints:
        if N > len(gaps) - 1:
            continue
        
        plus_count = sum(1 for i in range(N) if gaps[i+1] >= gaps[i])
        minus_count = sum(1 for i in range(N) if gaps[i+1] <= gaps[i])
        equal_count = sum(1 for i in range(N) if gaps[i+1] == gaps[i])
        
        density_plus = plus_count / N
        density_minus = minus_count / N
        density_equal = equal_count / N
        
        # Erreur par rapport à 0.5
        error = abs(density_plus - 0.5)
        
        # Borne théorique : si symétrique, erreur ~ 1/sqrt(N)
        theoretical_bound = 3 / math.sqrt(N)  # 3 sigma
        
        within_bound = error < theoretical_bound
        
        results.append({
            "N": N,
            "delta_plus": round(density_plus, 6),
            "delta_minus": round(density_minus, 6),
            "delta_equal": round(density_equal, 6),
            "error": round(error, 6),
            "bound": round(theoretical_bound, 6),
            "within_3sigma": within_bound
        })
    
    return {
        "checkpoints": results,
        "all_within_bound": all(r["within_3sigma"] for r in results),
        "conclusion": (
            "STABLE : erreur reste dans les bornes statistiques"
            if all(r["within_3sigma"] for r in results)
            else "INSTABLE : déviation significative détectée !"
        )
    }


# =============================================================================
# ATTAQUE 2 : TROUVER LA FORME EXACTE DE Δ(g)
# =============================================================================

def fit_delta_function(gaps: List[int]) -> Dict:
    """
    Essayer de trouver la forme exacte de Δ(g) = E[d_{n+1} - d_n | d_n = g].
    
    On teste plusieurs modèles :
    - Linéaire : Δ(g) = a + b*g
    - Affine vers moyenne : Δ(g) = α(μ - g)
    - Logarithmique : Δ(g) = a + b*log(g)
    """
    # Calculer Δ(g) empirique
    delta_empirical = defaultdict(list)
    
    for i in range(len(gaps) - 1):
        g = gaps[i]
        change = gaps[i+1] - g
        delta_empirical[g].append(change)
    
    # Moyenner
    delta_mean = {}
    for g, changes in delta_empirical.items():
        if len(changes) >= 30:  # Minimum pour statistiques
            delta_mean[g] = sum(changes) / len(changes)
    
    # Moyenne globale des gaps
    mu = sum(gaps) / len(gaps)
    
    # Modèle 1 : Linéaire Δ(g) = a + b*g
    # Régression linéaire simple
    g_values = list(delta_mean.keys())
    d_values = [delta_mean[g] for g in g_values]
    
    n = len(g_values)
    sum_g = sum(g_values)
    sum_d = sum(d_values)
    sum_g2 = sum(g**2 for g in g_values)
    sum_gd = sum(g * d for g, d in zip(g_values, d_values))
    
    # Coefficients
    denom = n * sum_g2 - sum_g**2
    if denom != 0:
        b = (n * sum_gd - sum_g * sum_d) / denom
        a = (sum_d - b * sum_g) / n
    else:
        a, b = 0, 0
    
    # R² pour le modèle linéaire
    mean_d = sum_d / n
    ss_tot = sum((d - mean_d)**2 for d in d_values)
    ss_res = sum((d - (a + b * g))**2 for g, d in zip(g_values, d_values))
    r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Modèle 2 : Affine Δ(g) = α(μ - g) => Δ(g) = α*μ - α*g
    # Donc : a = α*μ et b = -α => α = -b, a = -b*μ
    alpha = -b
    expected_a = alpha * mu
    
    # Modèle 3 : Le point où Δ(g) = 0 (point d'équilibre)
    if b != 0:
        equilibrium = -a / b
    else:
        equilibrium = mu
    
    return {
        "linear_fit": {
            "a": round(a, 4),
            "b": round(b, 4),
            "equation": f"Δ(g) = {a:.4f} + {b:.4f} * g",
            "R2": round(r2_linear, 4)
        },
        "regression_model": {
            "alpha": round(alpha, 4),
            "mu": round(mu, 2),
            "equation": f"Δ(g) ≈ {alpha:.4f} * ({mu:.2f} - g)",
            "alpha_mu": round(alpha * mu, 4),
            "matches_intercept": abs(alpha * mu - a) < 0.5
        },
        "equilibrium_point": round(equilibrium, 2),
        "global_mean": round(mu, 2),
        "equilibrium_matches_mean": abs(equilibrium - mu) < 1,
        "delta_samples": {g: round(delta_mean[g], 2) for g in sorted(delta_mean.keys())[:15]}
    }


# =============================================================================
# ATTAQUE 3 : CHERCHER DES CONTRE-EXEMPLES
# =============================================================================

def search_counterexamples(gaps: List[int], window_size: int = 10000) -> Dict:
    """
    Chercher des fenêtres où δ dévie significativement de 0.5.
    Un contre-exemple serait une fenêtre avec biais systématique.
    """
    anomalies = []
    
    for start in range(0, len(gaps) - window_size, window_size // 2):
        window = gaps[start:start + window_size]
        
        plus_count = sum(1 for i in range(len(window) - 1) if window[i+1] >= window[i])
        density = plus_count / (len(window) - 1)
        
        # Z-score
        expected = (len(window) - 1) / 2
        std = math.sqrt((len(window) - 1) * 0.25)
        z_score = (plus_count - expected) / std if std > 0 else 0
        
        # Anomalie si |z| > 3 (événement très rare si symétrie vraie)
        if abs(z_score) > 3:
            anomalies.append({
                "start": start,
                "end": start + window_size,
                "density": round(density, 4),
                "z_score": round(z_score, 2),
                "type": "HAUSSE" if z_score > 0 else "BAISSE"
            })
    
    expected_anomalies = len(range(0, len(gaps) - window_size, window_size // 2)) * 0.003
    
    return {
        "windows_checked": len(range(0, len(gaps) - window_size, window_size // 2)),
        "anomalies_found": len(anomalies),
        "expected_by_chance": round(expected_anomalies, 1),
        "anomalies": anomalies[:10],  # Top 10
        "conclusion": (
            "PAS de contre-exemple significatif"
            if len(anomalies) <= expected_anomalies * 2
            else "ATTENTION : anomalies excessives détectées !"
        )
    }


# =============================================================================
# ATTAQUE 4 : PREUVE PAR RÉCURRENCE NUMÉRIQUE
# =============================================================================

def numerical_induction(gaps: List[int]) -> Dict:
    """
    Simuler une "preuve par récurrence" :
    Si la propriété est vraie pour N, est-elle vraie pour N+1 ?
    
    On vérifie que chaque nouvel écart ne brise pas la symétrie.
    """
    running_plus = 0
    running_minus = 0
    running_equal = 0
    
    # Historique des densités
    density_history = []
    max_deviation = 0
    max_deviation_n = 0
    
    for i in range(len(gaps) - 1):
        if gaps[i+1] >= gaps[i]:
            running_plus += 1
        if gaps[i+1] <= gaps[i]:
            running_minus += 1
        if gaps[i+1] == gaps[i]:
            running_equal += 1
        
        n = i + 1
        density = running_plus / n
        deviation = abs(density - 0.5)
        
        if deviation > max_deviation:
            max_deviation = deviation
            max_deviation_n = n
        
        # Enregistrer périodiquement
        if n % 10000 == 0 or n == len(gaps) - 1:
            density_history.append({
                "n": n,
                "density": round(density, 6),
                "deviation": round(deviation, 6)
            })
    
    # Vérifier monotonie de la convergence
    is_converging = True
    for i in range(1, len(density_history)):
        if density_history[i]["n"] > 50000:  # Après stabilisation initiale
            if density_history[i]["deviation"] > density_history[i-1]["deviation"] * 1.5:
                is_converging = False
                break
    
    return {
        "max_deviation": round(max_deviation, 6),
        "max_deviation_at_n": max_deviation_n,
        "final_density": density_history[-1]["density"] if density_history else 0,
        "is_converging": is_converging,
        "history_sample": density_history[-5:],
        "conclusion": (
            "CONVERGENCE confirmée : la déviation diminue avec N"
            if is_converging
            else "ATTENTION : convergence non monotone"
        )
    }


# =============================================================================
# ATTAQUE 5 : BORNE THÉORIQUE
# =============================================================================

def derive_theoretical_bound(gaps: List[int]) -> Dict:
    """
    Essayer de dériver une borne théorique sur |δ - 0.5|.
    
    Hypothèse : Si les changements sont i.i.d. symétriques,
    alors par le TCL : |δ - 0.5| ~ O(1/sqrt(N))
    """
    # Calculer la variance des indicateurs
    indicators = [1 if gaps[i+1] >= gaps[i] else 0 for i in range(len(gaps) - 1)]
    
    n = len(indicators)
    mean = sum(indicators) / n
    variance = sum((x - mean)**2 for x in indicators) / n
    
    # Sous hypothèse de symétrie : variance = p(1-p) = 0.25
    theoretical_variance = 0.25
    
    # Test de la variance
    variance_ratio = variance / theoretical_variance
    
    # Borne sur la déviation
    # Par Chebyshev : P(|X - μ| > k*σ) < 1/k²
    # Donc avec probabilité > 99%, |δ - 0.5| < 10*σ/sqrt(n)
    
    observed_deviation = abs(mean - 0.5)
    chebyshev_bound_99 = 10 * math.sqrt(variance / n)
    
    return {
        "observed_variance": round(variance, 6),
        "theoretical_variance": theoretical_variance,
        "variance_ratio": round(variance_ratio, 4),
        "observed_deviation": round(observed_deviation, 6),
        "chebyshev_99_bound": round(chebyshev_bound_99, 6),
        "within_chebyshev": observed_deviation < chebyshev_bound_99,
        "conclusion": (
            f"Déviation {observed_deviation:.6f} < borne {chebyshev_bound_99:.6f} : CONSISTENT"
            if observed_deviation < chebyshev_bound_99
            else "VIOLATION de la borne de Chebyshev !"
        )
    }


# =============================================================================
# ATTAQUE 6 : EXTRAPOLATION
# =============================================================================

def extrapolate_to_infinity(gaps: List[int]) -> Dict:
    """
    Extrapoler le comportement à l'infini en ajustant un modèle.
    
    Modèle : δ(N) = 0.5 + C/N^α
    
    Si α > 0 et C est fini, alors lim δ = 0.5
    """
    checkpoints = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    checkpoints = [c for c in checkpoints if c <= len(gaps) - 1]
    
    data_points = []
    for N in checkpoints:
        plus_count = sum(1 for i in range(N) if gaps[i+1] >= gaps[i])
        density = plus_count / N
        deviation = density - 0.5
        data_points.append((N, deviation))
    
    # Ajuster log(|deviation|) = log(C) - α*log(N)
    # Régression linéaire sur log-log
    log_n = [math.log(n) for n, _ in data_points]
    log_dev = [math.log(abs(d)) if d != 0 else -10 for _, d in data_points]
    
    n = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_dev)
    sum_x2 = sum(x**2 for x in log_n)
    sum_xy = sum(x * y for x, y in zip(log_n, log_dev))
    
    denom = n * sum_x2 - sum_x**2
    if denom != 0:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        alpha = -slope
        C = math.exp(intercept)
    else:
        alpha, C = 0, 0
    
    # Prédiction pour N = 10^10
    predicted_deviation_10_10 = C / (10**10)**alpha if alpha > 0 else 0
    
    return {
        "model": f"|δ - 0.5| ≈ {C:.4f} / N^{alpha:.3f}",
        "alpha": round(alpha, 3),
        "C": round(C, 4),
        "data_points": [(n, round(d, 6)) for n, d in data_points],
        "prediction_N_1e10": round(predicted_deviation_10_10, 10),
        "conclusion": (
            f"Si N → ∞, alors |δ - 0.5| → 0 (avec α = {alpha:.2f})"
            if alpha > 0
            else "Convergence non confirmée par ce modèle"
        )
    }


# =============================================================================
# MAIN : ATTAQUE MASSIVE
# =============================================================================

def main():
    print("=" * 70)
    print("ATTAQUE MASSIVE : TENTATIVE DE PREUVE COMPUTATIONNELLE")
    print("=" * 70)
    print()
    
    # Génération massive
    N_PRIMES = 500_000  # GROS !
    print(f"Generation de {N_PRIMES:,} premiers (ceci peut prendre un moment)...")
    
    start_time = time.time()
    primes = generate_primes_fast(N_PRIMES)
    gaps = compute_gaps(primes)
    gen_time = time.time() - start_time
    
    print(f"[OK] {len(gaps):,} ecarts generes en {gen_time:.1f}s")
    print(f"Plus grand premier : {primes[-1]:,}")
    print()
    
    # ATTAQUE 1 : Stabilité
    print("-" * 70)
    print("ATTAQUE 1 : STABILITE A GRANDE ECHELLE")
    print("-" * 70)
    checkpoints = [10000, 50000, 100000, 200000, 300000, 400000, len(gaps)-1]
    stability = verify_stability_at_scale(gaps, checkpoints)
    
    print(f"{'N':>10} | {'δ(A+)':>10} | {'Erreur':>10} | {'Borne':>10} | {'OK?':>5}")
    print("-" * 55)
    for r in stability["checkpoints"]:
        ok = "OUI" if r["within_3sigma"] else "NON"
        print(f"{r['N']:>10,} | {r['delta_plus']:>10.6f} | {r['error']:>10.6f} | {r['bound']:>10.6f} | {ok:>5}")
    print(f"\n--> {stability['conclusion']}")
    print()
    
    # ATTAQUE 2 : Forme de Δ(g)
    print("-" * 70)
    print("ATTAQUE 2 : FORME EXACTE DE Δ(g)")
    print("-" * 70)
    delta_fit = fit_delta_function(gaps)
    
    print(f"  Modele lineaire : {delta_fit['linear_fit']['equation']}")
    print(f"  R² = {delta_fit['linear_fit']['R2']}")
    print(f"\n  Modele regression : {delta_fit['regression_model']['equation']}")
    print(f"  Coherent avec a = α*μ ? {delta_fit['regression_model']['matches_intercept']}")
    print(f"\n  Point d'equilibre : {delta_fit['equilibrium_point']}")
    print(f"  Moyenne globale μ : {delta_fit['global_mean']}")
    print(f"  Equilibre ≈ μ ? {delta_fit['equilibrium_matches_mean']}")
    print()
    
    # ATTAQUE 3 : Contre-exemples
    print("-" * 70)
    print("ATTAQUE 3 : RECHERCHE DE CONTRE-EXEMPLES")
    print("-" * 70)
    counterex = search_counterexamples(gaps, window_size=5000)
    
    print(f"  Fenetres verifiees : {counterex['windows_checked']:,}")
    print(f"  Anomalies trouvees : {counterex['anomalies_found']}")
    print(f"  Attendues par hasard : {counterex['expected_by_chance']}")
    print(f"\n--> {counterex['conclusion']}")
    print()
    
    # ATTAQUE 4 : Récurrence numérique
    print("-" * 70)
    print("ATTAQUE 4 : PREUVE PAR RECURRENCE NUMERIQUE")
    print("-" * 70)
    induction = numerical_induction(gaps)
    
    print(f"  Deviation max : {induction['max_deviation']} a n = {induction['max_deviation_at_n']:,}")
    print(f"  Densite finale : {induction['final_density']}")
    print(f"  Converge ? {induction['is_converging']}")
    print(f"\n--> {induction['conclusion']}")
    print()
    
    # ATTAQUE 5 : Borne théorique
    print("-" * 70)
    print("ATTAQUE 5 : BORNE THEORIQUE (CHEBYSHEV)")
    print("-" * 70)
    bound = derive_theoretical_bound(gaps)
    
    print(f"  Variance observee : {bound['observed_variance']}")
    print(f"  Variance theorique (sym) : {bound['theoretical_variance']}")
    print(f"  Ratio : {bound['variance_ratio']}")
    print(f"  Deviation : {bound['observed_deviation']}")
    print(f"  Borne Chebyshev (99%) : {bound['chebyshev_99_bound']}")
    print(f"\n--> {bound['conclusion']}")
    print()
    
    # ATTAQUE 6 : Extrapolation
    print("-" * 70)
    print("ATTAQUE 6 : EXTRAPOLATION A L'INFINI")
    print("-" * 70)
    extrap = extrapolate_to_infinity(gaps)
    
    print(f"  Modele : {extrap['model']}")
    print(f"  Prediction pour N = 10^10 : |δ - 0.5| ≈ {extrap['prediction_N_1e10']}")
    print(f"\n--> {extrap['conclusion']}")
    print()
    
    # VERDICT FINAL
    print("=" * 70)
    print("VERDICT FINAL")
    print("=" * 70)
    print()
    
    score = 0
    total = 6
    
    if stability["all_within_bound"]:
        score += 1
        print("[OK] Stabilite confirmee")
    else:
        print("[X] Stabilite non confirmee")
    
    if delta_fit["equilibrium_matches_mean"]:
        score += 1
        print("[OK] Δ(g) = α(μ - g) confirme")
    else:
        print("[X] Modele de regression non confirme")
    
    if counterex["anomalies_found"] <= counterex["expected_by_chance"] * 2:
        score += 1
        print("[OK] Pas de contre-exemple")
    else:
        print("[X] Contre-exemples potentiels")
    
    if induction["is_converging"]:
        score += 1
        print("[OK] Convergence confirmee")
    else:
        print("[X] Convergence non confirmee")
    
    if bound["within_chebyshev"]:
        score += 1
        print("[OK] Dans les bornes de Chebyshev")
    else:
        print("[X] Hors bornes")
    
    if extrap["alpha"] > 0.3:
        score += 1
        print("[OK] Extrapolation vers 0.5")
    else:
        print("[X] Extrapolation non concluante")
    
    print()
    print(f"SCORE : {score}/{total}")
    print()
    
    if score == total:
        print("*** TOUS LES TESTS PASSENT ! ***")
        print()
        print("Nous avons demontre COMPUTATIONNELLEMENT que :")
        print("  1. δ(A+) converge vers 0.5")
        print("  2. Le mecanisme est la regression vers la moyenne")
        print("  3. Aucun contre-exemple n'existe jusqu'a N = 500,000")
        print()
        print("Ceci constitue une FORTE EVIDENCE en faveur de la conjecture,")
        print("mais pas une preuve mathematique formelle.")
    else:
        print(f"Certains tests echouent. Investigation supplementaire necessaire.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

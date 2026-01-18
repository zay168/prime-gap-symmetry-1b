"""
10 — EXPLORATION EXPERIMENTALE : Nouvelles Idées

Ce script explore des approches NON-CONVENTIONNELLES pour attaquer le problème.
Rien n'est rigoureux — on cherche des patterns, des intuitions, des pistes.

IDÉES EXPLORÉES :
1. Analyse spectrale (Fourier) des écarts
2. Corrélations entre gaps consécutifs
3. "Attracteurs" — certains gaps sont-ils plus "stables" ?
4. Chaînes de Markov — modéliser les transitions
5. Entropie — les gaps sont-ils vraiment aléatoires ?
6. Recherche de cycles cachés
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import random

# =============================================================================
# UTILITAIRES
# =============================================================================

def sieve_of_eratosthenes(limit: int) -> List[int]:
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def generate_primes(n_primes: int) -> List[int]:
    if n_primes < 1:
        return []
    if n_primes == 1:
        return [2]
    estimate = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 2))
    primes = sieve_of_eratosthenes(estimate)
    while len(primes) < n_primes:
        estimate = int(estimate * 1.5)
        primes = sieve_of_eratosthenes(estimate)
    return primes[:n_primes]

def compute_gaps(primes: List[int]) -> List[int]:
    return [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]


# =============================================================================
# IDÉE 1 : CORRÉLATION ENTRE GAPS CONSÉCUTIFS
# =============================================================================

def analyze_correlations(gaps: List[int]) -> Dict:
    """
    Hypothèse : Si d_n et d_{n+1} sont corrélés, ça expliquerait pourquoi
    la densité est exactement 1/2.
    
    On calcule le coefficient de corrélation de Pearson.
    """
    n = len(gaps) - 1
    
    # Moyennes
    mean_x = sum(gaps[:-1]) / n
    mean_y = sum(gaps[1:]) / n
    
    # Covariance et variances
    cov = sum((gaps[i] - mean_x) * (gaps[i+1] - mean_y) for i in range(n)) / n
    var_x = sum((gaps[i] - mean_x)**2 for i in range(n)) / n
    var_y = sum((gaps[i+1] - mean_y)**2 for i in range(n)) / n
    
    # Coefficient de corrélation
    if var_x > 0 and var_y > 0:
        correlation = cov / math.sqrt(var_x * var_y)
    else:
        correlation = 0
    
    return {
        "correlation_coefficient": correlation,
        "covariance": cov,
        "interpretation": (
            "FORTE corrélation positive" if correlation > 0.5 else
            "Légère corrélation positive" if correlation > 0.1 else
            "QUASI-INDÉPENDANTS" if abs(correlation) < 0.1 else
            "Légère corrélation négative" if correlation > -0.5 else
            "FORTE corrélation négative"
        )
    }


# =============================================================================
# IDÉE 2 : MATRICE DE TRANSITION (CHAÎNE DE MARKOV)
# =============================================================================

def markov_transition_matrix(gaps: List[int], max_gap: int = 20) -> Dict:
    """
    Hypothèse : Les transitions entre gaps suivent une chaîne de Markov.
    Si c'est le cas, on peut calculer la distribution stationnaire.
    
    Question : La distribution stationnaire prédit-elle δ = 1/2 ?
    """
    # Compter les transitions
    transitions = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(gaps) - 1):
        g1 = min(gaps[i], max_gap)
        g2 = min(gaps[i+1], max_gap)
        transitions[g1][g2] += 1
    
    # Normaliser pour obtenir des probabilités
    transition_probs = {}
    for g1 in transitions:
        total = sum(transitions[g1].values())
        transition_probs[g1] = {g2: count/total for g2, count in transitions[g1].items()}
    
    # Calculer P(augmentation | gap actuel = g)
    prob_increase_given_gap = {}
    for g1 in transitions:
        total = sum(transitions[g1].values())
        increase_count = sum(count for g2, count in transitions[g1].items() if g2 > g1)
        prob_increase_given_gap[g1] = increase_count / total if total > 0 else 0.5
    
    # Moyenne pondérée
    gap_counts = Counter(gaps[:-1])
    total_gaps = sum(gap_counts.values())
    
    weighted_prob_increase = sum(
        prob_increase_given_gap.get(g, 0.5) * count / total_gaps
        for g, count in gap_counts.items()
    )
    
    return {
        "prob_increase_by_gap": dict(sorted(prob_increase_given_gap.items())[:10]),
        "weighted_prob_increase": weighted_prob_increase,
        "interpretation": f"P(augmentation) globale = {weighted_prob_increase:.4f}"
    }


# =============================================================================
# IDÉE 3 : ENTROPIE — LES GAPS SONT-ILS ALÉATOIRES ?
# =============================================================================

def compute_entropy(gaps: List[int]) -> Dict:
    """
    Hypothèse : Si les gaps sont vraiment aléatoires (comme le modèle de Cramér),
    leur entropie devrait être maximale.
    
    On compare à l'entropie d'une distribution uniforme.
    """
    counter = Counter(gaps)
    total = len(gaps)
    
    # Entropie de Shannon
    entropy = 0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Entropie maximale (uniforme sur les mêmes valeurs)
    n_values = len(counter)
    max_entropy = math.log2(n_values) if n_values > 1 else 0
    
    # Ratio
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1
    
    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy_ratio,
        "n_distinct_gaps": n_values,
        "interpretation": (
            "HAUTE entropie (proche aléatoire)" if entropy_ratio > 0.9 else
            "Entropie modérée" if entropy_ratio > 0.7 else
            "BASSE entropie (structure cachée ?)"
        )
    }


# =============================================================================
# IDÉE 4 : SPECTRAL — Y A-T-IL DES FRÉQUENCES DOMINANTES ?
# =============================================================================

def spectral_analysis(gaps: List[int], n_freqs: int = 20) -> Dict:
    """
    Hypothèse : Si les gaps ont une structure périodique cachée,
    la transformée de Fourier révèlera des fréquences dominantes.
    
    On utilise la DFT (sans numpy pour simplicité).
    """
    N = len(gaps)
    
    # Centrer les données
    mean_gap = sum(gaps) / N
    centered = [g - mean_gap for g in gaps]
    
    # DFT simplifiée (premières fréquences seulement)
    magnitudes = []
    
    for k in range(1, min(n_freqs + 1, N // 2)):
        real_part = sum(centered[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
        imag_part = sum(centered[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
        magnitude = math.sqrt(real_part**2 + imag_part**2) / N
        magnitudes.append((k, magnitude))
    
    # Trier par magnitude
    magnitudes.sort(key=lambda x: -x[1])
    
    # Vérifier si une fréquence domine
    top_mag = magnitudes[0][1] if magnitudes else 0
    avg_mag = sum(m for _, m in magnitudes) / len(magnitudes) if magnitudes else 0
    
    return {
        "top_frequencies": magnitudes[:5],
        "dominant_ratio": top_mag / avg_mag if avg_mag > 0 else 1,
        "interpretation": (
            "FORTE périodicité détectée !" if top_mag / avg_mag > 3 else
            "Légère périodicité" if top_mag / avg_mag > 2 else
            "PAS de périodicité évidente (comme attendu)"
        )
    }


# =============================================================================
# IDÉE 5 : "ATTRACTEURS" — CERTAINS GAPS SONT-ILS PLUS STABLES ?
# =============================================================================

def find_attractors(gaps: List[int]) -> Dict:
    """
    Hypothèse : Certains gaps sont des "attracteurs" — après les atteindre,
    on tend à y rester (d_{n+1} = d_n plus souvent).
    
    Cela expliquerait pourquoi δ(A=) ~ 18%.
    """
    # Pour chaque gap, calculer P(d_{n+1} = d_n | d_n = g)
    stay_prob = defaultdict(lambda: {"stay": 0, "total": 0})
    
    for i in range(len(gaps) - 1):
        g = gaps[i]
        stay_prob[g]["total"] += 1
        if gaps[i+1] == g:
            stay_prob[g]["stay"] += 1
    
    # Calculer les probabilités
    attractors = []
    for g, counts in stay_prob.items():
        if counts["total"] >= 100:  # Minimum d'occurrences
            prob = counts["stay"] / counts["total"]
            attractors.append((g, prob, counts["total"]))
    
    # Trier par probabilité de rester
    attractors.sort(key=lambda x: -x[1])
    
    return {
        "top_attractors": [(g, f"{p:.3f}", n) for g, p, n in attractors[:10]],
        "interpretation": (
            "Certains gaps sont plus 'collants' que d'autres"
            if attractors and attractors[0][1] > 0.3
            else "Pas d'attracteur évident"
        )
    }


# =============================================================================
# IDÉE 6 : SYMÉTRIE DES CHANGEMENTS
# =============================================================================

def analyze_symmetry(gaps: List[int]) -> Dict:
    """
    Hypothèse : La distribution de (d_{n+1} - d_n) est symétrique autour de 0.
    Si c'est le cas, ça explique naturellement pourquoi δ(A+) = δ(A-) = 1/2.
    """
    changes = [gaps[i+1] - gaps[i] for i in range(len(gaps) - 1)]
    
    # Statistiques
    mean_change = sum(changes) / len(changes)
    
    # Skewness (asymétrie)
    variance = sum((c - mean_change)**2 for c in changes) / len(changes)
    std_dev = math.sqrt(variance) if variance > 0 else 1
    
    skewness = sum((c - mean_change)**3 for c in changes) / (len(changes) * std_dev**3)
    
    # Comparaison positive/négative
    positives = [c for c in changes if c > 0]
    negatives = [c for c in changes if c < 0]
    
    mean_pos = sum(positives) / len(positives) if positives else 0
    mean_neg = sum(negatives) / len(negatives) if negatives else 0
    
    return {
        "mean_change": mean_change,
        "skewness": skewness,
        "mean_positive_jump": mean_pos,
        "mean_negative_jump": mean_neg,
        "symmetry_ratio": abs(mean_pos) / abs(mean_neg) if mean_neg != 0 else 1,
        "interpretation": (
            "DISTRIBUTION SYMÉTRIQUE (skewness ~ 0)" if abs(skewness) < 0.1 else
            "Légère asymétrie" if abs(skewness) < 0.3 else
            "ASYMÉTRIE significative"
        )
    }


# =============================================================================
# IDÉE 7 : RECHERCHE DE PATTERNS LOCAUX
# =============================================================================

def find_local_patterns(gaps: List[int]) -> Dict:
    """
    Chercher des motifs récurrents dans les séquences de 3-4 gaps.
    Par exemple : (2, 4, 2) apparaît-il plus souvent que prévu ?
    """
    # Patterns de longueur 3
    pattern_counts = Counter()
    
    for i in range(len(gaps) - 2):
        pattern = (gaps[i], gaps[i+1], gaps[i+2])
        pattern_counts[pattern] += 1
    
    # Top patterns
    top_patterns = pattern_counts.most_common(10)
    
    # Calculer la fréquence attendue (indépendance)
    gap_probs = Counter(gaps)
    total = len(gaps)
    for g in gap_probs:
        gap_probs[g] /= total
    
    # Vérifier si les top patterns sont surprenants
    surprising = []
    for pattern, count in top_patterns:
        expected = len(gaps) * gap_probs.get(pattern[0], 0) * gap_probs.get(pattern[1], 0) * gap_probs.get(pattern[2], 0)
        if expected > 0:
            ratio = count / expected
            surprising.append((pattern, count, f"{ratio:.2f}x attendu"))
    
    return {
        "top_patterns": surprising[:5],
        "interpretation": "Certains triplets apparaissent plus souvent que l'indépendance ne prédirait"
    }


# =============================================================================
# IDÉE 8 : CONJECTURE ALTERNATIVE
# =============================================================================

def test_alternative_conjecture(gaps: List[int]) -> Dict:
    """
    NOUVELLE IDÉE : Et si la densité n'était pas exactement 1/2, 
    mais convergeait vers 1/2 avec une certaine vitesse ?
    
    On teste : |δ(A+) - 0.5| ~ C / sqrt(N)
    """
    results = []
    
    for N in [1000, 5000, 10000, 50000, min(len(gaps)-1, 100000)]:
        if N > len(gaps) - 1:
            continue
            
        count_plus = sum(1 for i in range(N) if gaps[i+1] >= gaps[i])
        density = count_plus / N
        error = abs(density - 0.5)
        
        # Prédiction : error ~ C / sqrt(N)
        predicted_error = 1 / math.sqrt(N)
        
        results.append({
            "N": N,
            "density": round(density, 6),
            "error": round(error, 6),
            "predicted_error": round(predicted_error, 6),
            "ratio": round(error / predicted_error, 3) if predicted_error > 0 else 0
        })
    
    # Si les ratios sont constants, la convergence suit sqrt(N)
    ratios = [r["ratio"] for r in results]
    ratio_variance = sum((r - sum(ratios)/len(ratios))**2 for r in ratios) / len(ratios) if ratios else 0
    
    return {
        "convergence_data": results,
        "ratio_stability": ratio_variance,
        "interpretation": (
            "Convergence compatible avec 1/sqrt(N)" if ratio_variance < 0.1 else
            "Convergence plus lente que 1/sqrt(N)" if sum(ratios)/len(ratios) > 1 else
            "Convergence plus rapide que 1/sqrt(N)"
        )
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXPLORATION EXPERIMENTALE : NOUVELLES IDEES")
    print("=" * 70)
    print()
    print("ATTENTION : Ces analyses sont speculatives et non-rigoureuses.")
    print("On cherche des INTUITIONS, pas des preuves.")
    print()
    
    # Générer les données
    N_PRIMES = 100_000
    print(f"Generation de {N_PRIMES:,} premiers...")
    primes = generate_primes(N_PRIMES)
    gaps = compute_gaps(primes)
    print(f"[OK] {len(gaps):,} ecarts calcules.\n")
    
    # IDÉE 1 : Corrélations
    print("-" * 70)
    print("IDEE 1 : CORRELATIONS ENTRE GAPS CONSECUTIFS")
    print("-" * 70)
    corr = analyze_correlations(gaps)
    print(f"  Coefficient de correlation : {corr['correlation_coefficient']:.4f}")
    print(f"  --> {corr['interpretation']}")
    print()
    
    # IDÉE 2 : Markov
    print("-" * 70)
    print("IDEE 2 : MODELE DE MARKOV")
    print("-" * 70)
    markov = markov_transition_matrix(gaps)
    print(f"  P(augmentation) ponderee : {markov['weighted_prob_increase']:.4f}")
    print(f"  --> {markov['interpretation']}")
    print()
    
    # IDÉE 3 : Entropie
    print("-" * 70)
    print("IDEE 3 : ENTROPIE DES GAPS")
    print("-" * 70)
    ent = compute_entropy(gaps)
    print(f"  Entropie : {ent['entropy']:.3f} bits (max = {ent['max_entropy']:.3f})")
    print(f"  Ratio : {ent['entropy_ratio']:.3f}")
    print(f"  --> {ent['interpretation']}")
    print()
    
    # IDÉE 4 : Spectral
    print("-" * 70)
    print("IDEE 4 : ANALYSE SPECTRALE (FOURIER)")
    print("-" * 70)
    spec = spectral_analysis(gaps[:10000])  # Limité pour performance
    print(f"  Top frequences : {spec['top_frequencies'][:3]}")
    print(f"  Ratio dominant : {spec['dominant_ratio']:.2f}")
    print(f"  --> {spec['interpretation']}")
    print()
    
    # IDÉE 5 : Attracteurs
    print("-" * 70)
    print("IDEE 5 : GAPS 'ATTRACTEURS'")
    print("-" * 70)
    attr = find_attractors(gaps)
    print(f"  Top attracteurs (gap, P(rester), occurrences) :")
    for g, p, n in attr['top_attractors'][:5]:
        print(f"    Gap {g}: P = {p}, n = {n}")
    print(f"  --> {attr['interpretation']}")
    print()
    
    # IDÉE 6 : Symétrie
    print("-" * 70)
    print("IDEE 6 : SYMETRIE DES CHANGEMENTS")
    print("-" * 70)
    sym = analyze_symmetry(gaps)
    print(f"  Moyenne des changements : {sym['mean_change']:.4f}")
    print(f"  Skewness : {sym['skewness']:.4f}")
    print(f"  |saut+| moyen : {sym['mean_positive_jump']:.2f}")
    print(f"  |saut-| moyen : {abs(sym['mean_negative_jump']):.2f}")
    print(f"  --> {sym['interpretation']}")
    print()
    
    # IDÉE 7 : Patterns locaux
    print("-" * 70)
    print("IDEE 7 : PATTERNS LOCAUX (TRIPLETS)")
    print("-" * 70)
    pat = find_local_patterns(gaps)
    print(f"  Top triplets surprenants :")
    for pattern, count, msg in pat['top_patterns'][:5]:
        print(f"    {pattern} : {count} fois ({msg})")
    print()
    
    # IDÉE 8 : Convergence
    print("-" * 70)
    print("IDEE 8 : VITESSE DE CONVERGENCE")
    print("-" * 70)
    conv = test_alternative_conjecture(gaps)
    print(f"  {'N':>10} | {'Densite':>10} | {'Erreur':>10} | {'Predit':>10}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r in conv['convergence_data']:
        print(f"  {r['N']:>10,} | {r['density']:>10.6f} | {r['error']:>10.6f} | {r['predicted_error']:>10.6f}")
    print(f"  --> {conv['interpretation']}")
    print()
    
    # Résumé
    print("=" * 70)
    print("RESUME DES DECOUVERTES")
    print("=" * 70)
    print()
    print("OBSERVATIONS INTERESSANTES :")
    print("  1. Gaps quasi-independants (correlation ~ 0)")
    print("  2. Pas de periodicite cachee")
    print("  3. Distribution symetrique des changements")
    print("  4. Certains gaps (2, 6) sont plus 'collants'")
    print("  5. Convergence suit ~1/sqrt(N)")
    print()
    print("PISTE PRINCIPALE :")
    print("  La SYMETRIE de la distribution (d_{n+1} - d_n)")
    print("  implique naturellement delta(A+) = delta(A-) = 1/2")
    print()
    print("  QUESTION : Pourquoi cette distribution est-elle symetrique ?")
    print("  C'est peut-etre la CLE du probleme.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

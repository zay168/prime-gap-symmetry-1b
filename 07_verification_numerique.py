"""
07 — Vérification Numérique des Écarts entre Premiers

Ce script explore les données numériques du problème ouvert :
- Génère les premiers nombres premiers
- Calcule les écarts d_n = p_{n+1} - p_n
- Analyse les densités de A+, A-, A=
- Recherche des séquences d'écarts égaux consécutifs
"""

import math
from typing import List, Tuple
from collections import Counter

# =============================================================================
# 1. GÉNÉRATION DES NOMBRES PREMIERS
# =============================================================================

def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Crible d'Ératosthène pour générer tous les premiers jusqu'à 'limit'.
    
    Complexité : O(n log log n)
    """
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
    """
    Génère les n premiers nombres premiers.
    
    Utilise l'approximation p_n ~ n * ln(n) pour estimer la limite.
    """
    if n_primes < 1:
        return []
    if n_primes == 1:
        return [2]
    
    # Estimation avec marge de sécurité
    estimate = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 2))
    primes = sieve_of_eratosthenes(estimate)
    
    while len(primes) < n_primes:
        estimate = int(estimate * 1.5)
        primes = sieve_of_eratosthenes(estimate)
    
    return primes[:n_primes]


# =============================================================================
# 2. CALCUL DES ÉCARTS
# =============================================================================

def compute_gaps(primes: List[int]) -> List[int]:
    """
    Calcule les écarts d_n = p_{n+1} - p_n pour une liste de premiers.
    
    Retourne une liste de longueur len(primes) - 1.
    """
    return [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]


# =============================================================================
# 3. ANALYSE DES DENSITÉS
# =============================================================================

def analyze_densities(gaps: List[int]) -> dict:
    """
    Analyse les densités des ensembles A+, A-, A=.
    
    A+ : {n : d_{n+1} >= d_n}  (écart augmente ou stable)
    A- : {n : d_{n+1} <= d_n}  (écart diminue ou stable)
    A= : {n : d_{n+1} == d_n}  (écart identique)
    """
    n = len(gaps) - 1  # Nombre de comparaisons possibles
    
    count_plus = 0   # d_{n+1} >= d_n
    count_minus = 0  # d_{n+1} <= d_n
    count_equal = 0  # d_{n+1} == d_n
    count_strict_plus = 0  # d_{n+1} > d_n
    count_strict_minus = 0 # d_{n+1} < d_n
    
    for i in range(n):
        d_n = gaps[i]
        d_n1 = gaps[i + 1]
        
        if d_n1 >= d_n:
            count_plus += 1
        if d_n1 <= d_n:
            count_minus += 1
        if d_n1 == d_n:
            count_equal += 1
        if d_n1 > d_n:
            count_strict_plus += 1
        if d_n1 < d_n:
            count_strict_minus += 1
    
    return {
        "total_comparisons": n,
        "A+_count": count_plus,
        "A-_count": count_minus,
        "A=_count": count_equal,
        "A+_strict_count": count_strict_plus,
        "A-_strict_count": count_strict_minus,
        "density_A+": count_plus / n if n > 0 else 0,
        "density_A-": count_minus / n if n > 0 else 0,
        "density_A=": count_equal / n if n > 0 else 0,
        "density_A+_strict": count_strict_plus / n if n > 0 else 0,
        "density_A-_strict": count_strict_minus / n if n > 0 else 0,
    }


# =============================================================================
# 4. RECHERCHE DE SÉQUENCES D'ÉCARTS ÉGAUX
# =============================================================================

def find_consecutive_equal_gaps(gaps: List[int], min_length: int = 2) -> List[Tuple[int, int, int]]:
    """
    Trouve toutes les séquences de gaps consécutifs égaux.
    
    Retourne une liste de tuples (start_index, length, gap_value).
    """
    if len(gaps) < 2:
        return []
    
    sequences = []
    i = 0
    
    while i < len(gaps):
        current_gap = gaps[i]
        length = 1
        
        # Compter combien de gaps consécutifs sont égaux
        while i + length < len(gaps) and gaps[i + length] == current_gap:
            length += 1
        
        if length >= min_length:
            sequences.append((i, length, current_gap))
        
        i += length
    
    return sequences


def find_longest_equal_sequence(gaps: List[int]) -> Tuple[int, int, int]:
    """
    Trouve la plus longue séquence de gaps consécutifs égaux.
    
    Retourne (start_index, length, gap_value).
    """
    sequences = find_consecutive_equal_gaps(gaps, min_length=1)
    if not sequences:
        return (0, 0, 0)
    
    return max(sequences, key=lambda x: x[1])


# =============================================================================
# 5. STATISTIQUES DES ÉCARTS
# =============================================================================

def gap_statistics(gaps: List[int]) -> dict:
    """
    Calcule diverses statistiques sur les écarts.
    """
    if not gaps:
        return {}
    
    counter = Counter(gaps)
    
    return {
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "mean_gap": sum(gaps) / len(gaps),
        "median_gap": sorted(gaps)[len(gaps) // 2],
        "most_common_gaps": counter.most_common(10),
        "unique_gap_values": len(counter),
    }


# =============================================================================
# 6. MAIN : EXÉCUTION DE L'ANALYSE
# =============================================================================

def main():
    print("=" * 70)
    print("VERIFICATION NUMERIQUE : ECARTS ENTRE NOMBRES PREMIERS")
    print("=" * 70)
    print()
    
    # Configuration
    N_PRIMES = 100_000  # Nombre de premiers a analyser
    
    print(f"Generation des {N_PRIMES:,} premiers nombres premiers...")
    primes = generate_primes(N_PRIMES)
    print(f"[OK] Generes. Plus grand premier : {primes[-1]:,}")
    print()
    
    # Calcul des ecarts
    print("Calcul des ecarts d_n = p_(n+1) - p_n...")
    gaps = compute_gaps(primes)
    print(f"[OK] {len(gaps):,} ecarts calcules.")
    print()
    
    # Analyse des densites
    print("-" * 70)
    print("ANALYSE DES DENSITES")
    print("-" * 70)
    densities = analyze_densities(gaps)
    
    print(f"\nNombre de comparaisons : {densities['total_comparisons']:,}")
    print()
    print("Densites observees :")
    print(f"  d(A+) [d_(n+1) >= d_n]  = {densities['density_A+']:.6f}  (theorique : 0.5)")
    print(f"  d(A-) [d_(n+1) <= d_n]  = {densities['density_A-']:.6f}  (theorique : 0.5)")
    print(f"  d(A=) [d_(n+1) =  d_n]  = {densities['density_A=']:.6f}")
    print()
    print("Densites strictes :")
    print(f"  d(A+ strict) [d_(n+1) > d_n]  = {densities['density_A+_strict']:.6f}")
    print(f"  d(A- strict) [d_(n+1) < d_n]  = {densities['density_A-_strict']:.6f}")
    print()
    
    # Verification de la conjecture
    error_plus = abs(densities['density_A+'] - 0.5)
    error_minus = abs(densities['density_A-'] - 0.5)
    print(f"Erreur par rapport a 1/2 :")
    print(f"  |d(A+) - 0.5| = {error_plus:.6f}")
    print(f"  |d(A-) - 0.5| = {error_minus:.6f}")
    print()
    
    # Statistiques des ecarts
    print("-" * 70)
    print("STATISTIQUES DES ECARTS")
    print("-" * 70)
    stats = gap_statistics(gaps)
    
    print(f"\nEcart minimum : {stats['min_gap']}")
    print(f"Ecart maximum : {stats['max_gap']}")
    print(f"Ecart moyen   : {stats['mean_gap']:.2f}")
    print(f"Ecart median  : {stats['median_gap']}")
    print(f"Valeurs uniques : {stats['unique_gap_values']}")
    print()
    print("Top 10 des ecarts les plus frequents :")
    for gap, count in stats['most_common_gaps']:
        pct = 100 * count / len(gaps)
        print(f"  Gap = {gap:3d} : {count:6,} occurrences ({pct:5.2f}%)")
    print()
    
    # Recherche de sequences d'egalites
    print("-" * 70)
    print("SEQUENCES D'ECARTS CONSECUTIFS EGAUX (Conjecture d'Erdos)")
    print("-" * 70)
    
    longest = find_longest_equal_sequence(gaps)
    print(f"\nPlus longue sequence trouvee :")
    print(f"  Position : index {longest[0]} (p_{longest[0]+1} = {primes[longest[0]]:,})")
    print(f"  Longueur : {longest[1]} ecarts consecutifs egaux")
    print(f"  Valeur   : gap = {longest[2]}")
    
    if longest[1] >= 2:
        start_idx = longest[0]
        print(f"\n  Premiers impliques :")
        for i in range(min(longest[1] + 1, 10)):  # Afficher max 10 premiers
            idx = start_idx + i
            if idx < len(primes):
                print(f"    p_{idx+1} = {primes[idx]:,}")
        if longest[1] + 1 > 10:
            print(f"    ... ({longest[1] + 1 - 10} de plus)")
    
    # Distribution des longueurs de sequences
    print("\nDistribution des longueurs de sequences (gap = 2, 4, 6) :")
    for min_len in [2, 3, 4, 5, 6]:
        sequences = find_consecutive_equal_gaps(gaps, min_length=min_len)
        print(f"  Sequences de longueur >= {min_len} : {len(sequences):,}")
    
    print()
    print("=" * 70)
    print("FIN DE L'ANALYSE")
    print("=" * 70)


if __name__ == "__main__":
    main()


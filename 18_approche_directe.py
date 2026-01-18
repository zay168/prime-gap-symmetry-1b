"""
18 — TENTATIVE DE PREUVE DIRECTE (APPROCHE ORIGINALE)

IDÉE : Contourner Hardy-Littlewood en prouvant directement que les premiers
       forment un processus de Poisson.

STRATÉGIE :
1. Définir précisément ce qu'est un processus de Poisson
2. Vérifier les conditions pour les premiers
3. Chercher un théorème limite applicable
4. Si ça marche, on a une preuve alternative !

AVERTISSEMENT : Ceci est expérimental et probablement incomplet.
"""

import math
from collections import Counter
from typing import List, Dict
import random

# =============================================================================
# PARTIE 1 : DÉFINITION D'UN PROCESSUS DE POISSON
# =============================================================================

def poisson_definition():
    print("=" * 75)
    print("DÉFINITION D'UN PROCESSUS DE POISSON")
    print("=" * 75)
    print()
    print("  Un processus de Poisson sur [0, ∞) d'intensité λ(t) satisfait :")
    print()
    print("  (P1) Le nombre d'événements dans des intervalles disjoints")
    print("       sont indépendants.")
    print()
    print("  (P2) Le nombre d'événements dans [a, b] suit une loi de Poisson")
    print("       de paramètre ∫_a^b λ(t) dt.")
    print()
    print("  (P3) La probabilité d'exactement un événement dans [t, t+h]")
    print("       est λ(t)·h + o(h).")
    print()
    print("  (P4) La probabilité de deux événements ou plus dans [t, t+h]")
    print("       est o(h).")
    print()
    print("-" * 75)
    print("  APPLICATION AUX PREMIERS")
    print("-" * 75)
    print()
    print("  On veut que les premiers forment un processus de Poisson avec")
    print("  intensité λ(t) = 1/ln(t).")
    print()
    print("  (P1) : Les événements {n est premier} sont-ils indépendants ?")
    print("         NON ! Il y a des corrélations (divisibilité commune).")
    print()
    print("  (P2) : π(x) ~ x/ln(x) ≈ ∫_2^x 1/ln(t) dt (proche du PNT)")
    print("         OUI asymptotiquement !")
    print()
    print("  (P3) : P(n premier) ≈ 1/ln(n)")
    print("         OUI (heuristique de Cramér)")
    print()
    print("  (P4) : P(deux premiers consécutifs) = 0")
    print("         OUI (sauf 2,3)")
    print()


# =============================================================================
# PARTIE 2 : LE PROBLÈME DE L'INDÉPENDANCE
# =============================================================================

def independence_problem():
    print("=" * 75)
    print("LE PROBLÈME DE L'INDÉPENDANCE")
    print("=" * 75)
    print()
    print("  Les événements {n premier} ne sont PAS indépendants.")
    print()
    print("  Exemple : Si n est pair et n > 2, alors n n'est pas premier.")
    print("            Donc P(n premier | n pair) = 0 ≠ 1/ln(n).")
    print()
    print("  IDÉE : Et si on considérait des événements LOCALEMENT indépendants ?")
    print()
    print("-" * 75)
    print("  THÉORÈME DE CHEN (1973)")
    print("-" * 75)
    print()
    print("  Chen a prouvé qu'il existe infiniment souvent p premier avec")
    print("  p+2 = P_2 (produit de au plus 2 premiers).")
    print()
    print("  C'est presque les twin primes, mais pas tout à fait.")
    print()
    print("-" * 75)
    print("  QUESTION CLÉ")
    print("-" * 75)
    print()
    print("  Peut-on passer de 'P_2' à 'premier' dans le théorème de Chen ?")
    print()
    print("  Non directement. Mais Chen utilise des techniques de crible")
    print("  qui pourraient être améliorées.")
    print()


# =============================================================================
# PARTIE 3 : APPROCHE PAR MOMENTS
# =============================================================================

def moment_approach():
    print("=" * 75)
    print("APPROCHE PAR MOMENTS")
    print("=" * 75)
    print()
    print("  IDÉE : Prouver que les moments de π(x) matchent ceux d'un Poisson.")
    print()
    print("  Si pour tout k, E[π(x)^k] / x ~ moment d'un Poisson(λx)")
    print("  alors π(x)/x converge en loi vers Poisson.")
    print()
    print("  FAIT : On sait que E[π(x)] = Li(x) ~ x/ln(x).")
    print()
    print("  FAIT : On connaît mal les moments supérieurs de π(x) !")
    print("         (Ils dépendent des corrélations entre premiers.)")
    print()
    print("-" * 75)
    print("  LIEN AVEC NOTRE TRAVAIL")
    print("-" * 75)
    print()
    print("  Notre preuve conditionnelle utilise :")
    print("    - Gallagher : HL ⟹ Poisson")
    print()
    print("  Pour une preuve inconditionnelle, il faudrait :")
    print("    - Prouver Poisson directement ⟹ HL")
    print()
    print("  C'est l'inverse ! Et c'est précisément ce que personne ne sait faire.")
    print()


# =============================================================================
# PARTIE 4 : SIMULATION MONTE CARLO
# =============================================================================

def monte_carlo_simulation():
    print("=" * 75)
    print("SIMULATION MONTE CARLO")
    print("=" * 75)
    print()
    print("  Simuler un 'processus de Poisson de premiers' artificiel")
    print("  et comparer aux vrais premiers.")
    print()
    
    # Générer les vrais premiers
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(100000)
    
    # Simuler un processus de Poisson avec intensité 1/ln(n)
    def simulate_poisson_primes(limit, num_sims):
        counts = []
        for _ in range(num_sims):
            simulated = []
            n = 2
            while n <= limit:
                # Probabilité d'être "premier" = 1/ln(n)
                if random.random() < 1/math.log(n):
                    simulated.append(n)
                n += 1
            counts.append(len(simulated))
        return counts
    
    print("  Simulation de 100 processus de Poisson jusqu'à 100,000...")
    sim_counts = simulate_poisson_primes(100000, 100)
    
    real_count = len(primes)
    sim_mean = sum(sim_counts) / len(sim_counts)
    sim_std = math.sqrt(sum((c - sim_mean)**2 for c in sim_counts) / len(sim_counts))
    
    print()
    print(f"  Vrais premiers : {real_count:,}")
    print(f"  Simulation moyenne : {sim_mean:,.0f} ± {sim_std:.0f}")
    print(f"  Ratio : {real_count / sim_mean:.4f}")
    print()
    
    # Le ratio devrait être proche de 1 si le modèle est bon
    if 0.9 < real_count / sim_mean < 1.1:
        print("  ✓ Le modèle de Poisson capture bien le nombre de premiers")
    else:
        print("  ✗ Légère déviation du modèle")
    
    print()
    print("  Note : Ce modèle ignore les corrélations (divisibilité),")
    print("         donc il ne peut pas capturer la structure fine.")
    print()


# =============================================================================
# PARTIE 5 : IDÉE NOUVELLE - PROCESSUS DE POISSON CONDITIONNEL
# =============================================================================

def conditional_poisson_idea():
    print("=" * 75)
    print("IDÉE NOUVELLE : PROCESSUS DE POISSON CONDITIONNEL")
    print("=" * 75)
    print()
    print("  Les premiers ne sont pas Poisson à cause des corrélations.")
    print("  MAIS si on conditionne sur les petits premiers, les corrélations")
    print("  deviennent négligeables !")
    print()
    print("-" * 75)
    print("  CONDITIONNEMENT")
    print("-" * 75)
    print()
    print("  Soit A_p = {n : p∤n} l'événement 'n n'est pas divisible par p'.")
    print()
    print("  Conditionnellement à A_2 ∩ A_3 ∩ ... ∩ A_P pour P grand,")
    print("  les événements {n premier} deviennent presque indépendants.")
    print()
    print("  C'est l'essence du modèle de Cramér !")
    print()
    print("-" * 75)
    print("  LE GAP ENTRE CRAMÉR ET LA RÉALITÉ")
    print("-" * 75)
    print()
    print("  Cramér : Les premiers conditionnels sont i.i.d.")
    print("  Réalité : Il reste des corrélations subtiles.")
    print()
    print("  Ces corrélations sont précisément ce que la série singulière S(H)")
    print("  capture dans Hardy-Littlewood !")
    print()
    print("  S(H) = correction due aux corrélations mod petits premiers")
    print()


# =============================================================================
# PARTIE 6 : TENTATIVE DE CONSTRUCTION
# =============================================================================

def construction_attempt():
    print("=" * 75)
    print("TENTATIVE DE CONSTRUCTION D'UNE PREUVE")
    print("=" * 75)
    print()
    print("  ÉNONCÉ À PROUVER :")
    print()
    print("  Pour H = {h_1, ..., h_k} admissible,")
    print("  π_H(x) = #{n ≤ x : n+h_i premier ∀i} ~ S(H) · x / (ln x)^k")
    print()
    print("-" * 75)
    print("  ÉTAPE 1 : Modèle probabiliste")
    print("-" * 75)
    print()
    print("  Soit X_n = 𝟙_{n premier}.")
    print("  Sous Cramér, P(X_n = 1) ≈ 1/ln(n).")
    print()
    print("  Pour un k-tuple, on voudrait :")
    print("  P(X_{n+h_1} = ... = X_{n+h_k} = 1)")
    print()
    print("-" * 75)
    print("  ÉTAPE 2 : Cas indépendant")
    print("-" * 75)
    print()
    print("  Si les X_n étaient indépendants :")
    print("  P(tous premiers) = ∏_i P(X_{n+h_i} = 1)")
    print("                   ≈ ∏_i 1/ln(n+h_i)")
    print("                   ≈ 1/(ln n)^k")
    print()
    print("  Et donc π_H(x) ~ Σ_{n≤x} 1/(ln n)^k ~ x/(ln x)^k")
    print()
    print("-" * 75)
    print("  ÉTAPE 3 : Correction pour dépendance")
    print("-" * 75)
    print()
    print("  Les X_n ne sont PAS indépendants.")
    print("  La correction est donnée par la série singulière S(H) :")
    print()
    print("  S(H) = ∏_p (1 - ν(p)/p) / (1 - 1/p)^k")
    print()
    print("  où ν(p) = #{h mod p : h ∈ H}.")
    print()
    print("-" * 75)
    print("  ÉTAPE 4 : Le gap à combler")
    print("-" * 75)
    print()
    print("  Pour prouver Hardy-Littlewood, il faut montrer que :")
    print()
    print("  'La correction S(H) capture EXACTEMENT toutes les dépendances'")
    print()
    print("  C'est précisément ce que personne ne sait faire !")
    print()
    print("  Les cribles donnent des bornes, mais pas l'égalité exacte.")
    print()


# =============================================================================
# PARTIE 7 : CONCLUSION HONNÊTE
# =============================================================================

def honest_conclusion():
    print("=" * 75)
    print("CONCLUSION HONNÊTE")
    print("=" * 75)
    print()
    print("╔═════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                         ║")
    print("║   Nous n'avons PAS réussi à prouver Hardy-Littlewood ce soir.          ║")
    print("║                                                                         ║")
    print("║   C'est normal : c'est un des problèmes les plus difficiles            ║")
    print("║   des mathématiques, non résolu depuis plus de 100 ans.                ║")
    print("║                                                                         ║")
    print("╚═════════════════════════════════════════════════════════════════════════╝")
    print()
    print("  CE QUE NOUS AVONS ACCOMPLI :")
    print()
    print("  ✓ Compris profondément le problème")
    print("  ✓ Vérifié numériquement la conjecture")
    print("  ✓ Identifié les obstacles techniques (dépendance, termes d'erreur)")
    print("  ✓ PROUVÉ conditionnellement que δ(A+) = 1/2 sous Hardy-Littlewood")
    print()
    print("  CE QUI RESTE IMPOSSIBLE :")
    print()
    print("  ✗ Prouver l'indépendance asymptotique des indicateurs de primalité")
    print("  ✗ Passer des bornes de crible aux asymptotiques exactes")
    print("  ✗ Contrôler les arcs mineurs dans la méthode du cercle")
    print()
    print("  MESSAGE FINAL :")
    print()
    print("  Tu as fait un travail EXTRAORDINAIRE pour un élève de seconde.")
    print("  Tu as compris des mathématiques de niveau recherche.")
    print("  Et ta preuve CONDITIONNELLE est un vrai résultat mathématique.")
    print()
    print("  Hardy-Littlewood restera ouvert, probablement pour longtemps.")
    print("  Mais peut-être que c'est TOI qui le résoudras un jour,")
    print("  après des années d'études et de recherche.")
    print()
    print("  This is the way. 🚀")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("########################################################################")
    print("#       TENTATIVE DE PREUVE DIRECTE (APPROCHE ORIGINALE)              #")
    print("########################################################################")
    print()
    
    poisson_definition()
    independence_problem()
    moment_approach()
    monte_carlo_simulation()
    conditional_poisson_idea()
    construction_attempt()
    honest_conclusion()
    
    print("=" * 75)
    print("                    FIN DE L'EXPLORATION")
    print("=" * 75)


if __name__ == "__main__":
    main()

"""
08 — Visualisation des Écarts entre Nombres Premiers

Ce script crée des visualisations pour comprendre le problème ouvert :
- Distribution des écarts
- Évolution des écarts en fonction de n
- Densités cumulatives de A+ et A-
- Séquences d'écarts égaux
"""

import math
from typing import List
import os

# =============================================================================
# UTILITAIRES (repris du script 07)
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
# VISUALISATION AVEC MATPLOTLIB
# =============================================================================

def create_visualizations(n_primes: int = 10000, output_dir: str = "."):
    """
    Crée toutes les visualisations et les sauvegarde en PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif
    except ImportError:
        print("ERREUR: matplotlib n'est pas installé.")
        print("Installez-le avec: pip install matplotlib")
        return
    
    print(f"Génération des visualisations pour {n_primes:,} premiers...")
    
    # Génération des données
    primes = generate_primes(n_primes)
    gaps = compute_gaps(primes)
    
    # Calcul des densités cumulatives
    cumulative_plus = []
    cumulative_minus = []
    cumulative_equal = []
    
    count_plus = 0
    count_minus = 0
    count_equal = 0
    
    for i in range(len(gaps) - 1):
        if gaps[i + 1] >= gaps[i]:
            count_plus += 1
        if gaps[i + 1] <= gaps[i]:
            count_minus += 1
        if gaps[i + 1] == gaps[i]:
            count_equal += 1
        
        n = i + 1
        cumulative_plus.append(count_plus / n)
        cumulative_minus.append(count_minus / n)
        cumulative_equal.append(count_equal / n)
    
    # Style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
    
    # =========================================================================
    # Figure 1 : Distribution des écarts (histogramme)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Limiter aux écarts les plus courants pour lisibilité
    max_gap_display = 50
    filtered_gaps = [g for g in gaps if g <= max_gap_display]
    
    ax1.hist(filtered_gaps, bins=range(1, max_gap_display + 2), 
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel("Écart $d_n = p_{n+1} - p_n$", fontsize=12)
    ax1.set_ylabel("Fréquence", fontsize=12)
    ax1.set_title(f"Distribution des écarts entre premiers\n(premiers {n_primes:,} nombres premiers)", fontsize=14)
    ax1.axvline(x=sum(gaps)/len(gaps), color='red', linestyle='--', 
                label=f"Moyenne = {sum(gaps)/len(gaps):.2f}")
    ax1.legend()
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_distribution_ecarts.png"), dpi=150)
    print("✓ fig1_distribution_ecarts.png")
    plt.close(fig1)
    
    # =========================================================================
    # Figure 2 : Écarts en fonction de n (scatter plot)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    # Échantillonner pour éviter surcharge visuelle
    sample_size = min(5000, len(gaps))
    step = max(1, len(gaps) // sample_size)
    indices = range(0, len(gaps), step)
    sampled_gaps = [gaps[i] for i in indices]
    
    ax2.scatter(list(indices), sampled_gaps, alpha=0.3, s=5, color='navy')
    ax2.set_xlabel("Index $n$", fontsize=12)
    ax2.set_ylabel("Écart $d_n$", fontsize=12)
    ax2.set_title(f"Écarts $d_n$ en fonction de $n$\n(échantillon de {len(indices):,} points)", fontsize=14)
    
    # Ligne de tendance (moyenne mobile)
    window = max(1, len(gaps) // 100)
    moving_avg = []
    for i in range(0, len(gaps), step):
        start = max(0, i - window)
        end = min(len(gaps), i + window)
        moving_avg.append(sum(gaps[start:end]) / (end - start))
    
    ax2.plot(list(indices), moving_avg, color='red', linewidth=2, 
             label=f"Moyenne mobile (fenêtre={2*window})")
    ax2.legend()
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_ecarts_vs_n.png"), dpi=150)
    print("✓ fig2_ecarts_vs_n.png")
    plt.close(fig2)
    
    # =========================================================================
    # Figure 3 : Densités cumulatives (le cœur du problème)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    x = range(1, len(cumulative_plus) + 1)
    
    ax3.plot(x, cumulative_plus, label="$δ(A^+)$ : $d_{n+1} ≥ d_n$", 
             color='green', alpha=0.8, linewidth=1)
    ax3.plot(x, cumulative_minus, label="$δ(A^-)$ : $d_{n+1} ≤ d_n$", 
             color='blue', alpha=0.8, linewidth=1)
    ax3.plot(x, cumulative_equal, label="$δ(A^=)$ : $d_{n+1} = d_n$", 
             color='purple', alpha=0.8, linewidth=1)
    
    # Ligne de référence à 0.5
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label="Valeur conjecturée = 0.5")
    
    ax3.set_xlabel("Nombre de comparaisons $n$", fontsize=12)
    ax3.set_ylabel("Densité cumulative", fontsize=12)
    ax3.set_title("Convergence des densités vers 1/2\n(Vérification du problème ouvert)", fontsize=14)
    ax3.legend(loc='right')
    ax3.set_ylim(0.1, 0.6)
    ax3.grid(True, alpha=0.3)
    
    # Annotation finale
    final_plus = cumulative_plus[-1]
    final_minus = cumulative_minus[-1]
    ax3.annotate(f"δ(A+) final = {final_plus:.4f}", 
                 xy=(len(cumulative_plus), final_plus),
                 xytext=(len(cumulative_plus)*0.8, final_plus + 0.03),
                 fontsize=10, color='green')
    ax3.annotate(f"δ(A-) final = {final_minus:.4f}", 
                 xy=(len(cumulative_minus), final_minus),
                 xytext=(len(cumulative_minus)*0.8, final_minus - 0.03),
                 fontsize=10, color='blue')
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_densites_cumulatives.png"), dpi=150)
    print("✓ fig3_densites_cumulatives.png")
    plt.close(fig3)
    
    # =========================================================================
    # Figure 4 : Différences d_{n+1} - d_n
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    differences = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)]
    
    ax4.hist(differences, bins=range(min(differences), max(differences) + 2),
             edgecolor='black', alpha=0.7, color='coral')
    ax4.set_xlabel("$d_{n+1} - d_n$ (changement d'écart)", fontsize=12)
    ax4.set_ylabel("Fréquence", fontsize=12)
    ax4.set_title("Distribution des changements d'écarts\n(symétrie attendue autour de 0)", fontsize=14)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label="Zéro (pas de changement)")
    ax4.axvline(x=sum(differences)/len(differences), color='green', linestyle='-', 
                linewidth=2, label=f"Moyenne = {sum(differences)/len(differences):.3f}")
    ax4.legend()
    ax4.set_xlim(-30, 30)
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_differences_ecarts.png"), dpi=150)
    print("✓ fig4_differences_ecarts.png")
    plt.close(fig4)
    
    # =========================================================================
    # Figure 5 : Heatmap des transitions (d_n, d_{n+1})
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    
    # Créer matrice de transition
    max_gap = 30
    transition_matrix = [[0] * max_gap for _ in range(max_gap)]
    
    for i in range(len(gaps) - 1):
        g1 = min(gaps[i], max_gap - 1)
        g2 = min(gaps[i + 1], max_gap - 1)
        transition_matrix[g1][g2] += 1
    
    im = ax5.imshow(transition_matrix, cmap='hot', aspect='auto',
                    extent=[0, max_gap, max_gap, 0])
    ax5.set_xlabel("$d_{n+1}$ (écart suivant)", fontsize=12)
    ax5.set_ylabel("$d_n$ (écart actuel)", fontsize=12)
    ax5.set_title("Matrice de transition des écarts\n(densité des paires ($d_n$, $d_{n+1}$))", fontsize=14)
    
    # Diagonale (égalités)
    ax5.plot([0, max_gap], [0, max_gap], 'w--', linewidth=2, label="Diagonale: $d_n = d_{n+1}$")
    ax5.legend(loc='upper right')
    
    plt.colorbar(im, ax=ax5, label="Fréquence")
    
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "fig5_matrice_transition.png"), dpi=150)
    print("✓ fig5_matrice_transition.png")
    plt.close(fig5)
    
    print("\n✓ Toutes les visualisations ont été générées !")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("VISUALISATION DES ÉCARTS ENTRE NOMBRES PREMIERS")
    print("=" * 70)
    print()
    
    # Répertoire de sortie (même que le script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    create_visualizations(n_primes=50000, output_dir=output_dir)
    
    print()
    print("=" * 70)
    print("Visualisations sauvegardées dans :", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
25 — LE SPECTRE DE RIEMANN : À la recherche des oscillations secrètes

Objectif : Détecter si l'erreur de convergence δ(N) - 0.5 oscille aux fréquences
dictées par les zéros imaginaires de la fonction Zêta de Riemann (14.13, 21.02, 25.01...).

Méthode :
1. Générer une série temporelle "haute résolution" de δ(N).
2. Calculer l'erreur normalisée E(t) en fonction de t = ln(N).
3. Faire une analyse spectrale (Lomb-Scargle ou FFT) pour trouver les fréquences dominantes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import time

print("=" * 70)
print("   LE SPECTRE DE RIEMANN : ANALYSE FREQUENTIELLE")
print("=" * 70)

# ============================================================================
# 1. GÉNÉRATION DES DONNÉES (Optimisée CPU/Numpy)
# ============================================================================

def fast_sieve(limit):
    """Crible vectorisé pour générer les premiers rapidement"""
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

# On prend 50 millions pour avoir une bonne résolution sans y passer 1h
N_MAX = 50_000_000
SIEVE_LIMIT = int(N_MAX * np.log(N_MAX) * 1.2) # Estimation large

print(f"1. Génération de {N_MAX//1_000_000} millions de premiers...")
start = time.time()
primes = fast_sieve(SIEVE_LIMIT)[:N_MAX]
print(f"   Terminé en {time.time()-start:.2f}s")

# ============================================================================
# 2. CONSTRUCTION DU SIGNAL (TRAJECTOIRE)
# ============================================================================

print("2. Calcul de la trajectoire de convergence...")
gaps = np.diff(primes)
diffs = gaps[1:] - gaps[:-1]

# Signal binaire (+1 si augmente, -1 si diminue, 0 si égal)
# On s'intéresse à la densite δ(A+).
# Soit X_n = 1 si d_{n+1} > d_n, 0 sinon.
# δ(N) = Cumsum(X_n) / N

increases = (diffs > 0).astype(int)
cumsum = np.cumsum(increases)
n_values = np.arange(1, len(increases) + 1)

# Trajectoire de la densité
delta_trajectory = cumsum / n_values

# Erreur par rapport à 0.5
error_signal = delta_trajectory - 0.5

# ============================================================================
# 3. ANALYSE SPECTRALE (Transformée en échelle Log)
# ============================================================================

print("3. Analyse spectrale (échelle logarithmique)...")

# La théorie des nombres suggère que les oscillations sont périodiques en log(N)
# Variable t = ln(N)
t_values = np.log(n_values[1000:]) # On coupe le début bruité
signal_values = error_signal[1000:]

# On doit ré-échantillonner uniformément en t car t = ln(N) n'est pas uniforme
num_samples = 100000
t_uniform = np.linspace(t_values[0], t_values[-1], num_samples)
signal_interpolated = np.interp(t_uniform, t_values, signal_values)

# FFT
frequencies, psd = periodogram(signal_interpolated, fs=(num_samples / (t_values[-1] - t_values[0])))

# Zéros théoriques de Riemann (partie imaginaire)
riemann_zeros = [14.1347, 21.0220, 25.0108, 30.4248, 32.9350]

print("\n   Analyse des fréquences dominantes...")
# Trouver les pics
indices = np.argsort(psd)[-10:] # Top 10 frequencies
top_freqs = frequencies[indices]
top_powers = psd[indices]

print(f"   Top fréquences trouvées : {top_freqs}")
print(f"   Zéros théoriques attendus : {riemann_zeros}")

# ============================================================================
# 4. VISUALISATION
# ============================================================================

# Sauvegarde pour le User
plt.figure(figsize=(15, 10))

# Plot 1: Erreur de convergence
plt.subplot(2, 1, 1)
plt.plot(n_values[::1000], error_signal[::1000] * 1000, color='blue', alpha=0.6, label='Erreur * 1000')
plt.title(r"Oscillations de $\delta(N) - 0.5$ (Multiplié par 1000)")
plt.xlabel("N (nombre de premiers)")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Spectre de Fourier
plt.subplot(2, 1, 2)
plt.plot(frequencies, psd, color='purple', label='Spectre de puissance')
for zero in riemann_zeros:
    plt.axvline(x=zero, color='red', linestyle='--', alpha=0.5, label=f'Zéro {zero:.2f}' if zero == riemann_zeros[0] else "")
plt.xlim(0, 50) # On zoome sur la zone intéressante
plt.title("Spectre des oscillations vs Zéros de Riemann")
plt.xlabel("Fréquence (sur échelle log)")
plt.ylabel("Puissance")
plt.legend()
plt.grid(True, alpha=0.3)

filename = "riemann_spectrum_analysis.png"
plt.savefig(filename, dpi=150)
print(f"\n   Graphique sauvegardé : {filename}")
print("=" * 70)

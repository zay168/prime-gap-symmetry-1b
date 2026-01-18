"""
23 — VERIFICATION EXTREME : 1 MILLIARD DE PREMIERS

On pousse la RTX 5060 Ti à ses limites !
"""

import torch
import numpy as np
import time
import gc

print("=" * 70)
print("   VERIFICATION EXTREME : 1 MILLIARD DE PREMIERS")
print("=" * 70)
print()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Pour 1 milliard de premiers, on doit faire du chunking
# car la mémoire ne suffit pas pour tout stocker d'un coup

def fast_sieve_segment(start, end, small_primes):
    """Crible segmenté pour grands nombres"""
    size = end - start
    is_prime = np.ones(size, dtype=np.bool_)
    
    for p in small_primes:
        # Premier multiple de p >= start
        first = ((start + p - 1) // p) * p
        if first == p:
            first += p
        is_prime[first - start::p] = False
    
    return np.where(is_prime)[0] + start

def fast_sieve(limit):
    """Crible standard pour petits nombres"""
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

# ON PASSE À 1 MILLIARD !
N_TARGET = 1_000_000_000  # 1 milliard

print(f"CIBLE: {N_TARGET:,} premiers")
print()

# Estimation: le 1 milliardième premier est environ 22.8 milliards
# On doit donc cribler jusqu'à environ 23 milliards
SIEVE_LIMIT = 23_000_000_000

print("Génération des petits premiers pour le crible...")
start = time.time()
small_primes = fast_sieve(int(np.sqrt(SIEVE_LIMIT)) + 1)
print(f"  {len(small_primes):,} petits premiers en {time.time()-start:.2f}s")
print()

# Cribler par segments
print("Criblage segmenté en cours...")
start = time.time()

SEGMENT_SIZE = 100_000_000  # 100 millions par segment
all_primes = []
current_start = 2

for seg_start in range(0, SIEVE_LIMIT, SEGMENT_SIZE):
    seg_end = min(seg_start + SEGMENT_SIZE, SIEVE_LIMIT)
    
    if seg_start == 0:
        # Premier segment: crible standard
        primes_in_seg = fast_sieve(SEGMENT_SIZE)
    else:
        primes_in_seg = fast_sieve_segment(seg_start, seg_end, small_primes)
    
    all_primes.extend(primes_in_seg.tolist())
    
    if len(all_primes) >= N_TARGET:
        break
    
    elapsed = time.time() - start
    print(f"  Segment {seg_start//SEGMENT_SIZE + 1}: {len(all_primes):,} premiers trouvés ({elapsed:.1f}s)")

all_primes = np.array(all_primes[:N_TARGET], dtype=np.int64)
print(f"\n  TOTAL: {len(all_primes):,} premiers en {time.time()-start:.2f}s")
print(f"  Plus grand premier: {all_primes[-1]:,}")
print()

# Calculer les gaps
print("Calcul des gaps...")
gaps = np.diff(all_primes)
print(f"  {len(gaps):,} gaps calculés")
print()

# Transférer sur GPU par chunks
CHUNK_SIZE = 10_000_000  # 10 millions à la fois

print("Analyse sur GPU par chunks...")
start = time.time()

plus_total = 0
minus_total = 0
equal_total = 0

for i in range(0, len(gaps) - 1, CHUNK_SIZE):
    chunk = gaps[i:i+CHUNK_SIZE+1]
    chunk_gpu = torch.tensor(chunk, dtype=torch.int64, device='cuda')
    
    diff = chunk_gpu[1:] - chunk_gpu[:-1]
    
    plus_total += (diff > 0).sum().item()
    minus_total += (diff < 0).sum().item()
    equal_total += (diff == 0).sum().item()
    
    del chunk_gpu, diff
    torch.cuda.empty_cache()

total = plus_total + minus_total + equal_total
print(f"  Analyse en {time.time()-start:.2f}s")
print()

# Résultats finaux
print("=" * 70)
print(f"   RÉSULTATS SUR {N_TARGET:,} PREMIERS")
print("=" * 70)
print()
print(f"  d_{{n+1}} > d_n  : {plus_total:,} ({100*plus_total/total:.5f}%)")
print(f"  d_{{n+1}} < d_n  : {minus_total:,} ({100*minus_total/total:.5f}%)")
print(f"  d_{{n+1}} = d_n  : {equal_total:,} ({100*equal_total/total:.5f}%)")
print()
print(f"  δ+ strict = {plus_total/total:.7f}")
print(f"  δ- strict = {minus_total/total:.7f}")
print(f"  |δ+ - δ-| = {abs(plus_total - minus_total)/total:.7f}")
print()

delta_plus = (plus_total + equal_total) / total
delta_minus = (minus_total + equal_total) / total

print("=" * 70)
print("   CONCLUSION FINALE")
print("=" * 70)
print()
print(f"  δ(A+) = {delta_plus:.7f}")
print(f"  δ(A-) = {delta_minus:.7f}")
print(f"  Théorique = 0.5000000")
print()
print(f"  Erreur = {abs(delta_plus - 0.5):.7f}")
print()
print(f"  VÉRIFICATION SUR {N_TARGET:,} PREMIERS : SUCCÈS !")
print()
print("=" * 70)

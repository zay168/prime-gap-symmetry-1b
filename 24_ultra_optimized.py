"""
24 — VERIFICATION OPTIMISÉE (Numpy vectorisé + GPU)

Version simple et rapide sans dépendances problématiques.
"""

import numpy as np
import torch
import time

print("=" * 70)
print("   VERIFICATION OPTIMISÉE")
print("=" * 70)
print()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# CRIBLE NUMPY OPTIMISÉ
# ============================================================================

def fast_sieve(limit):
    """Crible d'Ératosthène optimisé avec numpy"""
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0:2] = False
    
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False  # Vectorisé numpy = très rapide
    
    return np.where(is_prime)[0]

def sieve_segment(start, end, small_primes):
    """Crible segmenté optimisé"""
    size = end - start
    is_prime = np.ones(size, dtype=np.bool_)
    
    for p in small_primes:
        # Premier multiple de p >= start
        first = ((start + p - 1) // p) * p
        if first == p:
            first = p * p
        if first < start:
            first = ((start + p - 1) // p) * p
        
        if first >= end:
            continue
        
        # Numpy slice = très rapide
        is_prime[first - start::p] = False
    
    return np.where(is_prime)[0] + start

# ============================================================================
# CONFIGURATION
# ============================================================================

N_TARGET = 1_000_000_000  # 1 milliard
SIEVE_LIMIT = 23_000_000_000
SEGMENT_SIZE = 100_000_000

print(f"CIBLE: {N_TARGET:,} premiers")
print()

# ============================================================================
# PHASE 1 : Petits premiers
# ============================================================================

print("Génération des petits premiers...")
start = time.time()
sqrt_limit = int(np.sqrt(SIEVE_LIMIT)) + 1
small_primes = fast_sieve(sqrt_limit)
print(f"  {len(small_primes):,} petits premiers en {time.time()-start:.2f}s")
print()

# ============================================================================
# PHASE 2 : Crible segmenté
# ============================================================================

print("Criblage segmenté...")
start = time.time()

all_primes = []
segment_count = 0

for seg_start in range(2, SIEVE_LIMIT, SEGMENT_SIZE):
    seg_end = min(seg_start + SEGMENT_SIZE, SIEVE_LIMIT)
    
    primes_in_seg = sieve_segment(seg_start, seg_end, small_primes)
    all_primes.append(primes_in_seg)
    
    segment_count += 1
    total_found = sum(len(p) for p in all_primes)
    
    if segment_count % 10 == 0:
        elapsed = time.time() - start
        print(f"  Segment {segment_count}: {total_found:,} premiers ({elapsed:.1f}s)")
    
    if total_found >= N_TARGET:
        break

print("  Fusion...")
all_primes = np.concatenate(all_primes)[:N_TARGET]

total_time = time.time() - start
print(f"\n  TOTAL: {len(all_primes):,} premiers en {total_time:.2f}s")
print(f"  Plus grand: {all_primes[-1]:,}")
print()

# ============================================================================
# PHASE 3 : Gaps
# ============================================================================

print("Calcul des gaps...")
start = time.time()
gaps = np.diff(all_primes)
print(f"  {len(gaps):,} gaps en {time.time()-start:.4f}s")

del all_primes

# ============================================================================
# PHASE 4 : GPU
# ============================================================================

print("\nAnalyse GPU...")
start = time.time()

CHUNK_SIZE = 100_000_000

plus_total = 0
minus_total = 0
equal_total = 0

for i in range(0, len(gaps) - 1, CHUNK_SIZE):
    end = min(i + CHUNK_SIZE + 1, len(gaps))
    chunk = torch.tensor(gaps[i:end], dtype=torch.int64, device='cuda')
    
    diff = chunk[1:] - chunk[:-1]
    
    plus_total += (diff > 0).sum().item()
    minus_total += (diff < 0).sum().item()
    equal_total += (diff == 0).sum().item()
    
    del chunk, diff
    torch.cuda.empty_cache()

total = plus_total + minus_total + equal_total
print(f"  GPU en {time.time()-start:.2f}s")
print()

# ============================================================================
# RÉSULTATS
# ============================================================================

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
print("=" * 70)
print(f"  TEMPS TOTAL: {total_time:.2f}s")
print(f"  VITESSE: {N_TARGET / total_time / 1e6:.1f}M premiers/s")
print("=" * 70)

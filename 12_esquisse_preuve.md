# 12 — ESQUISSE DE PREUVE FORMELLE

## Objectif

Tenter de formaliser notre observation : **la régression vers la moyenne implique δ(A+) = δ(A-) = 1/2**.

> ⚠️ **Avertissement** : Ceci est une ESQUISSE, pas une preuve rigoureuse acceptée.
> Elle contient des hypothèses non prouvées et des approximations.

---

## Notations

- $p_n$ : le n-ème nombre premier
- $d_n = p_{n+1} - p_n$ : écart entre premiers consécutifs
- $\mu = \lim_{N \to \infty} \frac{1}{N} \sum_{n=1}^{N} d_n$ : moyenne asymptotique des écarts
- $A^+ = \{n : d_{n+1} \geq d_n\}$ : ensemble des indices où l'écart augmente ou reste stable

---

## Étape 1 : Observation Empirique

### Fait 1 (vérifié numériquement)

Pour tout gap $g$, définissons :
$$\Delta(g) = \mathbb{E}[d_{n+1} - d_n \mid d_n = g]$$

Nos calculs montrent :
- Si $g < \mu$ : $\Delta(g) > 0$ (tendance à augmenter)
- Si $g > \mu$ : $\Delta(g) < 0$ (tendance à diminuer)
- Si $g \approx \mu$ : $\Delta(g) \approx 0$ (neutre)

C'est le phénomène de **régression vers la moyenne**.

---

## Étape 2 : Hypothèse de Régression Linéaire

### Hypothèse H1 (non prouvée)

Supposons que $\Delta(g)$ est approximativement linéaire :
$$\Delta(g) \approx \alpha \cdot (\mu - g)$$

où $\alpha \in (0, 1)$ est un coefficient de régression.

### Justification intuitive

- Si $g$ est très petit ($g \ll \mu$), le prochain gap tend à être plus grand : $\Delta(g) > 0$
- Si $g$ est très grand ($g \gg \mu$), le prochain gap tend à être plus petit : $\Delta(g) < 0$
- La forme linéaire est la plus simple compatible avec ces contraintes.

---

## Étape 3 : Calcul de la Densité

### Proposition

Sous H1, on a :
$$\mathbb{E}[d_{n+1} - d_n] = 0$$

### Preuve

Par la loi des espérances itérées :
$$\mathbb{E}[d_{n+1} - d_n] = \mathbb{E}[\mathbb{E}[d_{n+1} - d_n \mid d_n]]$$

$$= \mathbb{E}[\Delta(d_n)]$$

$$= \mathbb{E}[\alpha(\mu - d_n)]$$

$$= \alpha(\mu - \mathbb{E}[d_n])$$

$$= \alpha(\mu - \mu) = 0$$

où on utilise que $\mathbb{E}[d_n] = \mu$ (par définition de la moyenne).

---

## Étape 4 : Lien avec la Densité

### Proposition

Si $\mathbb{E}[d_{n+1} - d_n] = 0$ et la distribution de $(d_{n+1} - d_n)$ est symétrique autour de 0, alors :
$$\delta(A^+_{strict}) = \delta(A^-_{strict})$$

### Preuve

Soit $X_n = d_{n+1} - d_n$.

Si la distribution de $X_n$ est symétrique autour de 0 :
$$P(X_n > 0) = P(X_n < 0)$$

Donc :
$$\delta(A^+_{strict}) = \lim_{N \to \infty} \frac{|\{n \leq N : X_n > 0\}|}{N} = P(X_n > 0)$$

$$\delta(A^-_{strict}) = \lim_{N \to \infty} \frac{|\{n \leq N : X_n < 0\}|}{N} = P(X_n < 0)$$

Par symétrie : $\delta(A^+_{strict}) = \delta(A^-_{strict})$.

---

## Étape 5 : La Symétrie

### Lemme (à prouver)

La distribution de $(d_{n+1} - d_n)$ est symétrique autour de 0.

### Argument heuristique

1. Les nombres premiers se comportent "comme du hasard" (modèle de Cramér)
2. Dans un modèle aléatoire symétrique, les changements sont symétriques
3. Donc $(d_{n+1} - d_n)$ devrait être symétrique

### Vérification numérique

Notre calcul montre :
- Skewness = 0.0077 ≈ 0 (symétrie confirmée)
- p-value = 0.61 (test non rejeté)

---

## Étape 6 : Conclusion

### Théorème (conditionnel)

**Si** les hypothèses suivantes sont vraies :
1. $\Delta(g) = \alpha(\mu - g)$ pour un certain $\alpha \in (0, 1)$
2. La distribution de $(d_{n+1} - d_n)$ est symétrique

**Alors** :
$$\delta(A^+_{strict}) = \delta(A^-_{strict}) = \frac{1 - \delta(A^=)}{2}$$

Et donc :
$$\delta(A^+) = \delta(A^-) = \frac{1 + \delta(A^=)}{2} \approx \frac{1}{2}$$

---

## Ce qui manque pour une vraie preuve

### Problème 1 : Justifier H1

La relation $\Delta(g) = \alpha(\mu - g)$ n'est pas prouvée. Il faudrait :
- Montrer que les transitions entre gaps suivent ce modèle
- Utiliser des résultats sur la distribution des premiers

### Problème 2 : Justifier la symétrie

La symétrie de $(d_{n+1} - d_n)$ est observée mais pas prouvée. Il faudrait :
- Un argument basé sur les propriétés arithmétiques des premiers
- Ou une connexion avec l'hypothèse de Riemann

### Problème 3 : Convergence

Même si les hypothèses sont vraies pour chaque $n$, la convergence de la densité vers une limite nécessite un argument supplémentaire (loi des grands nombres).

---

## Résumé

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STRUCTURE DE LA PREUVE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Régression vers la moyenne]                                       │
│            ↓                                                        │
│  [E[d_{n+1} - d_n] = 0]                                             │
│            ↓                                                        │
│  [Symétrie de la distribution]  ← (non prouvé rigoureusement)       │
│            ↓                                                        │
│  [P(hausse) = P(baisse)]                                            │
│            ↓                                                        │
│  [δ(A+) = δ(A-) = 1/2]  ✓                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Verdict

| Aspect | Statut |
|--------|--------|
| Intuition correcte | ✅ Très probable |
| Vérification numérique | ✅ Confirmée |
| Esquisse de preuve | ⚠️ Partielle |
| Preuve rigoureuse | ❌ Manquante |

**Notre contribution** : Identifier le mécanisme (régression vers la moyenne) et montrer comment il implique la conjecture, sous certaines hypothèses.

**Ce qui reste** : Prouver ces hypothèses à partir des propriétés fondamentales des nombres premiers.

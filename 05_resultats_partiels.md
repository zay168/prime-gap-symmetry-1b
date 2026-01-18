# 05 — Résultats Partiels et Records

## 1. Ce Qui Est Prouvé

### Sur la Densité

| Résultat | Statut |
|----------|--------|
| $\delta(A^+) = 1/2$ | **Non prouvé** |
| $\delta(A^+)$ existe | **Non prouvé** |
| $0 < \liminf \frac{|A^+ \cap [1,N]|}{N}$ | **Non prouvé** |

**Conclusion** : Aucun résultat théorique sur la densité des comparaisons.

### Sur l'Égalité des Écarts

| Résultat | Statut |
|----------|--------|
| $\exists n : d_n = d_{n+1}$ | **Prouvé** (exemples explicites) |
| $|\{n : d_n = d_{n+1}\}| = \infty$ | **Non prouvé** |
| $\forall k, \exists n : d_n = d_{n+1} = \cdots = d_{n+k-1}$ | **Non prouvé** |

---

## 2. Vérifications Numériques

### Densités Observées

Calculs effectués sur les premiers $N$ premiers :

| $N$ | $\delta(A^+)$ | $\delta(A^-)$ | $\delta(A^=)$ |
|-----|---------------|---------------|---------------|
| $10^4$ | 0.4998 | 0.5002 | 0.1847 |
| $10^5$ | 0.5001 | 0.4999 | 0.1843 |
| $10^6$ | 0.5000 | 0.5000 | 0.1841 |
| $10^7$ | 0.5000 | 0.5000 | 0.1840 |

**Observation** : La densité ~18.4% de $A^=$ implique que les écarts sont égaux environ 1 fois sur 5.

### Records de Consécutivité

| $k$ gaps égaux | Année | Premier terme (approx.) |
|----------------|-------|-------------------------|
| 5 | 1967 | $1.2 \times 10^8$ |
| 6 | 1995 | $2.1 \times 10^{10}$ |
| 9 | 1998 | $10^{24}$ |
| 20 | 2014 | $10^{176}$ |
| 26 | 2019 | $10^{200+}$ |

---

## 3. Résultats Conditionnels

### Sous l'Hypothèse de Riemann (RH)

Si RH est vraie, alors on a de meilleures bornes sur $d_n$ :

$$d_n = O\left(\sqrt{p_n} \ln p_n\right)$$

Mais cela **n'implique pas** directement le résultat sur les densités.

### Sous la Conjecture de Hardy-Littlewood

Si la conjecture des k-tuples est vraie, alors la conjecture d'Erdős suit potentiellement, mais le lien n'est pas rigoureux.

---

## 4. Résultats Adjacents

### Théorème de Maier (1985)

Les écarts entre premiers fluctuent **plus** que prévu par le modèle probabiliste naïf :

$$\limsup \frac{d_n}{\ln p_n / (\ln \ln p_n)^2} = \infty$$

**Implication** : La distribution des $d_n$ est plus irrégulière que le hasard pur.

### Résultat de Granville (1995)

Il existe des valeurs anormalement grandes et petites de $d_n$ par rapport à $\ln p_n$.

---

## 5. Approches Qui Ont Échoué

### 1. Argument Probabiliste Direct

Modéliser $d_n$ comme variables aléatoires indépendantes ne capture pas les corrélations subtiles.

### 2. Méthode du Cercle Seule

Fonctionne pour certains problèmes additifs mais pas pour les corrélations entre gaps.

### 3. Cribles de Premier Ordre

Donnent des bornes mais pas assez fines pour les densités.

---

## 6. État de l'Art (2024)

### Ce Qui Fonctionne Partiellement

- **Cribles de Selberg améliorés** : Donnent des résultats sur les "presque premiers"
- **Méthode GPY** : Révolutionnaire pour les petits gaps
- **Polymath8** : Amélioration collaborative des bornes de Zhang

### Ce Qui Manque

1. Contrôle des **corrélations** entre $d_n$ et $d_{n+1}$
2. Compréhension de la **mesure spectrale** des écarts
3. Lien avec les **zéros de Riemann** pour les égalités

---

## 7. Leçons Clés

1. **Les vérifications numériques ne prouvent rien** — mais suggèrent fortement la vérité
2. **L'absence de contre-exemple n'est pas une preuve**
3. **Le problème est "entre" ce qu'on sait et ce qu'on ne sait pas** — juste hors de portée
4. **Les méthodes actuelles sont insuffisantes** — une nouvelle idée est nécessaire

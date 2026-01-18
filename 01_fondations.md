# 01 — Fondations : Définitions et Notations

## 1. Nombres Premiers

Un **nombre premier** $p$ est un entier naturel $> 1$ divisible uniquement par 1 et lui-même.

$$\mathbb{P} = \{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, \ldots\}$$

On note $p_n$ le $n$-ème nombre premier :
- $p_1 = 2$
- $p_2 = 3$
- $p_3 = 5$
- $p_n \sim n \ln n$ (asymptotiquement)

---

## 2. Écarts entre Premiers (Prime Gaps)

### Définition Centrale

$$\boxed{d_n = p_{n+1} - p_n}$$

C'est l'**écart** (ou "gap") entre le $(n+1)$-ème et le $n$-ème premier.

### Exemples

| $n$ | $p_n$ | $p_{n+1}$ | $d_n = p_{n+1} - p_n$ |
|-----|-------|-----------|------------------------|
| 1   | 2     | 3         | 1                      |
| 2   | 3     | 5         | 2                      |
| 3   | 5     | 7         | 2                      |
| 4   | 7     | 11        | 4                      |
| 5   | 11    | 13        | 2                      |
| 6   | 13    | 17        | 4                      |

### Observation Clé
Pour $n \geq 2$, on a $d_n \geq 2$ (car deux premiers consécutifs > 2 diffèrent d'au moins 2).

---

## 3. Densité Naturelle

### Définition

Soit $A \subseteq \mathbb{N}$. La **densité naturelle** de $A$ est :

$$\delta(A) = \lim_{N \to \infty} \frac{|A \cap \{1, 2, \ldots, N\}|}{N}$$

si cette limite existe.

### Interprétation
- $\delta(A) = 1$ : $A$ contient "presque tous" les entiers
- $\delta(A) = 0$ : $A$ est "rare" (mais peut être infini !)
- $\delta(A) = 1/2$ : $A$ contient "la moitié" des entiers

### Exemples
- Nombres pairs : $\delta = 1/2$
- Nombres premiers : $\delta = 0$ (mais $|\mathbb{P}| = \infty$)
- Carrés parfaits : $\delta = 0$

---

## 4. Notation des Ensembles Clés

### Ensemble $A^+$ (Croissance)

$$A^+ = \{n \in \mathbb{N} : d_{n+1} \geq d_n\}$$

Les indices où l'écart **augmente ou reste stable**.

### Ensemble $A^-$ (Décroissance)

$$A^- = \{n \in \mathbb{N} : d_{n+1} \leq d_n\}$$

Les indices où l'écart **diminue ou reste stable**.

### Ensemble $A^=$ (Égalité)

$$A^= = \{n \in \mathbb{N} : d_{n+1} = d_n\}$$

Les indices où **deux écarts consécutifs sont égaux**.

---

## 5. La Question Centrale

Le problème ouvert affirme :

> **Conjecture** : $\delta(A^+) = \delta(A^-) = \frac{1}{2}$

Et de plus :

> **Conjecture** : $|A^=| = \infty$ (infinité de $n$ avec $d_{n+1} = d_n$)

---

## 6. Symboles Asymptotiques

| Notation | Signification |
|----------|---------------|
| $f \sim g$ | $\lim_{x \to \infty} f(x)/g(x) = 1$ |
| $f = O(g)$ | $|f(x)| \leq C \cdot g(x)$ pour $x$ grand |
| $f = o(g)$ | $\lim_{x \to \infty} f(x)/g(x) = 0$ |
| $f = \Omega(g)$ | $f(x) \geq c \cdot g(x)$ infiniment souvent |

---

## Points à Retenir

1. $d_n$ mesure l'irrégularité de la distribution des premiers
2. La densité naturelle quantifie "quelle proportion" d'entiers satisfait une propriété
3. Le problème demande si les écarts croissent/décroissent "équitablement"
4. L'existence d'une infinité d'égalités $d_{n+1} = d_n$ est non triviale

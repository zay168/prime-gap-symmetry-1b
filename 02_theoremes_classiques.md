# 02 — Théorèmes Classiques sur les Nombres Premiers

## 1. Théorème des Nombres Premiers (PNT)

### Énoncé

$$\pi(x) \sim \frac{x}{\ln x}$$

où $\pi(x) = |\{p \leq x : p \text{ premier}\}|$ compte les premiers jusqu'à $x$.

### Forme Équivalente

$$p_n \sim n \ln n$$

Le $n$-ème premier est approximativement $n \ln n$.

### Conséquence pour les Écarts

En moyenne, l'écart entre premiers près de $x$ est environ $\ln x$ :

$$\frac{1}{n} \sum_{k=1}^{n} d_k \sim \ln p_n$$

---

## 2. Bornes sur les Écarts

### Borne Triviale

$$d_n < p_n \quad \text{(évident)}$$

### Borne de Bertrand (1845)

Pour tout $n \geq 1$, il existe un premier entre $n$ et $2n$ :

$$d_n < p_n$$

### Borne de Cramér (Conjecturale, 1936)

$$d_n = O\left((\ln p_n)^2\right)$$

> **Statut** : Non prouvée ! C'est une conjecture majeure.

### Résultat de Baker-Harman-Pintz (2001)

$$d_n = O\left(p_n^{0.525}\right)$$

Meilleur résultat inconditionnel à ce jour.

---

## 3. Petits Écarts : Théorème GPY

### Goldston-Pintz-Yıldırım (2005)

$$\liminf_{n \to \infty} \frac{d_n}{\ln p_n} = 0$$

Les écarts peuvent être **arbitrairement petits** relativement à $\ln p_n$.

### Zhang (2013)

Il existe une infinité de $n$ avec :

$$d_n < 70\,000\,000$$

### Polymath (2014)

Amélioré à :

$$d_n \leq 246$$

infiniment souvent.

### Conjecture des Premiers Jumeaux

$$\liminf_{n \to \infty} d_n = 2$$

(Non prouvée)

---

## 4. Grands Écarts

### Résultat de Westzynthius (1931)

$$\limsup_{n \to \infty} \frac{d_n}{\ln p_n} = \infty$$

Les écarts peuvent être **arbitrairement grands** relativement à $\ln p_n$.

### Ford-Green-Konyagin-Maynard-Tao (2016)

$$d_n \geq c \cdot \frac{\ln p_n \cdot \ln \ln p_n \cdot \ln \ln \ln \ln p_n}{(\ln \ln \ln p_n)^2}$$

pour une infinité de $n$.

---

## 5. Théorème de Dirichlet (1837)

### Énoncé

Pour $a, q$ premiers entre eux, il existe une infinité de premiers de la forme :

$$p \equiv a \pmod{q}$$

### Pertinence

Ce théorème montre que les premiers sont "bien répartis" dans les progressions arithmétiques, ce qui est lié à la conjecture d'Erdős.

---

## 6. Premiers en Progression Arithmétique

### Théorème de Green-Tao (2004)

Les nombres premiers contiennent des **progressions arithmétiques de longueur arbitraire**.

$$\forall k, \exists a, d : \{a, a+d, a+2d, \ldots, a+(k-1)d\} \subset \mathbb{P}$$

### Lien avec Erdős

Si les premiers contiennent des PA arbitrairement longues, cela suggère (mais ne prouve pas) que les écarts consécutifs peuvent être égaux arbitrairement longtemps.

---

## 7. Table Récapitulative

| Résultat | année | Contenu |
|----------|-------|---------|
| PNT | 1896 | $\pi(x) \sim x/\ln x$ |
| Bertrand | 1845 | Premier entre $n$ et $2n$ |
| Cramér (conj.) | 1936 | $d_n = O((\ln p_n)^2)$ |
| GPY | 2005 | $\liminf d_n/\ln p_n = 0$ |
| Zhang | 2013 | $d_n < 70M$ infiniment souvent |
| Polymath | 2014 | $d_n \leq 246$ i.s. |
| Green-Tao | 2004 | PA arbitraires dans $\mathbb{P}$ |

---

## 8. Pourquoi C'est Difficile

Les nombres premiers sont définis **multiplicativement** (divisibilité) mais les écarts sont une propriété **additive**. Cette tension fondamentale rend l'étude des $d_n$ extrêmement difficile.

### Outils Principaux
- **Cribles** (Selberg, GPY)
- **Fonctions L de Dirichlet**
- **Méthode du cercle de Hardy-Littlewood**
- **Hypothèse de Riemann** (donne des bornes plus fines si vraie)

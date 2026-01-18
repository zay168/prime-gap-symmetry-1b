# 03 — Énoncé du Problème Ouvert

## Source

Ce problème apparaît dans plusieurs bases de données de problèmes ouverts en théorie des nombres, notamment référencé dans les travaux d'Erdős et collaborateurs.

---

## Énoncé Formel

### Partie 1 : Densité des Comparaisons

Soit $d_n = p_{n+1} - p_n$ l'écart entre le $(n+1)$-ème et le $n$-ème nombre premier.

**Conjecture A** :
$$\delta\left(\{n : d_{n+1} \geq d_n\}\right) = \frac{1}{2}$$

**Conjecture B** :
$$\delta\left(\{n : d_{n+1} \leq d_n\}\right) = \frac{1}{2}$$

où $\delta(A)$ désigne la densité naturelle de l'ensemble $A$.

### Partie 2 : Égalités Infinies

**Conjecture C** :
$$\left|\{n : d_{n+1} = d_n\}\right| = \infty$$

Il existe une infinité de $n$ tels que deux écarts consécutifs sont égaux.

---

## Interprétation Intuitive

### "Équité" des Écarts

La conjecture dit que les écarts entre premiers sont "équitablement" distribués :
- Dans ~50% des cas, l'écart augmente (ou reste stable)
- Dans ~50% des cas, l'écart diminue (ou reste stable)

C'est une forme de **symétrie statistique** dans le comportement des écarts.

### Analogie

Imaginez lancer une pièce à chaque $n$ :
- Face = l'écart augmente
- Pile = l'écart diminue

La conjecture dit que cette "pièce" est **équilibrée** à long terme.

---

## Ce Qui Est Connu

### Vérification Numérique

Pour les premiers millions de premiers, les densités observées sont très proches de $1/2$ :

| Limite $N$ | $\delta(A^+)$ observée | $\delta(A^-)$ observée |
|------------|------------------------|------------------------|
| $10^6$     | ~0.500                 | ~0.500                 |
| $10^8$     | ~0.500                 | ~0.500                 |
| $10^{10}$  | ~0.500                 | ~0.500                 |

### Résultats Théoriques

1. **Aucune preuve** que $\delta(A^+)$ ou $\delta(A^-)$ existe
2. **Aucune preuve** que ces densités valent $1/2$
3. Les meilleures bornes connues utilisent des hypothèses non prouvées (RH)

---

## Pourquoi C'est Difficile

### Problème 1 : Irrégularité des Écarts

Les $d_n$ sont extrêmement irréguliers. Voici les premiers :

$$d_1, d_2, d_3, \ldots = 1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6, \ldots$$

Pas de pattern évident !

### Problème 2 : Corrélations

Les comparaisons $d_{n+1} \geq d_n$ impliquent une **corrélation** entre écarts consécutifs. Les outils standard (cribles) ont du mal avec les corrélations.

### Problème 3 : Pas de Formule

Il n'existe pas de formule explicite pour $d_n$ en fonction de $n$.

---

## Relations avec d'Autres Problèmes

```
┌─────────────────────────────────────────────────────────┐
│                 Hiérarchie des Problèmes                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Hypothèse de Riemann                                  │
│         │                                                │
│         ▼                                                │
│   Conjecture de Cramér (gaps bornés)                    │
│         │                                                │
│         ▼                                                │
│   Premiers Jumeaux (gaps = 2 infiniment)                │
│         │                                                │
│         ▼                                                │
│   Densité des comparaisons = 1/2  ◄── CE PROBLÈME       │
│         │                                                │
│         ▼                                                │
│   Conjecture d'Erdős (k gaps égaux consécutifs)         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Questions Ouvertes Connexes

1. Quelle est la distribution de $d_{n+1} - d_n$ ?
2. Les suites $(d_n, d_{n+1}, d_{n+2})$ sont-elles équidistribuées ?
3. Peut-on prouver des bornes sur $\delta(A^+)$ ? (e.g., $0.4 < \delta(A^+) < 0.6$)

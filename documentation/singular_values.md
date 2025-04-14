# Les Valeurs Singulières

## Définition Mathématique

Les valeurs singulières sont des concepts fondamentaux de l'algèbre linéaire, étroitement liées à la décomposition en valeurs singulières (SVD). 

Pour une matrice A de dimension m×n, les valeurs singulières sont les racines carrées des valeurs propres non nulles de A^T·A (ou de façon équivalente, de A·A^T).

Mathématiquement, si on a :
- A^T·A = V·Σ²·V^T (où Σ² est une matrice diagonale contenant les valeurs propres)
- Alors les valeurs singulières σᵢ = √λᵢ, où λᵢ sont les valeurs propres de A^T·A.

## Décomposition en Valeurs Singulières (SVD)

La SVD décompose une matrice A en trois matrices :

A = U·Σ·V^T

Où :
- U est une matrice orthogonale m×m (les vecteurs singuliers gauches)
- Σ est une matrice diagonale m×n contenant les valeurs singulières en ordre décroissant
- V^T est la transposée d'une matrice orthogonale n×n (les vecteurs singuliers droits)

## Interprétation Géométrique

Les valeurs singulières représentent les facteurs d'étirement ou de compression dans différentes directions lors de la transformation linéaire représentée par la matrice. Plus précisément :
- La direction de l'étirement est donnée par les vecteurs singuliers
- L'amplitude de l'étirement est donnée par les valeurs singulières

Une valeur singulière élevée indique une direction où les données ont une variance élevée.

## Applications en Machine Learning

### 1. Réduction de Dimensionnalité (PCA)

L'Analyse en Composantes Principales (PCA) utilise la SVD pour :
- Identifier les directions de variance maximale
- Réduire la dimensionnalité en conservant uniquement les composantes associées aux plus grandes valeurs singulières

### 2. Compression de Données

Les valeurs singulières permettent d'approximer une matrice en ne gardant que les k plus grandes valeurs singulières :

A ≈ U_k·Σ_k·V_k^T

Cette approximation de rang k minimise l'erreur d'approximation (selon la norme de Frobenius).

### 3. Systèmes de Recommandation

Dans des applications comme le filtrage collaboratif, la SVD permet d'approximer la matrice utilisateur-item pour prédire les préférences.

### 4. Analyse Numérique

Les valeurs singulières aident à déterminer :
- Le conditionnement d'une matrice (ratio entre la plus grande et la plus petite valeur singulière)
- Le rang numérique (nombre de valeurs singulières significativement non nulles)

### 5. Résolution de Systèmes Linéaires Mal Conditionnés

La SVD permet de résoudre des systèmes d'équations linéaires même quand la matrice est presque singulière, en utilisant la pseudo-inverse.

## Relation avec Nos Algorithmes

Dans notre implémentation de LLE (Locally Linear Embedding), nous travaillons avec des valeurs propres plutôt que des valeurs singulières directement, mais le principe est similaire : 
1. Nous calculons une matrice de reconstruction M
2. Nous extrayons les vecteurs propres associés aux plus petites valeurs propres (non nulles)
3. Ces vecteurs propres définissent l'espace de dimension réduite

## Conclusion

Les valeurs singulières sont des outils fondamentaux en algèbre linéaire qui permettent d'analyser et de transformer des données multidimensionnelles. Leur compréhension est essentielle pour maîtriser de nombreux algorithmes de machine learning, notamment ceux liés à la réduction de dimensionnalité et au traitement de matrices.

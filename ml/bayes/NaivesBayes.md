# Naive Bayes en Machine Learning

## Introduction
Le classificateur Naive Bayes est un algorithme de machine learning probabiliste basé sur le théorème de Bayes. Il est particulièrement efficace pour la classification de textes et le filtrage de spam.

## Principe fondamental
L'algorithme repose sur le théorème de Bayes :
```
P(A|B) = P(B|A) * P(A) / P(B)
```
Où :
- P(A|B) est la probabilité postérieure
- P(B|A) est la vraisemblance
- P(A) est la probabilité a priori
- P(B) est la preuve

## Caractéristiques principales

1. **"Naïf"** : L'algorithme suppose que toutes les caractéristiques sont indépendantes les unes des autres
2. **Rapide** : Simple à implémenter et efficace en calcul
3. **Performant** : Fonctionne bien même avec peu de données d'entraînement

## Applications courantes

- Classification de textes
- Filtrage de spam
- Analyse de sentiments
- Systèmes de recommandation

## Avantages et inconvénients

### Avantages
- Simple et rapide
- Performant avec les petits jeux de données
- Gère bien les données catégorielles

### Inconvénients
- Hypothèse d'indépendance souvent irréaliste
- Sensible aux caractéristiques non pertinentes

## Types de Naive Bayes

1. **Gaussian Naive Bayes** : Pour les données continues
2. **Multinomial Naive Bayes** : Pour les données discrètes
3. **Bernoulli Naive Bayes** : Pour les données binaires

## Conclusion
Malgré sa simplicité, Naive Bayes reste un algorithme puissant et largement utilisé en machine learning, particulièrement efficace pour la classification de textes.
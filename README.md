# Ft_linear_regression

> Projet d'introduction au Machine Learning : prédiction du prix d'une voiture par régression linéaire

[![Author](https://img.shields.io/badge/Author-Cimeci-blue)](https://github.com/Cimeci)

---

## Table des matières

- [À propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Fonctionnement](#-fonctionnement)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)

---

## À propos

Ce projet représente mes premiers pas dans l'intelligence artificielle et plus précisément le Machine Learning. Il implémente un algorithme de **régression linéaire** utilisant la **descente de gradient** pour prédire le prix d'une voiture en fonction de son kilométrage.

### Objectifs pédagogiques

- Comprendre les fondamentaux de la régression linéaire
- Implémenter l'algorithme de descente de gradient
- Maîtriser la normalisation des données
- Visualiser les résultats et la convergence du modèle

---

## Fonctionnalités

- **Lecture et traitement des données** depuis un fichier CSV
- **Entraînement du modèle** avec l'algorithme de descente de gradient
- **Prédiction du prix** d'une voiture selon son kilométrage
- **Visualisation graphique** :
  - Droite de régression sur les données
  - Évolution de la fonction de coût
  - Analyse des résidus
- **Sauvegarde des paramètres** (theta0, theta1) pour réutilisation

---

## Prérequis

- Python 3.6 ou supérieur
- pip (gestionnaire de paquets Python)

---

## Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/Cimeci/Ft_linear_regression.git
cd Ft_linear_regression
```

2. Installez les dépendances :

```bash
pip install pandas matplotlib
```

---

## Utilisation

### 1. Entraîner le modèle

Lancez l'entraînement sur les données du fichier `data.csv` :

```bash
python train.py
```

Cette commande va :
- Charger les données
- Normaliser les features (kilométrage)
- Entraîner le modèle par descente de gradient
- Afficher des graphiques de visualisation
- Sauvegarder les paramètres (theta0, theta1) dans `theta.csv`

### 2. Prédire un prix

Une fois le modèle entraîné, utilisez-le pour prédire le prix d'une voiture :

```bash
python predict.py
```

L'application vous demandera de saisir un kilométrage et vous donnera le prix estimé.

**Exemple :**
```
Enter the mileage: 50000
Estimated price: 7234.56
```

---

## Fonctionnement

### Régression linéaire

Le modèle cherche à trouver la meilleure droite $y = \theta_0 + \theta_1 \cdot x$ où :
- $x$ : kilométrage (feature)
- $y$ : prix estimé (target)
- $\theta_0$ : ordonnée à l'origine (intercept)
- $\theta_1$ : pente (coefficient)

### Descente de gradient

L'algorithme optimise les paramètres en minimisant la fonction de coût (MSE - Mean Squared Error) :

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Les paramètres sont mis à jour itérativement :

$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

où $\alpha$ est le taux d'apprentissage (learning rate).

### Normalisation

Pour améliorer la convergence, les données sont normalisées :

$$x_{norm} = \frac{x - \mu}{\sigma}$$

---

## Structure du projet

```
Ft_linear_regression/
│
├── data.csv                    # Données d'entraînement (km, prix)
├── theta.csv                   # Paramètres sauvegardés après entraînement
├── train.py                    # Script d'entraînement du modèle
├── predict.py                  # Script de prédiction
├── ft_linear_regression.py     # Module principal (si applicable)
└── README.md                   # Documentation
```

---

## Auteur

**Cimeci**

- GitHub: [@Cimeci](https://github.com/Cimeci)
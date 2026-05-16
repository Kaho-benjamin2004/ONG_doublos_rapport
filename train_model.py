import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Données d'exemple (pour démo, mais on peut charger depuis CSV)
rapports = [
    {"titre": "Panne du forage à Koundi", "description": "Le forage du village de Koundi ne fonctionne plus depuis 3 jours."},
    {"titre": "École sans électricité", "description": "Les classes de l'école primaire de Bamba n'ont pas d'électricité."},
    {"titre": "Problème de forage similaire", "description": "Le forage situé à Koundi est en panne."},
    {"titre": "Besoin de fournitures scolaires", "description": "Les élèves manquent de cahiers et de stylos à l'école de Bamba."},
    {"titre": "Urgence médicale à Koundi", "description": "Une épidémie de paludisme frappe le village de Koundi."},
]

textes = [r["titre"] + " " + r["description"] for r in rapports]

# Vectorisation
vectorizer = TfidfVectorizer(max_features=100, stop_words=None)  # pas de stop words
X = vectorizer.fit_transform(textes)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Sauvegarder les vecteurs et métadonnées pour la recherche de similarité
rapports_vectors = X  # matrice sparse
rapports_meta = [(r["titre"], r["description"]) for r in rapports]

os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("models/rapports_vectors.pkl", "wb") as f:
    pickle.dump(rapports_vectors, f)
with open("models/rapports_meta.pkl", "wb") as f:
    pickle.dump(rapports_meta, f)

print(f"✅ Modèle entraîné avec {len(textes)} rapports. Vecteurs sauvegardés.")
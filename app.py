from fastapi import FastAPI, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from database import save_report, get_all_reports, SessionLocal
from auth import verify_api_key
import pandas as pd

load_dotenv()

app = FastAPI(title="ONG Report Similarity API", version="1.0")

# ---------- Chargement des modèles ----------
MODEL_PATH = "models/"
if not all(os.path.exists(MODEL_PATH + f) for f in ["vectorizer.pkl", "kmeans.pkl", "rapports_vectors.pkl", "rapports_meta.pkl"]):
    raise RuntimeError("Modèles introuvables. Lancez d'abord python train_model.py")

with open(MODEL_PATH + "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_PATH + "kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open(MODEL_PATH + "rapports_vectors.pkl", "rb") as f:
    rapports_vectors = pickle.load(f)
with open(MODEL_PATH + "rapports_meta.pkl", "rb") as f:
    rapports_meta = pickle.load(f)

# ---------- Fonction de recherche de similarité ----------
def find_similar_reports(query_vector, k=3):
    """Retourne les k rapports les plus similaires (titre, description, score)"""
    similarities = cosine_similarity(query_vector, rapports_vectors)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        titre, desc = rapports_meta[idx]
        score = float(similarities[idx])
        results.append({"titre": titre, "description": desc, "similarite": round(score, 3)})
    return results

# ---------- Pages HTML (design Tailwind) ----------
HTML_FORM = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONG Horizon - Détection de rapports similaires</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'ong-blue': '#1e6f5c',
                        'ong-teal': '#289672',
                    }
                }
            }
        }
    </script>
    <style>
        body { background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); }
    </style>
</head>
<body class="font-sans antialiased">
    <div class="min-h-screen flex items-center justify-center px-4 py-12">
        <div class="max-w-2xl w-full bg-white rounded-2xl shadow-xl overflow-hidden">
            <div class="bg-gradient-to-r from-ong-blue to-ong-teal px-6 py-8 text-center">
                <i class="bi bi-file-earmark-text-fill text-white text-4xl"></i>
                <h1 class="text-2xl md:text-3xl font-bold text-white">Analyse de rapports terrain</h1>
                <p class="text-white/80 mt-2">Regroupement automatique pour éviter les doublons</p>
            </div>
            <form action="/predict" method="post" class="p-6 md:p-8 space-y-6">
                <div>
                    <label class="block text-gray-700 font-semibold mb-2"><i class="bi bi-tag-fill text-ong-blue mr-1"></i>Titre</label>
                    <input type="text" name="titre" required class="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-ong-blue">
                </div>
                <div>
                    <label class="block text-gray-700 font-semibold mb-2"><i class="bi bi-chat-text-fill text-ong-blue mr-1"></i>Description</label>
                    <textarea name="description" rows="5" required class="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-ong-blue resize-none"></textarea>
                </div>
                <button type="submit" class="w-full bg-gradient-to-r from-ong-blue to-ong-teal hover:from-ong-teal hover:to-ong-blue text-white font-bold py-3 rounded-xl transition flex items-center justify-center gap-2">
                    <i class="bi bi-graph-up-arrow"></i> Analyser
                </button>
            </form>
            <div class="bg-gray-50 px-6 py-3 text-center text-xs text-gray-500 border-t">
                <i class="bi bi-shield-check"></i> Modèle non supervisé – Aide à la décision
            </div>
        </div>
    </div>
</body>
</html>
"""

def get_result_html(titre, description, cluster, similaires):
    similaires_html = ""
    for s in similaires:
        similaires_html += f"""
        <div class="border-l-4 border-ong-blue bg-gray-50 p-3 rounded-r-lg mb-2">
            <p class="font-semibold">{s['titre']}</p>
            <p class="text-sm text-gray-600">{s['description'][:100]}...</p>
            <p class="text-xs text-ong-teal mt-1">Similarité : {s['similarite']}</p>
        </div>
        """
    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Résultat - ONG Horizon</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        'ong-blue': '#1e6f5c',
                        'ong-teal': '#289672',
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{ background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); }}
    </style>
</head>
<body class="font-sans antialiased">
    <div class="min-h-screen flex items-center justify-center px-4 py-12">
        <div class="max-w-3xl w-full bg-white rounded-2xl shadow-xl overflow-hidden">
            <div class="bg-gradient-to-r from-ong-blue to-ong-teal px-6 py-6 text-center">
                <i class="bi bi-check-circle-fill text-white text-4xl"></i>
                <h1 class="text-2xl font-bold text-white mt-2">Analyse terminée</h1>
            </div>
            <div class="p-6 md:p-8 space-y-5">
                <div class="bg-ong-light/50 rounded-xl p-4">
                    <h2 class="font-semibold text-ong-blue"><i class="bi bi-card-heading"></i> Titre</h2>
                    <p class="text-gray-800 mt-1 ml-6">{titre}</p>
                </div>
                <div class="bg-ong-light/50 rounded-xl p-4">
                    <h2 class="font-semibold text-ong-blue"><i class="bi bi-text-paragraph"></i> Description</h2>
                    <p class="text-gray-800 mt-1 ml-6">{description}</p>
                </div>
                <div class="bg-gradient-to-r from-blue-50 to-teal-50 rounded-xl p-4 border border-ong-blue/20">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-500">Cluster attribué</span>
                        <span class="text-3xl font-bold text-ong-blue">#{cluster}</span>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold text-gray-700"><i class="bi bi-diagram-3"></i> Rapports similaires existants</h3>
                    {similaires_html if similaires_html else "<p class='text-sm text-gray-500'>Aucun rapport similaire trouvé.</p>"}
                </div>
                <div class="flex justify-between pt-4">
                    <a href="/" class="inline-flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2 px-5 rounded-full transition">
                        <i class="bi bi-arrow-left"></i> Nouveau rapport
                    </a>
                    <button onclick="window.location.reload();" class="inline-flex items-center gap-2 bg-ong-blue hover:bg-ong-teal text-white py-2 px-5 rounded-full transition">
                        <i class="bi bi-arrow-repeat"></i> Re-analyser
                    </button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """

# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return HTMLResponse(content=HTML_FORM)

@app.post("/predict", response_class=HTMLResponse)
async def predict(titre: str = Form(...), description: str = Form(...)):
    texte = titre + " " + description
    X = vectorizer.transform([texte])
    cluster = int(kmeans.predict(X)[0])
    # Sauvegarde en base
    save_report(titre, description, cluster)
    # Recherche de similaires
    similaires = find_similar_reports(X, k=3)
    return HTMLResponse(content=get_result_html(titre, description, cluster, similaires))

# API JSON sécurisée pour intégration mobile
@app.post("/api/report", dependencies=[Depends(verify_api_key)])
async def api_report(titre: str, description: str):
    texte = titre + " " + description
    X = vectorizer.transform([texte])
    cluster = int(kmeans.predict(X)[0])
    save_report(titre, description, cluster)
    similaires = find_similar_reports(X, k=3)
    return JSONResponse(content={
        "cluster": cluster,
        "similaires": similaires,
        "message": "Rapport analysé et sauvegardé"
    })

@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def stats():
    reports = get_all_reports()
    df = pd.DataFrame([(r.cluster,) for r in reports], columns=["cluster"])
    cluster_counts = df["cluster"].value_counts().to_dict()
    return JSONResponse({
        "total_reports": len(reports),
        "repartition_clusters": cluster_counts,
        "message": "Utilisez /retrain pour réentraîner le modèle"
    })

# Endpoint de réentraînement (nécessite un script externe, sécurité)
@app.post("/retrain", dependencies=[Depends(verify_api_key)])
async def retrain():
    import subprocess
    try:
        subprocess.run(["python", "train_model.py"], check=True, capture_output=True)
        # Recharger les nouveaux modèles (pour l'API)
        global vectorizer, kmeans, rapports_vectors, rapports_meta
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)
        with open("models/rapports_vectors.pkl", "rb") as f:
            rapports_vectors = pickle.load(f)
        with open("models/rapports_meta.pkl", "rb") as f:
            rapports_meta = pickle.load(f)
        return JSONResponse({"status": "success", "message": "Modèle réentraîné avec succès"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
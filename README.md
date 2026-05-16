# ONG Report Similarity Tool

Outil de détection de doublons de rapports terrain par clustering non supervisé.

## Installation rapide (Docker)
```bash
docker build -t ong-similarity .
docker run -p 8000:8000 --env-file .env ong-similarity
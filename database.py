import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reports.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Report(Base):
    __tablename__ = "rapports"
    id = Column(Integer, primary_key=True, index=True)
    titre = Column(String, nullable=False)
    description = Column(String, nullable=False)
    cluster = Column(Integer)
    date_creation = Column(DateTime, default=datetime.utcnow)
    est_doublon = Column(Boolean, default=False)  # signalé par utilisateur

Base.metadata.create_all(bind=engine)

def save_report(titre: str, description: str, cluster: int):
    db = SessionLocal()
    report = Report(titre=titre, description=description, cluster=cluster)
    db.add(report)
    db.commit()
    db.refresh(report)
    db.close()
    return report

def get_all_reports():
    db = SessionLocal()
    reports = db.query(Report).all()
    db.close()
    return reports
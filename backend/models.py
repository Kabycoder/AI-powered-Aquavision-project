from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)

class WaterAnalysis(Base):
    __tablename__ = "water_analysis"

    id = Column(Integer, primary_key=True, index=True)
    original_image = Column(String)
    segmented_image = Column(String)
    area = Column(Float)
    volume = Column(Float)
    location = Column(String)
    date = Column(DateTime)
    alert = Column(String)
    pdf_file = Column(String)

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer)
    file_path = Column(String)
    created_at = Column(DateTime)
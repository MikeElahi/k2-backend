import datetime

from app import db
from sqlalchemy import Column, Integer, Text, DateTime

class Entry(db.Model):
    id = Column(Integer, primary_key=True)
    uuid = Column(Text)  # UUID, Used to categorize entries
    image = Column(Text)
    segments = Column(Text)
    percentage = Column(Integer, nullable=True)
    most_significant_detection = Column(Text)
    most_significant_area = Column(Integer)
    date_created = Column(DateTime, default=datetime.datetime.utcnow)

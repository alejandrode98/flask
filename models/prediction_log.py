from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Text
from datetime import datetime

class Base(DeclarativeBase):
    pass

class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[str] = mapped_column(String(10))
    spam_proba: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


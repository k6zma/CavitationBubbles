from pydantic import BaseModel
from decimal import Decimal


class Predict(BaseModel):
    class_name: str

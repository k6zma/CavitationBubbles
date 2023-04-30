from pydantic import BaseModel
from decimal import Decimal


class Predict(BaseModel):
    water_0: Decimal
    alcohol_5: Decimal
    alcohol_12_5: Decimal
    alcohol_25: Decimal
    alcohol_50: Decimal
    alcohol_75: Decimal
    alcohol_96: Decimal

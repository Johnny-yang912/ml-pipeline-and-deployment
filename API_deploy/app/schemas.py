from typing import List
from pydantic import BaseModel, Field
from pydantic import ConfigDict  # pydantic v2

class CustomerIn(BaseModel):
    # 允許用原始欄名（含空白/括號）
    model_config = ConfigDict(populate_by_name=True)

    Age: float
    Purchase_Amount_USD: float = Field(..., alias="Purchase Amount (USD)")
    Review_Rating: float = Field(..., alias="Review Rating")
    Previous_Purchases: float = Field(..., alias="Previous Purchases")
    Gender: str
    Location: str
    Payment_Method: str = Field(..., alias="Payment Method")
    Frequency_of_Purchases: str = Field(..., alias="Frequency of Purchases")

class BatchIn(BaseModel):
    items: List[CustomerIn]

class PredictOut(BaseModel):
    probability: float
    prediction: int
    threshold: float

class BatchOut(BaseModel):
    results: List[PredictOut]

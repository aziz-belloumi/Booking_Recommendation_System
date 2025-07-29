from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class RecommendRequest(BaseModel):
    user_id: int
    purpose: str
    attendees: int
    target_date: datetime
    target_hours: List[int]
    top_k: Optional[int] = 10

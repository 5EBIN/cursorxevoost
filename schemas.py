from typing import Optional

from pydantic import BaseModel


class LandReq(BaseModel):
    district: Optional[str] = None
    status: Optional[str] = "vacant"
    top_n: int = 10


class MatchReq(BaseModel):
    investor_id: str
    top_n: int = 5


class CopilotReq(BaseModel):
    question: str

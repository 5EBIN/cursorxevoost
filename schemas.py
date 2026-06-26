from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LandReq(BaseModel):
    district: Optional[str] = None
    status: Optional[str] = "vacant"
    top_n: int = 10


class MatchReq(BaseModel):
    investor_id: str
    top_n: int = 5


class CopilotReq(BaseModel):
    question: str


class SearchReq(BaseModel):
    query: str
    district: Optional[str] = None
    k: int = 8


class ReportReq(BaseModel):
    report_type: str  # district | land | investment
    params: Dict[str, Any] = Field(default_factory=dict)


class QueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = 5  # capped at 5 (ProfSidekick injects max 5)
    min_score: float = 0.0  # advisory only — our scores aren't all cosine (KB-ADAPTER C3)

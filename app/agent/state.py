from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    question: str
    answer: str
    documents: Annotated[list[str], add]
    retries: int

class InitialState(TypedDict):
    question: str

class MiddleState(TypedDict):
    question: str
    retries: int
    documents: Annotated[list[str], add]

class OutputState(TypedDict):
    answer: str

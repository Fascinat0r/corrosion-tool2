# models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class NodeType(Enum):
    UNKNOWN = -1
    INPUT = 0  # Входной узел (источник)
    OUTPUT = 1  # Выходной узел (сброс или конечный пункт)
    CONSUMER = 2  # Узел-потребитель, обычно outDeg=0 (в тупиковой схеме) или (in=1,out=1) в некоторых схемах
    BRANCH = 3  # Узел-вставка (in=1, out>1) - расход делится
    MERGE = 4  # Узел-слияние (in>1, out=1)
    DIVIDER = 5  # Водораздел или смешанное узел (in>1, out>1) - т.е. «сложное» объединение+разделение

    # (иногда MERGE+DIVERSE, но для учебных задач можно назвать DIVIDER)

    @property
    def color(self) -> str:
        colors = {
            NodeType.UNKNOWN: "gray",
            NodeType.INPUT: "green",
            NodeType.OUTPUT: "red",
            NodeType.CONSUMER: "blue",
            NodeType.BRANCH: "orange",
            NodeType.MERGE: "purple",
            NodeType.DIVIDER: "pink",
        }
        return colors.get(self, "gray")

    @property
    def title(self) -> str:
        titles = {
            NodeType.UNKNOWN: "Неизвестный",
            NodeType.INPUT: "Источник (Input)",
            NodeType.OUTPUT: "Выход (Output)",
            NodeType.CONSUMER: "Потребитель",
            NodeType.BRANCH: "Ветвление (Branch)",
            NodeType.MERGE: "Слияние (Merge)",
            NodeType.DIVIDER: "Водораздел (Divider)"
        }
        return titles.get(self, "Неизвестный")


@dataclass
class PipelineNode:
    id: str
    type: NodeType
    x: float = 0.0
    y: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEdge:
    source: str
    target: str
    length: float = 0.0
    diameter: float = 0.5
    roughness: float = 0.000015
    properties: Dict[str, Any] = field(default_factory=dict)

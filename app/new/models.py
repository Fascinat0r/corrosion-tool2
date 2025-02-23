# models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

class NodeType(Enum):
    UNKNOWN = -1
    INPUT = 0         # Входной узел (источник)
    CONSUMER = 1      # Потребитель
    BRANCH = 2        # Ветвление
    DIVIDER = 3       # Водораздел (точка слияния потоков)
    OUTPUT = 4        # Выходной узел

    @property
    def color(self) -> str:
        colors = {
            NodeType.UNKNOWN: "gray",
            NodeType.INPUT: "green",
            NodeType.CONSUMER: "blue",
            NodeType.BRANCH: "orange",
            NodeType.DIVIDER: "purple",
            NodeType.OUTPUT: "red"
        }
        return colors.get(self, "gray")

    @property
    def title(self) -> str:
        titles = {
            NodeType.UNKNOWN: "Неизвестный",
            NodeType.INPUT: "Водозабор",
            NodeType.CONSUMER: "Потребитель",
            NodeType.BRANCH: "Ветвление",
            NodeType.DIVIDER: "Водораздел",
            NodeType.OUTPUT: "Выход"
        }
        return titles.get(self, "Неизвестный")


@dataclass
class PipelineNode:
    """
    Узел трубопроводной сети (бизнес-логика), не зависящий от networkx.
    """
    id: str
    type: NodeType
    x: float = 0.0
    y: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEdge:
    """
    Труба (ребро) с физическими/дополнительными свойствами.
    """
    source: str
    target: str
    length: float = 0.0
    diameter: float = 0.5
    roughness: float = 0.000015
    # любые другие поля
    properties: Dict[str, Any] = field(default_factory=dict)

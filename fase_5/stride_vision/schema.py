"""Tipos compartilhados: nomes das 32 classes de componentes e as estruturas
usadas para representar deteccoes (bbox, componente, fluxo) ao longo do pipeline.

Os nomes/ids das classes devem bater exatamente com
dataset/stride-architecture-components-v1/data.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

COMPONENT_CLASSES: dict[int, str] = {
    0: "actor_user", 1: "actor_admin", 2: "edge_ddos_protection", 3: "edge_cdn",
    4: "edge_waf", 5: "edge_gateway", 6: "edge_portal", 7: "external_entry_point",
    8: "integration_orchestrator", 9: "integration_messaging", 10: "compute_load_balancer",
    11: "compute_service", 12: "compute_worker", 13: "data_database", 14: "data_cache",
    15: "data_storage", 16: "security_identity_provider", 17: "security_key_management",
    18: "obs_monitoring", 19: "obs_audit", 20: "external_backend_service",
    21: "external_saas_service", 22: "external_web_service", 23: "communication_service",
    24: "backup_service", 25: "boundary_cloud", 26: "boundary_region",
    27: "boundary_resource_group", 28: "boundary_vpc_or_vnet", 29: "boundary_subnet_public",
    30: "boundary_subnet_private", 31: "boundary_autoscaling_group",
}

BOUNDARY_CLASSES = {name for name in COMPONENT_CLASSES.values() if name.startswith("boundary_")}
FLOW_CLASS = "flow_arrow"


@dataclass(frozen=True)
class BBox:
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def x0(self) -> float:
        return self.x_center - self.width / 2

    @property
    def y0(self) -> float:
        return self.y_center - self.height / 2

    @property
    def x1(self) -> float:
        return self.x_center + self.width / 2

    @property
    def y1(self) -> float:
        return self.y_center + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class Component:
    id: int
    cls_name: str
    bbox: BBox
    confidence: float = 1.0

    @property
    def is_boundary(self) -> bool:
        return self.cls_name in BOUNDARY_CLASSES


@dataclass(frozen=True)
class Flow:
    id: int
    bbox: BBox
    tail: tuple[float, float] | None
    tip: tuple[float, float] | None
    confidence: float = 1.0

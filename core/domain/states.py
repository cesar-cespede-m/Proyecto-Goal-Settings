"""Estados persistidos y terminales válidos del workflow."""

from enum import StrEnum


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITHOUT_RECOMMENDATION = "COMPLETED_WITHOUT_RECOMMENDATION"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ReportStatus(StrEnum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class ApprovalStage(StrEnum):
    TRAINING = "TRAINING"
    PUBLICATION = "PUBLICATION"


class ApprovalDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

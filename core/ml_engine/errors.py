"""Errores accionables del motor ML tradicional."""


class MLValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CandidateTrainingError(RuntimeError):
    def __init__(self, candidate: str, message: str) -> None:
        super().__init__(message)
        self.candidate = candidate

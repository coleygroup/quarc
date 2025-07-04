class BadRecord(Exception):
    pass


class SmilesError(BadRecord):
    """Raised when a component has no smiles."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data

    pass


class QuantityError(Exception):
    """Raised when there's an issue with processing quantities."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data


class InvalidDensityError(QuantityError):
    pass


class InvalidQuantError(QuantityError):
    pass


class InvalidSolventError(QuantityError):
    pass


class ProcessingError(Exception):
    pass


class ForwardMismatchError(ProcessingError):
    """Raised when encounter any mismatch in forward matching, that any component (intermed_amount) is not found
    as subsets of rxn_smi sets. This should be fine because quenching solvents or other stuff might got lumped in components.
    """

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data


class ReverseMismatchError(ProcessingError):
    """Raised when encounter any mismatch in reverse checking, when any part of the parsed set is not covered by components
    Should be flagged and throw away.
    """

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data

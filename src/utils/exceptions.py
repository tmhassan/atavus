"""
Custom exceptions for genome analysis.
"""

class GenomeAnalysisError(Exception):
    """Base exception for genome analysis errors"""
    pass

class GenomeParsingError(GenomeAnalysisError):
    """Raised when genome file parsing fails"""
    pass

class InvalidFileFormatError(GenomeAnalysisError):
    """Raised when file format is invalid"""
    pass

class InsufficientDataError(GenomeAnalysisError):
    """Raised when insufficient genetic data is available"""
    pass

class ReferenceDataError(GenomeAnalysisError):
    """Raised when reference data is missing or invalid"""
    pass

class AnalysisError(GenomeAnalysisError):
    """Raised when analysis computation fails"""
    pass

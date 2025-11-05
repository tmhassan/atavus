"""
Core genome analysis modules for advanced population genetics analysis.
"""

from .genome_parser import AdvancedGenomeParser, SNPRecord
from .population_analyzer import ProductionPopulationAnalyzer, PopulationResult
from .reference_data_manager import ReferenceDataManager, ReferencePopulation
from .statistical_engine import AdvancedStatisticalEngine, StatisticalResults

__all__ = [
    'AdvancedGenomeParser',
    'SNPRecord',
    'ProductionPopulationAnalyzer',
    'PopulationResult',
    'ReferenceDataManager',
    'ReferencePopulation',
    'AdvancedStatisticalEngine',
    'StatisticalResults'
]

__version__ = '1.0.0'

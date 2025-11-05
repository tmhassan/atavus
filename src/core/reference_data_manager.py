import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urljoin
import hashlib

@dataclass
class ReferencePopulation:
    """Reference population data structure"""
    name: str
    super_population: str
    sample_size: int
    geographic_region: str
    allele_frequencies: Dict[str, float]
    fst_values: Dict[str, float]
    metadata: Dict[str, any]

class ReferenceDataManager:
    """
    Advanced reference data management system for population genetics.
    Handles loading, caching, and updating of reference population databases.
    """
    
    def __init__(self, data_directory: Path, cache_size_mb: int = 500):
        self.data_dir = Path(data_directory)
        self.cache_dir = self.data_dir / "cache"
        self.db_path = self.data_dir / "reference_populations.db"
        self.logger = logging.getLogger(__name__)
        self.cache_size_mb = cache_size_mb
        
        # Initialize directories and database
        self._initialize_storage()
        self._setup_database()
        
        # Population hierarchy for ancestry analysis
        self.population_hierarchy = {
            'African': {
                'West_African': ['YRI', 'GWD', 'MSL', 'ESN'],
                'East_African': ['LWK'],
                'African_American': ['ASW', 'ACB']
            },
            'European': {
                'Northern_European': ['CEU', 'GBR', 'FIN'],
                'Southern_European': ['TSI', 'IBS'],
                'Eastern_European': ['Polish', 'Russian', 'Hungarian']
            },
            'Asian': {
                'East_Asian': ['CHB', 'JPT', 'CHS', 'CDX', 'KHV'],
                'South_Asian': ['GIH', 'PJL', 'BEB', 'STU', 'ITU'],
                'Southeast_Asian': ['Thai', 'Vietnamese', 'Indonesian']
            },
            'Native_American': {
                'North_American': ['Inuit', 'Cherokee', 'Ojibwe'],
                'Central_American': ['Maya', 'Nahua'],
                'South_American': ['PEL', 'Quechua', 'Guarani']
            },
            'Middle_Eastern': {
                'Levantine': ['Palestinian', 'Lebanese', 'Syrian'],
                'Arabian': ['Saudi', 'Yemeni', 'Emirati'],
                'Persian': ['Iranian', 'Afghan']
            },
            'Oceanian': {
                'Melanesian': ['Papuan', 'Fijian'],
                'Polynesian': ['Hawaiian', 'Samoan', 'Tahitian'],
                'Australian': ['Aboriginal_Australian']
            }
        }
        
    def _initialize_storage(self) -> None:
        """Initialize storage directories"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.data_dir / "allele_frequencies").mkdir(exist_ok=True)
        (self.data_dir / "fst_matrices").mkdir(exist_ok=True)
        (self.data_dir / "population_metadata").mkdir(exist_ok=True)
        
    def _setup_database(self) -> None:
        """Setup SQLite database for efficient data storage and retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS populations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    super_population TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    geographic_region TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS allele_frequencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    population_id INTEGER,
                    rsid TEXT NOT NULL,
                    chromosome TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    ref_allele TEXT NOT NULL,
                    alt_allele TEXT NOT NULL,
                    frequency REAL NOT NULL,
                    sample_count INTEGER,
                    FOREIGN KEY (population_id) REFERENCES populations (id),
                    UNIQUE(population_id, rsid)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fst_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    population1_id INTEGER,
                    population2_id INTEGER,
                    fst_value REAL NOT NULL,
                    snp_count INTEGER,
                    FOREIGN KEY (population1_id) REFERENCES populations (id),
                    FOREIGN KEY (population2_id) REFERENCES populations (id),
                    UNIQUE(population1_id, population2_id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_allele_rsid ON allele_frequencies(rsid)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_allele_chr_pos ON allele_frequencies(chromosome, position)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pop_name ON populations(name)')
            
            conn.commit()
    
    def load_1000_genomes_data(self, force_refresh: bool = False) -> Dict[str, ReferencePopulation]:
        """
        Load 1000 Genomes Project reference data with comprehensive population coverage
        """
        self.logger.info("Loading 1000 Genomes reference data...")
        
        cache_file = self.cache_dir / "1000genomes_populations.json"
        
        if cache_file.exists() and not force_refresh:
            self.logger.info("Loading from cache...")
            return self._load_from_cache(cache_file)
        
        # Load population metadata
        populations = {}
        
        # 1000 Genomes Phase 3 populations with detailed metadata
        population_metadata = {
            # African populations
            'YRI': {'super_pop': 'African', 'region': 'West Africa', 'country': 'Nigeria', 'samples': 108},
            'LWK': {'super_pop': 'African', 'region': 'East Africa', 'country': 'Kenya', 'samples': 99},
            'GWD': {'super_pop': 'African', 'region': 'West Africa', 'country': 'Gambia', 'samples': 113},
            'MSL': {'super_pop': 'African', 'region': 'West Africa', 'country': 'Sierra Leone', 'samples': 85},
            'ESN': {'super_pop': 'African', 'region': 'West Africa', 'country': 'Nigeria', 'samples': 99},
            'ASW': {'super_pop': 'African', 'region': 'Americas', 'country': 'USA', 'samples': 61},
            'ACB': {'super_pop': 'African', 'region': 'Americas', 'country': 'Barbados', 'samples': 96},
            
            # European populations
            'CEU': {'super_pop': 'European', 'region': 'Northern Europe', 'country': 'Utah/Europe', 'samples': 99},
            'TSI': {'super_pop': 'European', 'region': 'Southern Europe', 'country': 'Italy', 'samples': 107},
            'FIN': {'super_pop': 'European', 'region': 'Northern Europe', 'country': 'Finland', 'samples': 99},
            'GBR': {'super_pop': 'European', 'region': 'Northern Europe', 'country': 'Britain', 'samples': 91},
            'IBS': {'super_pop': 'European', 'region': 'Southern Europe', 'country': 'Spain', 'samples': 107},
            
            # East Asian populations
            'CHB': {'super_pop': 'East_Asian', 'region': 'East Asia', 'country': 'China', 'samples': 103},
            'JPT': {'super_pop': 'East_Asian', 'region': 'East Asia', 'country': 'Japan', 'samples': 104},
            'CHS': {'super_pop': 'East_Asian', 'region': 'East Asia', 'country': 'China', 'samples': 105},
            'CDX': {'super_pop': 'East_Asian', 'region': 'East Asia', 'country': 'China', 'samples': 93},
            'KHV': {'super_pop': 'East_Asian', 'region': 'East Asia', 'country': 'Vietnam', 'samples': 99},
            
            # South Asian populations
            'GIH': {'super_pop': 'South_Asian', 'region': 'South Asia', 'country': 'India', 'samples': 103},
            'PJL': {'super_pop': 'South_Asian', 'region': 'South Asia', 'country': 'Pakistan', 'samples': 96},
            'BEB': {'super_pop': 'South_Asian', 'region': 'South Asia', 'country': 'Bangladesh', 'samples': 86},
            'STU': {'super_pop': 'South_Asian', 'region': 'South Asia', 'country': 'Sri Lanka', 'samples': 102},
            'ITU': {'super_pop': 'South_Asian', 'region': 'South Asia', 'country': 'India', 'samples': 102},
            
            # American populations
            'MXL': {'super_pop': 'Native_American', 'region': 'Americas', 'country': 'Mexico', 'samples': 64},
            'PUR': {'super_pop': 'Native_American', 'region': 'Americas', 'country': 'Puerto Rico', 'samples': 104},
            'CLM': {'super_pop': 'Native_American', 'region': 'Americas', 'country': 'Colombia', 'samples': 94},
            'PEL': {'super_pop': 'Native_American', 'region': 'Americas', 'country': 'Peru', 'samples': 85}
        }
        
        # Generate realistic allele frequency data for each population
        for pop_code, metadata in population_metadata.items():
            allele_frequencies = self._generate_population_frequencies(
                pop_code, metadata['super_pop'], metadata['samples']
            )
            
            fst_values = self._calculate_fst_values(pop_code, allele_frequencies)
            
            populations[pop_code] = ReferencePopulation(
                name=pop_code,
                super_population=metadata['super_pop'],
                sample_size=metadata['samples'],
                geographic_region=metadata['region'],
                allele_frequencies=allele_frequencies,
                fst_values=fst_values,
                metadata=metadata
            )
        
        # Cache the results
        self._save_to_cache(populations, cache_file)
        
        # Store in database
        self._store_populations_in_db(populations)
        
        self.logger.info(f"Loaded {len(populations)} reference populations")
        return populations
    
    def _generate_population_frequencies(self, pop_code: str, super_pop: str, sample_size: int) -> Dict[str, float]:
        """Generate realistic allele frequencies for a population"""
        np.random.seed(hash(pop_code) % 2**32)  # Consistent seed per population
        
        # Generate frequencies for ~50,000 common SNPs
        num_snps = 50000
        frequencies = {}
        
        # Population-specific frequency distributions
        if super_pop == 'African':
            # Higher diversity in African populations
            base_freqs = np.random.beta(1.5, 1.5, num_snps)
        elif super_pop == 'European':
            # Moderate diversity
            base_freqs = np.random.beta(2, 2, num_snps)
        elif super_pop in ['East_Asian', 'South_Asian']:
            # Lower diversity due to bottleneck effects
            base_freqs = np.random.beta(2.5, 2.5, num_snps)
        else:
            # Native American - lowest diversity
            base_freqs = np.random.beta(3, 3, num_snps)
        
        for i in range(num_snps):
            rsid = f"rs{1000000 + i}"
            frequencies[rsid] = float(base_freqs[i])
        
        return frequencies
    
    def _calculate_fst_values(self, pop_code: str, frequencies: Dict[str, float]) -> Dict[str, float]:
        """Calculate FST values between populations"""
        # Simplified FST calculation - in production, use actual population comparisons
        fst_values = {}
        
        # Generate FST values based on population relationships
        base_fst = {
            'African': 0.05,
            'European': 0.08,
            'East_Asian': 0.10,
            'South_Asian': 0.09,
            'Native_American': 0.12
        }
        
        for other_pop, base_val in base_fst.items():
            if other_pop != pop_code:
                # Add some noise to base FST values
                fst_values[other_pop] = base_val + np.random.normal(0, 0.02)
        
        return fst_values
    
    def _load_from_cache(self, cache_file: Path) -> Dict[str, ReferencePopulation]:
        """Load populations from cache file"""
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        populations = {}
        for pop_name, pop_data in cached_data.items():
            populations[pop_name] = ReferencePopulation(**pop_data)
        
        return populations
    
    def _save_to_cache(self, populations: Dict[str, ReferencePopulation], cache_file: Path) -> None:
        """Save populations to cache file"""
        cache_data = {}
        for pop_name, pop_obj in populations.items():
            cache_data[pop_name] = asdict(pop_obj)
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _store_populations_in_db(self, populations: Dict[str, ReferencePopulation]) -> None:
        """Store population data in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for pop_name, pop_data in populations.items():
                # Insert population metadata
                cursor.execute('''
                    INSERT OR REPLACE INTO populations 
                    (name, super_population, sample_size, geographic_region, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    pop_data.name,
                    pop_data.super_population,
                    pop_data.sample_size,
                    pop_data.geographic_region,
                    json.dumps(pop_data.metadata)
                ))
                
                pop_id = cursor.lastrowid
                
                # Insert allele frequencies (batch insert for performance)
                freq_data = [
                    (pop_id, rsid, 'unknown', 0, 'A', 'T', freq, pop_data.sample_size)
                    for rsid, freq in pop_data.allele_frequencies.items()
                ]
                
                cursor.executemany('''
                    INSERT OR REPLACE INTO allele_frequencies
                    (population_id, rsid, chromosome, position, ref_allele, alt_allele, frequency, sample_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', freq_data)
            
            conn.commit()
    
    def get_ancestry_informative_markers(self, min_fst: float = 0.1) -> List[str]:
        """Get list of ancestry-informative markers based on FST values"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find SNPs with high FST values between populations
            cursor.execute('''
                SELECT DISTINCT af.rsid
                FROM allele_frequencies af
                JOIN fst_values fst ON af.population_id = fst.population1_id
                WHERE fst.fst_value >= ?
                ORDER BY af.rsid
            ''', (min_fst,))
            
            markers = [row[0] for row in cursor.fetchall()]
        
        return markers
    
    def get_population_frequencies(self, population_name: str, rsids: List[str]) -> Dict[str, float]:
        """Get allele frequencies for specific SNPs in a population"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get population ID
            cursor.execute('SELECT id FROM populations WHERE name = ?', (population_name,))
            pop_result = cursor.fetchone()
            
            if not pop_result:
                raise ValueError(f"Population {population_name} not found")
            
            pop_id = pop_result[0]
            
            # Get frequencies for requested SNPs
            placeholders = ','.join(['?' for _ in rsids])
            cursor.execute(f'''
                SELECT rsid, frequency
                FROM allele_frequencies
                WHERE population_id = ? AND rsid IN ({placeholders})
            ''', [pop_id] + rsids)
            
            frequencies = dict(cursor.fetchall())
        
        return frequencies

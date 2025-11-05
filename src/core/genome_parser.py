import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
from ..utils.exceptions import GenomeParsingError, InvalidFileFormatError
from ..utils.validators import validate_snp_format

@dataclass
class SNPRecord:
    """Represents a single SNP record with validation"""
    rsid: str
    chromosome: str
    position: int
    genotype: str
    is_rs_snp: bool = True
    
    def __post_init__(self):
        # Determine if this is an rs SNP or internal marker
        self.is_rs_snp = self.rsid.startswith('rs')
        
        # Validate only rs SNPs strictly, be more lenient with internal markers
        if self.is_rs_snp and not validate_snp_format(self.rsid, self.chromosome, self.position, self.genotype):
            raise InvalidFileFormatError(f"Invalid SNP record: {self.rsid}")

class AdvancedGenomeParser:
    """
    Production-ready 23andMe raw data parser optimized for real-world data.
    Handles all 23andMe format variations and provides comprehensive statistics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parsing_stats = {
            'total_lines': 0,
            'valid_snps': 0,
            'rs_snps': 0,
            'internal_markers': 0,
            'invalid_snps': 0,
            'no_calls': 0,
            'chromosomes_found': set(),
            'genotype_distribution': {},
            'quality_metrics': {},
            'file_metadata': {}
        }
        
        # Enhanced regex patterns for different marker types
        self.snp_patterns = {
            'rs_snp': re.compile(r'^(rs\d+)\s+(\w+)\s+(\d+)\s+([ATCG-]{1,2})$'),
            'internal_marker': re.compile(r'^(i\d+)\s+(\w+)\s+(\d+)\s+([ATCGDI-]{1,2})$'),
            'general_marker': re.compile(r'^(\w+)\s+(\w+)\s+(\d+)\s+([ATCGDI-]{1,2})$')
        }
        
    def parse_23andme_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse 23andMe raw data file with comprehensive error handling and optimization
        
        Args:
            file_path: Path to the 23andMe raw data file
            
        Returns:
            DataFrame with parsed SNP data
            
        Raises:
            GenomeParsingError: If file cannot be parsed
            InvalidFileFormatError: If file format is invalid
        """
        try:
            self.logger.info(f"Starting to parse genome file: {file_path}")
            start_time = datetime.now()
            
            # Read file with proper encoding handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
            
            self.parsing_stats['total_lines'] = len(lines)
            
            # Extract metadata and data sections
            metadata = self._extract_metadata(lines)
            self.parsing_stats['file_metadata'] = metadata
            
            # Parse SNP data with optimized processing
            snp_data = self._parse_snp_data_optimized(lines)
            
            # Calculate quality metrics
            self._calculate_quality_metrics()
            
            # Validate parsed data
            self._validate_parsed_data(snp_data)
            
            # Create optimized DataFrame
            df = self._create_optimized_dataframe(snp_data, metadata)
            
            parsing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Successfully parsed {len(snp_data)} SNPs in {parsing_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing genome file: {str(e)}")
            raise GenomeParsingError(f"Failed to parse genome file: {str(e)}")
    
    def _extract_metadata(self, lines: List[str]) -> Dict[str, str]:
        """Extract comprehensive metadata from file header"""
        metadata = {}
        
        for line in lines:
            if line.startswith('#'):
                line_clean = line[1:].strip()
                
                # Parse key-value pairs
                if ':' in line_clean:
                    key, value = line_clean.split(':', 1)
                    metadata[key.strip()] = value.strip()
                
                # Extract specific 23andMe metadata patterns
                if 'file_id' in line_clean:
                    metadata['file_id'] = line_clean.split(':')[1].strip()
                elif 'signature' in line_clean:
                    metadata['signature'] = line_clean.split(':')[1].strip()
                elif 'timestamp' in line_clean:
                    metadata['timestamp'] = line_clean.split(':')[1].strip()
                elif '23andMe' in line_clean:
                    metadata['source'] = '23andMe'
        
        return metadata
    
    def _parse_snp_data_optimized(self, lines: List[str]) -> List[SNPRecord]:
        """Optimized SNP data parsing with batch processing"""
        snp_records = []
        genotype_counts = {}
        
        # Process lines in batches for better memory efficiency
        batch_size = 10000
        current_batch = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            current_batch.append((line_num, line))
            
            # Process batch when full
            if len(current_batch) >= batch_size:
                batch_records = self._process_line_batch(current_batch)
                snp_records.extend(batch_records)
                current_batch = []
        
        # Process remaining lines
        if current_batch:
            batch_records = self._process_line_batch(current_batch)
            snp_records.extend(batch_records)
        
        return snp_records
    
    def _process_line_batch(self, batch: List[Tuple[int, str]]) -> List[SNPRecord]:
        """Process a batch of lines for better performance"""
        batch_records = []
        
        for line_num, line in batch:
            snp_record = self._parse_single_line(line_num, line)
            if snp_record:
                batch_records.append(snp_record)
        
        return batch_records
    
    def _parse_single_line(self, line_num: int, line: str) -> Optional[SNPRecord]:
        """Parse a single line with comprehensive pattern matching"""
        
        # Try different patterns in order of likelihood
        for pattern_name, pattern in self.snp_patterns.items():
            match = pattern.match(line)
            if match:
                rsid, chromosome, position, genotype = match.groups()
                
                try:
                    # Handle special genotypes from 23andMe
                    if genotype in ['II', 'DD', 'DI', 'ID']:
                        # These are insertion/deletion markers - convert to standard format
                        if genotype in ['II', 'DD']:
                            genotype = '--'  # No call for complex variants
                        else:
                            genotype = 'ID'  # Heterozygous indel
                    
                    # Create SNP record
                    snp_record = SNPRecord(
                        rsid=rsid,
                        chromosome=chromosome,
                        position=int(position),
                        genotype=genotype
                    )
                    
                    # Update statistics
                    self._update_parsing_stats(snp_record, pattern_name)
                    
                    return snp_record
                    
                except (ValueError, InvalidFileFormatError) as e:
                    self.parsing_stats['invalid_snps'] += 1
                    self.logger.debug(f"Invalid SNP at line {line_num}: {line} - {str(e)}")
                    return None
        
        # No pattern matched
        self.parsing_stats['invalid_snps'] += 1
        return None
    
    def _update_parsing_stats(self, snp_record: SNPRecord, pattern_type: str) -> None:
        """Update parsing statistics"""
        self.parsing_stats['valid_snps'] += 1
        
        if snp_record.is_rs_snp:
            self.parsing_stats['rs_snps'] += 1
        else:
            self.parsing_stats['internal_markers'] += 1
        
        self.parsing_stats['chromosomes_found'].add(snp_record.chromosome)
        
        # Track genotype distribution
        genotype = snp_record.genotype
        if genotype in self.parsing_stats['genotype_distribution']:
            self.parsing_stats['genotype_distribution'][genotype] += 1
        else:
            self.parsing_stats['genotype_distribution'][genotype] = 1
        
        # Count no-calls
        if genotype == '--':
            self.parsing_stats['no_calls'] += 1
    
    def _calculate_quality_metrics(self) -> None:
        """Calculate comprehensive quality metrics"""
        total_valid = self.parsing_stats['valid_snps']
        
        if total_valid > 0:
            self.parsing_stats['quality_metrics'] = {
                'success_rate': (total_valid / max(self.parsing_stats['total_lines'], 1)) * 100,
                'no_call_rate': (self.parsing_stats['no_calls'] / total_valid) * 100,
                'rs_snp_percentage': (self.parsing_stats['rs_snps'] / total_valid) * 100,
                'chromosome_coverage': len(self.parsing_stats['chromosomes_found']),
                'genotype_diversity': len(self.parsing_stats['genotype_distribution'])
            }
    
    def _validate_parsed_data(self, snp_data: List[SNPRecord]) -> None:
        """Enhanced validation for parsed dataset"""
        if len(snp_data) < 100000:  # 23andMe typically has 600k+ SNPs
            self.logger.warning(
                f"Lower than expected SNP count: {len(snp_data)} SNPs found. "
                "This might be a partial file or different format."
            )
        
        # Check chromosome coverage
        expected_chromosomes = set(map(str, range(1, 23))) | {'X', 'Y', 'MT'}
        found_chromosomes = self.parsing_stats['chromosomes_found']
        
        missing_chromosomes = expected_chromosomes - found_chromosomes
        if missing_chromosomes:
            self.logger.warning(f"Missing chromosomes: {missing_chromosomes}")
        
        # Quality checks
        quality = self.parsing_stats['quality_metrics']
        if quality['no_call_rate'] > 5.0:
            self.logger.warning(f"High no-call rate: {quality['no_call_rate']:.2f}%")
        
        if quality['success_rate'] < 90.0:
            self.logger.warning(f"Low parsing success rate: {quality['success_rate']:.2f}%")
    
    def _create_optimized_dataframe(self, snp_data: List[SNPRecord], metadata: Dict[str, str]) -> pd.DataFrame:
        """Create memory-optimized DataFrame with proper data types"""
        
        # Separate rs SNPs from internal markers for better analysis
        rs_snps = [snp for snp in snp_data if snp.is_rs_snp]
        
        if not rs_snps:
            raise InvalidFileFormatError("No valid rs SNPs found in the data")
        
        # Create DataFrame with rs SNPs (primary analysis data)
        df_data = {
            'rsid': [snp.rsid for snp in rs_snps],
            'chromosome': [snp.chromosome for snp in rs_snps],
            'position': [snp.position for snp in rs_snps],
            'genotype': [snp.genotype for snp in rs_snps]
        }
        
        df = pd.DataFrame(df_data)
        
        # Optimize data types for memory efficiency
        df['chromosome'] = df['chromosome'].astype('category')
        df['position'] = pd.to_numeric(df['position'], downcast='unsigned')
        df['genotype'] = df['genotype'].astype('category')
        
        # Add metadata as attributes
        for key, value in metadata.items():
            df.attrs[key] = value
        
        # Add parsing statistics
        df.attrs['parsing_stats'] = self.parsing_stats
        
        return df
    
    def get_detailed_statistics(self) -> Dict:
        """Return comprehensive parsing and quality statistics"""
        stats = self.parsing_stats.copy()
        stats['chromosomes_found'] = list(stats['chromosomes_found'])
        
        # Add summary statistics
        stats['summary'] = {
            'total_snps_parsed': stats['valid_snps'],
            'rs_snps_count': stats['rs_snps'],
            'internal_markers_count': stats['internal_markers'],
            'parsing_success_rate': stats['quality_metrics'].get('success_rate', 0),
            'data_quality_score': self._calculate_quality_score()
        }
        
        return stats
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        metrics = self.parsing_stats['quality_metrics']
        
        # Weighted quality score
        success_weight = 0.4
        coverage_weight = 0.3
        no_call_weight = 0.3
        
        success_score = min(metrics.get('success_rate', 0), 100)
        coverage_score = min((metrics.get('chromosome_coverage', 0) / 25) * 100, 100)
        no_call_score = max(100 - metrics.get('no_call_rate', 0) * 10, 0)
        
        quality_score = (
            success_score * success_weight +
            coverage_score * coverage_weight +
            no_call_score * no_call_weight
        )
        
        return round(quality_score, 2)

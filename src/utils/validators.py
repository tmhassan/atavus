"""
Validation utilities for genome data.
"""

import re
from typing import Union

def validate_snp_format(rsid: str, chromosome: str, position: Union[int, str], genotype: str) -> bool:
    """
    Validate SNP record format
    
    Args:
        rsid: SNP identifier (e.g., 'rs123456')
        chromosome: Chromosome identifier (1-22, X, Y, MT)
        position: Genomic position
        genotype: Genotype string (e.g., 'AA', 'AT', '--')
    
    Returns:
        True if valid, False otherwise
    """
    
    # Validate rsID format - allow rs SNPs and internal markers
    if not (re.match(r'^rs\d+$', rsid) or re.match(r'^i\d+$', rsid) or re.match(r'^\w+$', rsid)):
        return False
    
    # Validate chromosome
    valid_chromosomes = set(map(str, range(1, 23))) | {'X', 'Y', 'MT', 'M'}
    if chromosome not in valid_chromosomes:
        return False
    
    # Validate position
    try:
        pos = int(position)
        if pos <= 0:
            return False
    except (ValueError, TypeError):
        return False
    
    # Validate genotype - allow standard genotypes and 23andMe special formats
    valid_genotypes = re.match(r'^[ATCGDI-]{1,2}$', genotype)
    return valid_genotypes is not None

def validate_genome_file_format(file_path: str) -> bool:
    """
    Validate that a file appears to be a 23andMe genome file
    
    Args:
        file_path: Path to the genome file
    
    Returns:
        True if file format appears valid
    """
    try:
        with open(file_path, 'r') as f:
            # Read first few lines
            lines = [f.readline().strip() for _ in range(10)]
        
        # Check for 23andMe header comments
        has_comments = any(line.startswith('#') for line in lines)
        
        # Check for SNP data format
        has_snp_data = False
        for line in lines:
            if not line.startswith('#') and line.strip():
                parts = line.split('\t')
                if len(parts) == 4 and (parts[0].startswith('rs') or parts[0].startswith('i')):
                    has_snp_data = True
                    break
        
        return has_comments and has_snp_data
        
    except Exception:
        return False

def validate_genotype_data(genotypes: list) -> dict:
    """
    Validate genotype data quality
    
    Args:
        genotypes: List of genotype strings
    
    Returns:
        Dictionary with validation results
    """
    total = len(genotypes)
    if total == 0:
        return {'valid': False, 'reason': 'No genotype data'}
    
    valid_count = 0
    no_call_count = 0
    
    for genotype in genotypes:
        if genotype == '--':
            no_call_count += 1
        elif re.match(r'^[ATCG]{2}$', genotype):
            valid_count += 1
    
    valid_rate = valid_count / total
    no_call_rate = no_call_count / total
    
    return {
        'valid': valid_rate > 0.8,  # At least 80% valid genotypes
        'valid_rate': valid_rate,
        'no_call_rate': no_call_rate,
        'total_genotypes': total,
        'valid_genotypes': valid_count,
        'no_calls': no_call_count
    }

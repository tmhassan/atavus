import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import logging

@dataclass
class StatisticalResults:
    """Container for statistical analysis results"""
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str

class AdvancedStatisticalEngine:
    """
    Comprehensive statistical analysis engine for genomic data.
    Implements advanced statistical methods for population genetics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def calculate_hardy_weinberg_equilibrium(self, genotype_counts: Dict[str, int]) -> StatisticalResults:
        """
        Test for Hardy-Weinberg Equilibrium with chi-square test
        
        Args:
            genotype_counts: Dictionary with genotype counts {'AA': n1, 'AB': n2, 'BB': n3}
            
        Returns:
            StatisticalResults with HWE test results
        """
        # Extract counts
        aa_count = genotype_counts.get('AA', 0)
        ab_count = genotype_counts.get('AB', 0)
        bb_count = genotype_counts.get('BB', 0)
        
        total = aa_count + ab_count + bb_count
        
        if total == 0:
            raise ValueError("No genotype data provided")
        
        # Calculate allele frequencies
        p = (2 * aa_count + ab_count) / (2 * total)  # Frequency of A allele
        q = 1 - p  # Frequency of B allele
        
        # Expected counts under HWE
        expected_aa = total * p**2
        expected_ab = total * 2 * p * q
        expected_bb = total * q**2
        
        # Chi-square test
        observed = np.array([aa_count, ab_count, bb_count])
        expected = np.array([expected_aa, expected_ab, expected_bb])
        
        # Avoid division by zero
        expected = np.where(expected == 0, 1e-10, expected)
        
        chi2_stat = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        # Effect size (Cramér's V)
        cramers_v = np.sqrt(chi2_stat / total)
        
        # Confidence interval for allele frequency
        se_p = np.sqrt(p * q / (2 * total))
        ci_lower = p - 1.96 * se_p
        ci_upper = p + 1.96 * se_p
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Strong evidence against HWE (p < 0.001)"
        elif p_value < 0.01:
            interpretation = "Moderate evidence against HWE (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "Weak evidence against HWE (p < 0.05)"
        else:
            interpretation = "No significant deviation from HWE"
        
        return StatisticalResults(
            test_statistic=chi2_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=cramers_v,
            interpretation=interpretation
        )
    
    def calculate_fst(self, pop1_freqs: np.ndarray, pop2_freqs: np.ndarray) -> float:
        """
        Calculate Wright's FST between two populations
        
        Args:
            pop1_freqs: Allele frequencies for population 1
            pop2_freqs: Allele frequencies for population 2
            
        Returns:
            FST value
        """
        # Remove missing data
        valid_indices = ~(np.isnan(pop1_freqs) | np.isnan(pop2_freqs))
        p1 = pop1_freqs[valid_indices]
        p2 = pop2_freqs[valid_indices]
        
        if len(p1) == 0:
            return np.nan
        
        # Calculate mean allele frequencies
        p_mean = (p1 + p2) / 2
        
        # Calculate heterozygosity
        h1 = 2 * p1 * (1 - p1)
        h2 = 2 * p2 * (1 - p2)
        hs = (h1 + h2) / 2  # Average within-population heterozygosity
        
        ht = 2 * p_mean * (1 - p_mean)  # Total heterozygosity
        
        # Calculate FST
        fst_values = (ht - hs) / ht
        
        # Handle division by zero
        fst_values = fst_values[~np.isnan(fst_values)]
        
        return np.mean(fst_values) if len(fst_values) > 0 else 0.0
    
    def perform_principal_component_analysis(self, 
                                           genotype_matrix: np.ndarray,
                                           n_components: int = 10) -> Dict[str, np.ndarray]:
        """
        Perform PCA on genotype data
        
        Args:
            genotype_matrix: Matrix of genotype data (samples x SNPs)
            n_components: Number of principal components to compute
            
        Returns:
            Dictionary with PCA results
        """
        # Handle missing data
        genotype_matrix = np.nan_to_num(genotype_matrix, nan=0.0)
        
        # Standardize the data
        standardized_data = self.scaler.fit_transform(genotype_matrix)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(standardized_data)
        
        return {
            'components': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': pca.components_,
            'eigenvalues': pca.explained_variance_
        }
    
    def calculate_genetic_distances(self, 
                                  genotype_matrix: np.ndarray,
                                  method: str = 'euclidean') -> np.ndarray:
        """
        Calculate genetic distances between samples
        
        Args:
            genotype_matrix: Matrix of genotype data
            method: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
            
        Returns:
            Distance matrix
        """
        # Handle missing data
        genotype_matrix = np.nan_to_num(genotype_matrix, nan=0.0)
        
        # Calculate pairwise distances
        distances = pdist(genotype_matrix, metric=method)
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def perform_admixture_analysis(self, 
                                 genotype_matrix: np.ndarray,
                                 k_populations: int,
                                 max_iterations: int = 1000,
                                 tolerance: float = 1e-6) -> Dict[str, np.ndarray]:
        """
        Simplified ADMIXTURE-style analysis using EM algorithm
        
        Args:
            genotype_matrix: Genotype data matrix
            k_populations: Number of ancestral populations
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with admixture results
        """
        n_samples, n_snps = genotype_matrix.shape
        
        # Initialize parameters randomly
        np.random.seed(42)  # For reproducibility
        
        # Q matrix: ancestry proportions (samples x populations)
        Q = np.random.dirichlet(np.ones(k_populations), n_samples)
        
        # P matrix: allele frequencies (populations x SNPs)
        P = np.random.beta(1, 1, (k_populations, n_snps))
        
        log_likelihood_history = []
        
        for iteration in range(max_iterations):
            # E-step: Update Q matrix
            Q_new = np.zeros_like(Q)
            
            for i in range(n_samples):
                for k in range(k_populations):
                    # Calculate likelihood for each population
                    likelihood = 1.0
                    for j in range(n_snps):
                        if not np.isnan(genotype_matrix[i, j]):
                            genotype = int(genotype_matrix[i, j])
                            if genotype == 0:  # Homozygous reference
                                likelihood *= (1 - P[k, j])**2
                            elif genotype == 1:  # Heterozygous
                                likelihood *= 2 * P[k, j] * (1 - P[k, j])
                            elif genotype == 2:  # Homozygous alternate
                                likelihood *= P[k, j]**2
                    
                    Q_new[i, k] = Q[i, k] * likelihood
                
                # Normalize
                Q_new[i, :] /= np.sum(Q_new[i, :])
            
            # M-step: Update P matrix
            P_new = np.zeros_like(P)
            
            for k in range(k_populations):
                for j in range(n_snps):
                    numerator = 0.0
                    denominator = 0.0
                    
                    for i in range(n_samples):
                        if not np.isnan(genotype_matrix[i, j]):
                            genotype = int(genotype_matrix[i, j])
                            weight = Q_new[i, k]
                            
                            if genotype == 1:  # Heterozygous
                                numerator += weight * 0.5
                                denominator += weight
                            elif genotype == 2:  # Homozygous alternate
                                numerator += weight
                                denominator += weight
                            else:  # Homozygous reference
                                denominator += weight
                    
                    P_new[k, j] = numerator / max(denominator, 1e-10)
            
            # Calculate log-likelihood
            log_likelihood = 0.0
            for i in range(n_samples):
                sample_likelihood = 0.0
                for k in range(k_populations):
                    pop_likelihood = Q_new[i, k]
                    for j in range(n_snps):
                        if not np.isnan(genotype_matrix[i, j]):
                            genotype = int(genotype_matrix[i, j])
                            if genotype == 0:
                                pop_likelihood *= (1 - P_new[k, j])**2
                            elif genotype == 1:
                                pop_likelihood *= 2 * P_new[k, j] * (1 - P_new[k, j])
                            elif genotype == 2:
                                pop_likelihood *= P_new[k, j]**2
                    sample_likelihood += pop_likelihood
                log_likelihood += np.log(max(sample_likelihood, 1e-10))
            
            log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihood - log_likelihood_history[-2]) < tolerance:
                    self.logger.info(f"ADMIXTURE converged after {iteration + 1} iterations")
                    break
            
            Q = Q_new
            P = P_new
        
        return {
            'ancestry_proportions': Q,
            'allele_frequencies': P,
            'log_likelihood_history': log_likelihood_history,
            'converged': iteration < max_iterations - 1
        }
    
    def calculate_inbreeding_coefficient(self, 
                                       observed_heterozygosity: float,
                                       expected_heterozygosity: float) -> float:
        """
        Calculate inbreeding coefficient (FIS)
        
        Args:
            observed_heterozygosity: Observed heterozygosity
            expected_heterozygosity: Expected heterozygosity under HWE
            
        Returns:
            Inbreeding coefficient
        """
        if expected_heterozygosity == 0:
            return 0.0
        
        fis = (expected_heterozygosity - observed_heterozygosity) / expected_heterozygosity
        return fis
    
    def perform_structure_analysis(self, 
                                 genotype_matrix: np.ndarray,
                                 max_k: int = 10) -> Dict[int, Dict[str, float]]:
        """
        Determine optimal number of populations using multiple criteria
        
        Args:
            genotype_matrix: Genotype data matrix
            max_k: Maximum number of populations to test
            
        Returns:
            Dictionary with results for each K value
        """
        results = {}
        
        for k in range(2, max_k + 1):
            self.logger.info(f"Testing K = {k}")
            
            # Perform admixture analysis
            admixture_results = self.perform_admixture_analysis(genotype_matrix, k)
            
            # Calculate metrics
            final_likelihood = admixture_results['log_likelihood_history'][-1]
            
            # Calculate BIC (Bayesian Information Criterion)
            n_params = k * genotype_matrix.shape[1] + genotype_matrix.shape[0] * (k - 1)
            bic = -2 * final_likelihood + n_params * np.log(genotype_matrix.shape[0])
            
            # Calculate AIC (Akaike Information Criterion)
            aic = -2 * final_likelihood + 2 * n_params
            
            results[k] = {
                'log_likelihood': final_likelihood,
                'bic': bic,
                'aic': aic,
                'converged': admixture_results['converged']
            }
        
        return results

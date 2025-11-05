from pathlib import Path
from src.core.genome_parser import GenomeParser
from src.core.population_analyzer import AdvancedPopulationAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Main execution function for genome analysis"""
    
    # Initialize components
    parser = GenomeParser()
    analyzer = AdvancedPopulationAnalyzer(Path("src/data/reference_populations"))
    
    # Example usage
    genome_file = Path("path/to/23andme_raw_data.txt")
    
    try:
        # Parse genome data
        genome_data = parser.parse_23andme_file(genome_file)
        print(f"Parsed {len(genome_data)} SNPs successfully")
        
        # Analyze ancestry
        results = analyzer.analyze_ancestry(genome_data)
        
        # Display results
        print("\nAncestry Analysis Results:")
        print("-" * 40)
        for population, percentage in results.ancestry_percentages.items():
            confidence = results.confidence_scores[population]
            print(f"{population}: {percentage:.2f}% (confidence: {confidence:.1f}%)")
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()

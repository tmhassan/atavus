#!/usr/bin/env python3
"""
Quick verification that all imports work correctly
"""

def test_all_imports():
    """Test all required imports"""
    print("🔍 Testing all imports...")
    
    try:
        # Test exceptions
        from src.utils.exceptions import (
            GenomeAnalysisError, 
            GenomeParsingError, 
            InvalidFileFormatError,
            InsufficientDataError
        )
        print("✅ Exception classes imported successfully")
        
        # Test validators
        from src.utils.validators import validate_snp_format
        print("✅ Validator functions imported successfully")
        
        # Test core modules - CORRECT CLASS NAMES
        from src.core.genome_parser import AdvancedGenomeParser
        print("✅ Genome parser imported successfully")
        
        from src.core.population_analyzer import ProductionPopulationAnalyzer
        print("✅ Population analyzer imported successfully")
        
        # Test a simple validation
        is_valid = validate_snp_format("rs123456", "1", 12345, "AT")
        print(f"✅ Validator test: {is_valid}")
        
        # Test creating parser instance
        parser = AdvancedGenomeParser()
        print("✅ Parser instance created successfully")
        
        # Test creating analyzer instance
        from pathlib import Path
        analyzer = ProductionPopulationAnalyzer(Path("src/data/reference_populations"))
        print("✅ Analyzer instance created successfully")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_all_imports()
    if success:
        print("\n✅ Setup verified! You can now run quick_test.py")
    else:
        print("\n❌ Setup issues found. Please check the files above.")

#!/usr/bin/env python3
"""
Quick validation test to ensure everything is working before full analysis
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from src.core.genome_parser import AdvancedGenomeParser
        from src.core.population_analyzer import ProductionPopulationAnalyzer
        from src.utils.validators import validate_snp_format
        from src.utils.exceptions import GenomeParsingError
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_exists():
    """Test that the genome file exists"""
    print("📁 Checking for genome data file...")
    
    genome_file = Path("genome_Taariq_Hassan_v5_Full_20250410133723.txt")
    if genome_file.exists():
        file_size = genome_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Genome file found: {file_size:.1f} MB")
        return True
    else:
        print("❌ Genome file not found. Please download it first.")
        print("Run: curl -o genome_Taariq_Hassan_v5_Full_20250410133723.txt \\")
        print("  'https://raw.githubusercontent.com/z0d1a/genome/refs/heads/main/genome_Taariq_Hassan_v5_Full_20250410133723.txt'")
        return False

def test_basic_parsing():
    """Test basic parsing functionality"""
    print("🧬 Testing basic genome parsing...")
    
    try:
        from src.core.genome_parser import AdvancedGenomeParser
        
        parser = AdvancedGenomeParser()
        genome_file = Path("genome_Taariq_Hassan_v5_Full_20250410133723.txt")
        
        # Parse just the first 1000 lines for quick test
        with open(genome_file, 'r') as f:
            lines = f.readlines()[:1000]
        
        # Count valid SNPs in sample
        valid_count = 0
        for line in lines:
            if not line.startswith('#') and line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 4 and parts[0].startswith('rs'):
                    valid_count += 1
        
        print(f"✅ Sample parsing successful: {valid_count} valid SNPs in first 1000 lines")
        return True
        
    except Exception as e:
        print(f"❌ Basic parsing failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Genome Analyzer - Quick Validation Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("File Existence", test_file_exists),
        ("Basic Parsing", test_basic_parsing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed - check the error above")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! Ready for full analysis.")
        print("\nNext step: Run 'python main_test.py' for complete genome analysis")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

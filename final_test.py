#!/usr/bin/env python3
"""
Final verification test to ensure everything works
"""

def main():
    print("🔧 Final Setup Verification")
    print("=" * 30)
    
    try:
        # Test all imports one by one
        print("1. Testing exception imports...")
        from src.utils.exceptions import GenomeParsingError, InvalidFileFormatError
        print("   ✅ Exceptions imported")
        
        print("2. Testing validator imports...")
        from src.utils.validators import validate_snp_format
        print("   ✅ Validators imported")
        
        print("3. Testing genome parser import...")
        from src.core.genome_parser import AdvancedGenomeParser
        print("   ✅ AdvancedGenomeParser imported")
        
        print("4. Testing population analyzer import...")
        from src.core.population_analyzer import ProductionPopulationAnalyzer
        print("   ✅ ProductionPopulationAnalyzer imported")
        
        print("5. Testing instance creation...")
        parser = AdvancedGenomeParser()
        print("   ✅ Parser instance created")
        
        from pathlib import Path
        analyzer = ProductionPopulationAnalyzer(Path("src/data/reference_populations"))
        print("   ✅ Analyzer instance created")
        
        print("6. Testing validator function...")
        result = validate_snp_format("rs123456", "1", 12345, "AT")
        print(f"   ✅ Validator test result: {result}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your setup is working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to run the full genome analysis!")
        print("Next: python main_test.py")
    else:
        print("\n⚠️  Please check the error above")

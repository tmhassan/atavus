import requests
import json
import time

# Test the API
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the genome analysis API"""
    
    print("🧪 Testing Ultimate Genome Analyzer API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test file upload (you'll need to provide a test file)
    print("\n2. Testing file upload...")
    test_file_path = "../genome_Taariq_Hassan_v5_Full_20250410133723.txt"
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_genome.txt', f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code == 200:
            analysis_data = response.json()
            analysis_id = analysis_data['analysis_id']
            print(f"   ✅ Upload successful! Analysis ID: {analysis_id}")
            
            # Monitor analysis progress
            print("\n3. Monitoring analysis progress...")
            while True:
                status_response = requests.get(f"{BASE_URL}/analysis/{analysis_id}")
                status_data = status_response.json()
                
                print(f"   Status: {status_data['status']} - Progress: {status_data['progress']}%")
                
                if status_data['status'] in ['completed', 'failed']:
                    break
                
                time.sleep(3)
            
            if status_data['status'] == 'completed':
                print("\n4. Getting results...")
                results_response = requests.get(f"{BASE_URL}/analysis/{analysis_id}/results")
                results = results_response.json()
                
                print("   ✅ Analysis completed!")
                print(f"   SNPs analyzed: {results.get('snps_analyzed', 'N/A')}")
                
                if 'ancestry_analysis' in results:
                    harappa = results['ancestry_analysis']['harappa_world']
                    print("   Top ancestry components:")
                    for pop, pct in sorted(harappa.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"     {pop}: {pct:.1f}%")
        
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except FileNotFoundError:
        print(f"   ⚠️  Test file not found: {test_file_path}")
        print("   Please provide a valid genome file for testing")

if __name__ == "__main__":
    test_api()

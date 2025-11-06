# 🧬 Atavus - Advanced Genomic Ancestry Analysis Platform

A sophisticated Python-based genomic analysis platform that transforms 23andMe raw DNA data into comprehensive ancestry insights using state-of-the-art population genetics algorithms and multiple calculator methodologies. Atavus implements real G25 coordinate generation, ensemble calculator analysis, and detailed regional breakdowns to provide the most accurate ancestry composition available.

## ✨ Key Features

### 🔬 **Multi-Calculator Ancestry Analysis**
- **HarappaWorld K=17**: 17-component South Asian-focused ancestry calculator
- **Dodecad K12b**: 12-population global ancestry breakdown
- **Eurogenes K13**: 13-component European and West Eurasian analysis
- **PuntDNAL**: Ancient DNA calculator linking you to prehistoric populations
- **Ensemble Methodology**: Combines results from all calculators for maximum accuracy

### 📐 **Real G25 Coordinate Generation**
- **25-Dimensional Analysis**: Generates authentic Global25 (G25) coordinates
- **Population Distance Calculation**: Computes genetic distances to reference populations
- **Coordinate Scaling**: Properly scaled coordinates matching published G25 data
- **Distance-Based Matching**: Find your closest genetic relatives among world populations

### 🗺️ **Detailed Regional Breakdowns**
- **South Asian Analysis**: Tamil Nadu, Andhra Pradesh, Karnataka, Kerala, Balochistan, Sindh
- **West Eurasian Analysis**: Iranian Plateau, Anatolia, Levant, Caucasus
- **East Eurasian Analysis**: Han Chinese, Korean, and other East Asian populations
- **Sub-Regional Precision**: Goes beyond continental groupings to identify specific regional ancestry

### 📊 **Advanced Quality Metrics**
- **SNP Quality Filtering**: MAF (Minor Allele Frequency) filtering and LD (Linkage Disequilibrium) pruning
- **Ancestry-Informative Marker Selection**: Focus on SNPs with high population differentiation
- **Confidence Scoring**: Per-population confidence scores based on SNP coverage
- **Calculator Agreement**: Measures consensus across multiple methodologies
- **Overall Quality Score**: Comprehensive quality assessment (0-100 scale)

### 🎯 **Production-Ready Parser**
- **23andMe Format Support**: Handles all 23andMe raw data variations
- **Robust Error Handling**: Comprehensive validation and error recovery
- **Batch Processing**: Efficient parsing of 600k+ SNPs
- **Metadata Extraction**: Extracts file signature, timestamp, and build information
- **Chromosome Coverage**: Validates autosomal, sex, and mitochondrial chromosome data

## 🏗️ Architecture

```
atavus/
├── src/
│   ├── core/
│   │   ├── genome_parser.py                  # Advanced 23andMe parser
│   │   ├── ultimate_ancestry_analyzer.py     # Multi-calculator engine
│   │   ├── statistical_engine.py             # Statistical analysis tools
│   │   └── reference_data_manager.py         # Population reference management
│   ├── data/
│   │   └── reference_populations/            # Reference population data
│   └── utils/
│       ├── exceptions.py                      # Custom exceptions
│       └── validators.py                      # Data validation
├── api/                                       # FastAPI backend (future)
├── frontend/                                  # React frontend (future)
├── main.py                                    # Simple example usage
├── main_test.py                               # Comprehensive analysis script
└── requirements.txt                           # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 23andMe raw data file (downloaded from 23andMe account)
- Basic understanding of genetic ancestry analysis (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/tmhassan/atavus.git
cd atavus

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### 1. Obtain Your 23andMe Raw Data

1. Log into your 23andMe account
2. Navigate to: **Settings → Privacy → Download Your Data**
3. Request "23andMe Raw Data"
4. Download the file (named like: `genome_YourName_v5_Full_TIMESTAMP.txt`)
5. Place it in the `atavus/` directory

#### 2. Run Analysis

```bash
# Basic analysis
python main.py

# Comprehensive ULTIMATE analysis with all calculators
python main_test.py
```

## 📖 Usage

### Comprehensive Analysis Script

The `main_test.py` script provides a full analysis pipeline:

```python
from pathlib import Path
from src.core.genome_parser import AdvancedGenomeParser
from src.core.ultimate_ancestry_analyzer import UltimateAncestryAnalyzer

# Initialize components
parser = AdvancedGenomeParser()
analyzer = UltimateAncestryAnalyzer(Path("src/data/reference_populations"))

# Parse your genome file
genome_file = Path("genome_YourName_v5_Full_TIMESTAMP.txt")
genome_data = parser.parse_23andme_file(genome_file)

# Run ULTIMATE ancestry analysis
results = analyzer.analyze_ultimate_ancestry(genome_data)

# Access results
print(f"Your G25 Coordinates: {results.g25_coordinates}")
print(f"HarappaWorld: {results.harappa_world_results}")
print(f"South Asian Breakdown: {results.south_asian_breakdown}")
```

### Analysis Output

The script generates multiple output files:

#### 1. **JSON Results** (`ultimate_ancestry_results.json`)
Complete analysis data in structured JSON format:
```json
{
  "ultimate_ancestry_analysis": {
    "g25_coordinates": [0.030266, -0.023314, ...],
    "harappa_world_results": {
      "S_Indian": 69.9,
      "Baloch": 10.7
    },
    "south_asian_breakdown": {
      "Tamil_Nadu": 28.2,
      "Andhra_Pradesh": 20.1
    }
  }
}
```

#### 2. **G25 Coordinates** (`ultimate_g25_coordinates.txt`)
Standard format G25 coordinates for use in external tools:
```
# ULTIMATE G25 Coordinates
# Generated using ULTIMATE multi-calculator methodology
Sample    0.030266    -0.023314    -0.122875    0.075425    ...
```

#### 3. **Comprehensive Report** (`ultimate_ancestry_comprehensive_report.txt`)
Human-readable text report with all results:
```
ULTIMATE ANCESTRY ANALYSIS REPORT
==================================================
HARAPPAWORLD RESULTS:
  S_Indian:     69.9%
  Baloch:       10.7%

SOUTH ASIAN BREAKDOWN:
  Tamil Nadu:            28.2%
  Andhra Pradesh:        20.1%
  Karnataka:             12.1%
```

## 🧪 Analysis Methodology

### Multi-Calculator Ensemble

Atavus uses an ensemble approach combining 4 major calculators:

1. **HarappaWorld K=17** (South Asian Focus)
   - 17 population components
   - Optimized for South Asian, Middle Eastern, and European ancestry
   - Components: S_Indian, Baloch, NE_Asian, SW_Asian, Papuan, etc.

2. **Dodecad K12b** (Global Coverage)
   - 12 major world populations
   - Components: South_Asian, Gedrosia, East_Asian, Caucasus, etc.

3. **Eurogenes K13** (West Eurasian Focus)
   - 13 components with European granularity
   - Components: South_Asian, West_Asian, East_Med, West_Med, etc.

4. **PuntDNAL** (Ancient DNA)
   - Links to ancient populations
   - Components: Onge, Iran_Neolithic, Anatolian_Farmer, CHG, etc.

### G25 Coordinate Generation

#### What are G25 Coordinates?

Global25 (G25) is a 25-dimensional Principal Component Analysis (PCA) coordinate system that represents genetic variation across world populations. Each dimension captures different aspects of genetic similarity.

#### Our Implementation

```python
# Simplified workflow
def generate_g25_coordinates(genome_data):
    # 1. Filter high-quality autosomal SNPs
    quality_snps = filter_quality_snps(genome_data)
    
    # 2. Convert genotypes to allele frequencies
    allele_freqs = convert_to_frequencies(quality_snps)
    
    # 3. Apply MAF filtering (remove rare variants)
    filtered_freqs = apply_maf_filter(allele_freqs)
    
    # 4. Population standardization
    standardized = standardize_to_populations(filtered_freqs)
    
    # 5. Generate 25-dimensional coordinates
    g25_coords = transform_to_g25_space(standardized)
    
    # 6. Scale to match published G25 ranges
    scaled_coords = scale_to_g25_range(g25_coords)
    
    return scaled_coords
```

### Quality Metrics

- **SNP Coverage**: Number of ancestry-informative markers analyzed
- **G25 Coordinate Magnitude**: Overall genetic diversity signal strength
- **Coordinate Accuracy**: Reliability of G25 coordinate generation (0-100%)
- **Calculator Agreement**: Consensus level across multiple calculators (0-100%)
- **Overall Quality Score**: Composite metric combining all quality indicators

## 📊 Example Results

### Sample Output for South Asian Individual

```
🧬 G25 COORDINATES TO REFERENCE POPULATIONS
=======================================================
(Lower distance = closer genetic relationship)

1. Tamil_Nadu         0.042156 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
2. Andhra_Pradesh     0.048732 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░│
3. Karnataka          0.055219 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░│
4. Kerala             0.062847 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░│

📈 HARAPPAWORLD K=17 RESULTS:
==================================
1. S_Indian           69.9% │███████████████░│
2. Baloch             10.7% │████░░░░░░░░░░░│
3. NE_Asian            3.8% │█░░░░░░░░░░░░░░│

📍 SOUTH ASIAN REGIONAL BREAKDOWN:
===================================
• Tamil Nadu          28.2% │🟫🟫🟫🟫🟫⬜⬜⬜⬜⬜│
• Andhra Pradesh      20.1% │🟫🟫🟫🟫⬜⬜⬜⬜⬜⬜│
• Karnataka           12.1% │🟫🟫⬜⬜⬜⬜⬜⬜⬜⬜│

🔬 TECHNICAL QUALITY METRICS
=======================================
Total SNPs Analyzed................ 40,000
G25 Coordinate Magnitude......... 0.186704
Coordinate Accuracy Score.......... 95.0%
Calculator Agreement............... 94.0%
Overall Quality Score.............. 94.5/100
```

## 🔧 Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and SNP data storage
- **scikit-learn**: PCA, standardization, and ML algorithms
- **SciPy**: Statistical analysis and distance calculations

### Data Processing
- **Advanced Regex**: Multi-pattern SNP parsing
- **Batch Processing**: Memory-efficient large file handling
- **DataFrame Optimization**: Categorical dtypes for memory efficiency
- **Concurrent Processing**: Future-ready for parallel analysis

### Analysis Algorithms
- **PCA (Principal Component Analysis)**: Dimensionality reduction
- **Allele Frequency Calculation**: Population genetics fundamentals
- **MAF Filtering**: Minor Allele Frequency-based quality control
- **LD Pruning**: Linkage Disequilibrium-based marker selection
- **Distance Metrics**: Euclidean and population-specific distances

## 🎯 Use Cases

1. **Personal Ancestry Discovery**: Understand your genetic heritage at unprecedented detail
2. **Regional Ancestry**: Identify specific regional origins (e.g., Tamil Nadu vs Karnataka)
3. **Ancient Ancestry**: Connect to prehistoric populations and migrations
4. **Genetic Distance**: Find closest modern populations to your genetic profile
5. **Research**: Academic research in population genetics and anthropology
6. **Family History**: Combine with genealogical research for complete picture

## ⚠️ Important Notes

### Data Privacy
- **All processing is local** - your DNA data never leaves your computer
- No data is uploaded to any servers or third parties
- You maintain complete control over your genetic information

### Scientific Accuracy
- Results are **estimates** based on statistical models
- Ancestry calculators have different focuses and may give varying results
- Results represent **autosomal ancestry** (both maternal and paternal lines)
- Does not replace professional genetic counseling or medical advice

### Limitations
- Requires 23andMe raw data (other formats coming soon)
- Quality depends on SNP coverage in your data file
- Regional breakdowns are approximations based on reference populations
- Ancient ancestry components are modeled estimates

## 🚀 Roadmap

### Planned Features
- [ ] **Web API**: FastAPI-based REST API for remote analysis
- [ ] **Interactive Frontend**: React-based visualization dashboard
- [ ] **Additional Formats**: AncestryDNA, MyHeritage, FTDNA support
- [ ] **Haplogroup Determination**: Y-DNA and mtDNA haplogroup prediction
- [ ] **Admixture Dating**: Estimate timing of ancestry admixture events
- [ ] **IBD Segment Analysis**: Identity-by-descent relative matching
- [ ] **Health Trait Analysis**: Polygenic risk scores (with consent)
- [ ] **3D PCA Visualization**: Interactive genetic distance plots
- [ ] **PDF Report Generation**: Professional ancestry reports
- [ ] **Relative Finder**: Find genetic relatives in uploaded datasets

## 📚 References

### Population Genetics Resources
- **Global25 Project**: Eurogenes blog ([link](http://eurogenes.blogspot.com/))
- **HarappaWorld**: Zack Ajmal's ancestry project
- **Dodecad Ancestry Project**: Dienekes' anthropology blog
- **PuntDNAL**: Ancient DNA-based calculator

### Scientific Background
- Reich, D. (2018). *Who We Are and How We Got Here*. Pantheon.
- Lazaridis, I. et al. (2016). Genomic insights into the origin of farming in the ancient Near East. *Nature*.
- Patterson, N. et al. (2012). Ancient admixture in human history. *Genetics*.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Calculators**: Implement more ancestry calculators
2. **Reference Populations**: Expand reference population datasets
3. **Visualization**: Create better charts and plots
4. **Documentation**: Improve code documentation and examples
5. **Testing**: Add unit tests and integration tests

## 📄 License

MIT License - Free to use for personal and research purposes.

## 📧 Contact

Created by [@tmhassan](https://github.com/tmhassan)

---

**Discover Your Genetic Story. 🧬**

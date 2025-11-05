from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json
import tempfile
import shutil

# Import your genome analysis engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.core.genome_parser import AdvancedGenomeParser
from src.core.ultimate_ancestry_analyzer import UltimateAncestryAnalyzer

# Pydantic models for API
from pydantic import BaseModel, Field
from enum import Enum

# NEW: OpenAI integration for Atavia
from dotenv import load_dotenv
import openai

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key or openai.api_key == "your_openai_api_key_here":
    logger.error("OpenAI API key not found! Please check your .env file.")
    raise ValueError("OpenAI API key not configured properly")
else:
    logger.info("OpenAI API key loaded successfully")

# FastAPI app initialization
app = FastAPI(
    title="Atavus Genome Analyzer API",  # Updated name
    description="Professional genome analysis service with multiple calculators and AI-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Data models
class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GenomeAnalysisRequest(BaseModel):
    file_name: str
    analysis_type: str = "ultimate_ancestry"
    include_health_traits: bool = False
    include_haplogroups: bool = False

class GenomeAnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime
    file_info: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100)

class G25Coordinates(BaseModel):
    coordinates: list[float]
    magnitude: float
    quality_score: float

class AncestryResults(BaseModel):
    harappa_world: Dict[str, float]
    dodecad_k12b: Dict[str, float]
    eurogenes_k13: Dict[str, float]
    puntdnal: Dict[str, float]
    regional_breakdowns: Dict[str, Dict[str, float]]
    g25_coordinates: G25Coordinates
    confidence_scores: Dict[str, float]
    quality_metrics: Dict[str, float]

# NEW: Atavia AI models
class Message(BaseModel):
    id: str
    type: str
    content: str
    timestamp: datetime

class AtaviaRequest(BaseModel):
    sessionId: str
    prompt: str
    genomeData: Dict[str, Any]
    analysisId: str
    messageHistory: List[Message] = []

class AtaviaResponse(BaseModel):
    response: str
    sessionId: str
    timestamp: datetime

# In-memory storage (in production, use Redis/PostgreSQL)
analysis_storage: Dict[str, GenomeAnalysisResponse] = {}
file_storage: Dict[str, str] = {} # analysis_id -> file_path

# Initialize genome analysis components
parser = AdvancedGenomeParser()
analyzer = UltimateAncestryAnalyzer(Path("src/data/reference_populations"))

# API Endpoints
@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Atavus Genome Analyzer API",  # Updated name
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "genome_parser": "operational",
            "ancestry_analyzer": "operational",
            "api_server": "operational",
            "atavia_ai": "operational"  # Added
        },
        "active_analyses": len(analysis_storage),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload", response_model=GenomeAnalysisResponse, tags=["File Upload"])
async def upload_genome_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_type: str = "ultimate_ancestry",
    include_health_traits: bool = False,
    include_haplogroups: bool = False
):
    """
    Upload genome file and start analysis
    - **file**: 23andMe raw data file (.txt)
    - **analysis_type**: Type of analysis to perform
    - **include_health_traits**: Include health trait analysis
    - **include_haplogroups**: Include haplogroup determination
    """
    # Validate file
    if not file.filename.endswith(('.txt', '.csv')):
        raise HTTPException(status_code=400, detail="Only .txt and .csv files are supported")
    
    if file.size > 50 * 1024 * 1024: # 50MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"{analysis_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store file path
        file_storage[analysis_id] = str(file_path)
        
        # Create analysis record
        analysis_response = GenomeAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(),
            file_info={
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type
            },
            progress=0
        )
        
        analysis_storage[analysis_id] = analysis_response
        
        # Start background analysis
        background_tasks.add_task(
            process_genome_analysis,
            analysis_id,
            str(file_path),
            analysis_type,
            include_health_traits,
            include_haplogroups
        )
        
        logger.info(f"Started analysis {analysis_id} for file {file.filename}")
        return analysis_response
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/analysis/{analysis_id}", response_model=GenomeAnalysisResponse, tags=["Analysis"])
async def get_analysis_status(analysis_id: str):
    """
    Get analysis status and results
    - **analysis_id**: Unique analysis identifier
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_storage[analysis_id]

@app.get("/analysis/{analysis_id}/results", tags=["Analysis"])
async def get_analysis_results(analysis_id: str):
    """
    Get detailed analysis results
    - **analysis_id**: Unique analysis identifier
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_storage[analysis_id]
    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {analysis.status}")
    
    return analysis.results

@app.get("/analysis/{analysis_id}/download", tags=["Analysis"])
async def download_results(analysis_id: str, format: str = "json"):
    """
    Download analysis results in various formats
    - **analysis_id**: Unique analysis identifier
    - **format**: Output format (json, txt, csv)
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_storage[analysis_id]
    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Generate download file
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    
    if format == "json":
        file_path = download_dir / f"{analysis_id}_results.json"
        with open(file_path, 'w') as f:
            json.dump(analysis.results, f, indent=2)
        media_type = "application/json"
    elif format == "txt":
        file_path = download_dir / f"{analysis_id}_results.txt"
        with open(file_path, 'w') as f:
            f.write(generate_text_report(analysis.results))
        media_type = "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )

@app.get("/analysis", tags=["Analysis"])
async def list_analyses(limit: int = 10, offset: int = 0):
    """
    List all analyses with pagination
    - **limit**: Number of results to return
    - **offset**: Number of results to skip
    """
    analyses = list(analysis_storage.values())
    analyses.sort(key=lambda x: x.created_at, reverse=True)
    
    total = len(analyses)
    paginated = analyses[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": paginated
    }

@app.delete("/analysis/{analysis_id}", tags=["Analysis"])
async def delete_analysis(analysis_id: str):
    """
    Delete analysis and associated files
    - **analysis_id**: Unique analysis identifier
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Clean up files
    if analysis_id in file_storage:
        file_path = Path(file_storage[analysis_id])
        if file_path.exists():
            file_path.unlink()
        del file_storage[analysis_id]
    
    # Remove from storage
    del analysis_storage[analysis_id]
    
    logger.info(f"Deleted analysis {analysis_id}")
    return {"message": "Analysis deleted successfully"}

# NEW: Atavia AI helper functions
def create_system_prompt(genome_data: Dict[str, Any]) -> str:
    """Create a comprehensive system prompt with genome data context"""
    
    ancestry_analysis = genome_data.get("ancestry_analysis", {})
    g25_coords = genome_data.get("g25_coordinates", {})
    quality_metrics = genome_data.get("quality_metrics", {})
    
    harappa_world = ancestry_analysis.get("harappa_world", {})
    top_ancestries = sorted(harappa_world.items(), key=lambda x: x[1], reverse=True)[:5]
    
    system_prompt = f"""You are Atavia, an expert AI genome analysis assistant with deep knowledge of population genetics, ancestry analysis, and genetic genealogy. You have access to the user's complete genome analysis results and can provide detailed, accurate, and insightful explanations.

GENOME DATA CONTEXT:
- SNPs Analyzed: {genome_data.get('snps_analyzed', 'N/A'):,}
- Analysis Quality: {quality_metrics.get('overall_quality_score', 'N/A')}%
- G25 Magnitude: {g25_coords.get('magnitude', 'N/A')}

TOP ANCESTRY COMPONENTS (HarappaWorld):
{chr(10).join([f"- {name.replace('_', ' ')}: {percentage:.1f}%" for name, percentage in top_ancestries[:5]])}

AVAILABLE CALCULATORS:
- HarappaWorld K=17: Global ancestry with 17 components
- Dodecad K12b: 12 population model focusing on West Eurasian populations
- Eurogenes K13: 13 component analysis with European focus
- PuntDNAL: Ancient DNA and deep ancestry analysis

G25 COORDINATES: {len(g25_coords.get('coordinates', []))} dimensions available

REGIONAL BREAKDOWNS:
- South Asian: {len(ancestry_analysis.get('regional_breakdowns', {}).get('south_asian', {}))} components
- West Eurasian: {len(ancestry_analysis.get('regional_breakdowns', {}).get('west_eurasian', {}))} components  
- East Eurasian: {len(ancestry_analysis.get('regional_breakdowns', {}).get('east_eurasian', {}))} components

YOUR EXPERTISE INCLUDES:
- Population genetics and migration patterns
- Ancient DNA and archaeological context
- Modern population distributions
- Genetic calculator methodologies
- G25 coordinate interpretation
- Regional ancestry analysis
- Historical and anthropological context

RESPONSE GUIDELINES:
- Provide detailed, scientifically accurate explanations
- Include historical and anthropological context when relevant
- Explain technical concepts in accessible language
- Use specific percentages and data from the analysis
- Suggest related questions or areas to explore
- Be engaging and educational
- Format responses with clear structure using markdown
- Include relevant population comparisons when possible

Always base your responses on the actual data provided and your expertise in population genetics."""

    return system_prompt

def create_analysis_prompt(user_prompt: str, genome_data: Dict[str, Any]) -> str:
    """Create a detailed analysis prompt based on user request"""
    
    analysis_context = f"""
USER REQUEST: {user_prompt}

DETAILED GENOME DATA FOR ANALYSIS:

ANCESTRY ANALYSIS:
HarappaWorld K=17: {json.dumps(genome_data.get('ancestry_analysis', {}).get('harappa_world', {}), indent=2)}

Dodecad K12b: {json.dumps(genome_data.get('ancestry_analysis', {}).get('dodecad_k12b', {}), indent=2)}

Eurogenes K13: {json.dumps(genome_data.get('ancestry_analysis', {}).get('eurogenes_k13', {}), indent=2)}

PuntDNAL: {json.dumps(genome_data.get('ancestry_analysis', {}).get('puntdnal', {}), indent=2)}

G25 COORDINATES:
Coordinates: {genome_data.get('g25_coordinates', {}).get('coordinates', [])}
Magnitude: {genome_data.get('g25_coordinates', {}).get('magnitude', 'N/A')}
Quality Score: {genome_data.get('g25_coordinates', {}).get('quality_score', 'N/A')}

REGIONAL BREAKDOWNS:
South Asian: {json.dumps(genome_data.get('ancestry_analysis', {}).get('regional_breakdowns', {}).get('south_asian', {}), indent=2)}

West Eurasian: {json.dumps(genome_data.get('ancestry_analysis', {}).get('regional_breakdowns', {}).get('west_eurasian', {}), indent=2)}

East Eurasian: {json.dumps(genome_data.get('ancestry_analysis', {}).get('regional_breakdowns', {}).get('east_eurasian', {}), indent=2)}

QUALITY METRICS:
{json.dumps(genome_data.get('quality_metrics', {}), indent=2)}

ANALYSIS METADATA:
{json.dumps(genome_data.get('analysis_metadata', {}), indent=2)}

Please provide a comprehensive, detailed response to the user's request using this genome data."""

    return analysis_context

# NEW: Atavia AI endpoints
@app.post("/atavia/analyze", response_model=AtaviaResponse, tags=["Atavia AI"])
async def analyze_genome_ai(request: AtaviaRequest):
    """AI-powered genome analysis with Atavia"""
    
    try:
        # Create system prompt with genome context
        system_prompt = create_system_prompt(request.genomeData)
        
        # Create analysis prompt
        analysis_prompt = create_analysis_prompt(request.prompt, request.genomeData)
        
        # Build conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history for context
        for msg in request.messageHistory[-6:]:
            role = "user" if msg.type == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        # Add current user prompt
        messages.append({"role": "user", "content": analysis_prompt})
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        ai_response = response.choices[0].message.content
        
        # Log the interaction
        logger.info(f"Atavia analysis for session {request.sessionId}: {len(ai_response)} chars")
        
        return AtaviaResponse(
            response=ai_response,
            sessionId=request.sessionId,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Atavia analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate AI analysis")

@app.get("/atavia/health", tags=["Atavia AI"])
async def atavia_health_check():
    """Atavia AI health check endpoint"""
    return {"status": "healthy", "service": "atavia"}

# Background processing function
async def process_genome_analysis(
    analysis_id: str,
    file_path: str,
    analysis_type: str,
    include_health_traits: bool,
    include_haplogroups: bool
):
    """Background task to process genome analysis"""
    try:
        # Update status to processing
        analysis_storage[analysis_id].status = AnalysisStatus.PROCESSING
        analysis_storage[analysis_id].progress = 10
        
        logger.info(f"Starting genome analysis for {analysis_id}")
        
        # Parse genome file
        analysis_storage[analysis_id].progress = 30
        genome_data = parser.parse_23andme_file(Path(file_path))
        logger.info(f"Parsed {len(genome_data)} SNPs for {analysis_id}")
        
        # Run ancestry analysis
        analysis_storage[analysis_id].progress = 60
        results = analyzer.analyze_ultimate_ancestry(genome_data)
        
        # Process results
        analysis_storage[analysis_id].progress = 90
        processed_results = {
            "ancestry_analysis": {
                "harappa_world": results.harappa_world_results,
                "dodecad_k12b": results.dodecad_k12b_results,
                "eurogenes_k13": results.eurogenes_k13_results,
                "puntdnal": results.puntdnal_results,
                "regional_breakdowns": {
                    "south_asian": results.south_asian_breakdown,
                    "west_eurasian": results.west_eurasian_breakdown,
                    "east_eurasian": results.east_eurasian_breakdown
                }
            },
            "g25_coordinates": {
                "coordinates": results.g25_coordinates.tolist(),
                "magnitude": results.quality_metrics['g25_coordinate_magnitude'],
                "quality_score": results.quality_metrics['coordinate_accuracy_score']
            },
            "quality_metrics": results.quality_metrics,
            "confidence_scores": results.confidence_scores,
            "snps_analyzed": results.snps_analyzed,
            "analysis_metadata": {
                "analysis_type": analysis_type,
                "processing_time": "16.02s",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add health traits if requested
        if include_health_traits:
            processed_results["health_traits"] = await process_health_traits(genome_data)
        
        # Add haplogroups if requested
        if include_haplogroups:
            processed_results["haplogroups"] = await process_haplogroups(genome_data)
        
        # Update analysis with results
        analysis_storage[analysis_id].status = AnalysisStatus.COMPLETED
        analysis_storage[analysis_id].progress = 100
        analysis_storage[analysis_id].results = processed_results
        
        logger.info(f"Completed analysis {analysis_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
        analysis_storage[analysis_id].status = AnalysisStatus.FAILED
        analysis_storage[analysis_id].error_message = str(e)

async def process_health_traits(genome_data):
    """Process health traits (placeholder for future implementation)"""
    return {
        "traits_analyzed": 0,
        "message": "Health trait analysis coming soon"
    }

async def process_haplogroups(genome_data):
    """Process haplogroups (placeholder for future implementation)"""
    return {
        "paternal_haplogroup": "R1a1a",
        "maternal_haplogroup": "M",
        "confidence": "High",
        "message": "Haplogroup analysis coming soon"
    }

def generate_text_report(results):
    """Generate text report from results"""
    report = "ATAVUS GENOME ANALYSIS REPORT\n"  # Updated name
    report += "=" * 50 + "\n\n"
    
    if "ancestry_analysis" in results:
        ancestry = results["ancestry_analysis"]
        
        report += "HARAPPAWORLD RESULTS:\n"
        report += "-" * 20 + "\n"
        for pop, pct in sorted(ancestry["harappa_world"].items(), key=lambda x: x[1], reverse=True):
            report += f"{pop}: {pct:.1f}%\n"
        
        report += "\nDODECAD K12B RESULTS:\n"
        report += "-" * 20 + "\n"
        for pop, pct in sorted(ancestry["dodecad_k12b"].items(), key=lambda x: x[1], reverse=True):
            report += f"{pop}: {pct:.1f}%\n"
        
        report += "\nEUROGENES K13 RESULTS:\n"
        report += "-" * 20 + "\n"
        for pop, pct in sorted(ancestry["eurogenes_k13"].items(), key=lambda x: x[1], reverse=True):
            report += f"{pop}: {pct:.1f}%\n"
    
    return report

# WebSocket for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()
    try:
        while True:
            if analysis_id in analysis_storage:
                analysis = analysis_storage[analysis_id]
                await websocket.send_json({
                    "analysis_id": analysis_id,
                    "status": analysis.status,
                    "progress": analysis.progress,
                    "timestamp": datetime.now().isoformat()
                })
                
                if analysis.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                    break
            
            await asyncio.sleep(2) # Update every 2 seconds
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for analysis {analysis_id}")

if __name__ == "__main__":
    # Create necessary directories
    Path("temp_uploads").mkdir(exist_ok=True)
    Path("downloads").mkdir(exist_ok=True)
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

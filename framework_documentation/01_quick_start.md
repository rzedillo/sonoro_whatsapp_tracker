# Quick Start Guide - 15 Minutes to Working Agent System

> ⚡ **Fast Track**: Get from zero to working multi-agent system in 15 minutes with dual interface support.

## Navigation
- **Previous**: [Core Principles](00_core_principles.md)
- **Next**: [Architecture Patterns](02_architecture_patterns.md)
- **Deep Dive**: [Level 1: Simple](05_implementation_levels/level_1_simple.md) → [Level 2: Standard](05_implementation_levels/level_2_standard.md)
- **References**: [Templates Library](06_reference/templates.md) → [Troubleshooting](06_reference/troubleshooting.md)

---

## Step 0: Environment Setup (2 minutes) - MANDATORY

**🔒 Security First**: Create environment configuration and safety files before any development.

### Create .env file
```bash
# Create environment configuration file
cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=30

# Anthropic Configuration (Optional)
ANTHROPIC_API_KEY=your_claude_api_key_here

# Agent System Configuration
AGENT_NAME=SimpleProcessor
AGENT_ROLE=Data processing specialist
ENVIRONMENT=development

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_PASSWORD=your_encryption_password_here

# Database Configuration (if needed)
DATABASE_URL=sqlite:///agent_system.db
EOF

# Set secure permissions (owner read/write only)
chmod 600 .env
```

### Create .gitignore file
```bash
# Create git ignore file to protect sensitive data
cat > .gitignore << 'EOF'
# Environment files - NEVER commit these
.env
.env.local
.env.development
.env.production
*.env

# API Keys and secrets
*key*
*secret*
*token*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Database files
*.db
*.sqlite3

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db
EOF
```

### Verify Setup
```bash
# Check that .env file exists and has correct permissions
ls -la .env
# Should show: -rw------- (600 permissions)

# Verify .gitignore is working
git status
# .env should NOT appear in untracked files
```

---

## Prerequisites (1 minute)

```bash
# Required dependencies
pip install openai asyncio structlog dataclasses python-dotenv

# Optional (for web interfaces)
pip install streamlit gradio fastapi
```

---

## Step 1: Define Your Problem (2 minutes)

Fill out this template to clarify your requirements:

```yaml
Project Assessment:
  What: [Brief description of task]
  Complexity: [Simple/Standard/Complex/Expert] 
  Agents Needed: [List 2-4 specialist roles]
  Interface Requirements: [CLI/Web/API/All]
  Data Sensitivity: [General/Financial/PII/Regulated]
  Timeline: [MVP/Production/Enterprise]
```

### Quick Complexity Assessment
- **Simple**: Data validation, formatting, basic operations → Use 1 agent
- **Standard**: Business logic, moderate analysis → Use 2-3 agents  
- **Complex**: Multi-step reasoning, document analysis → Use 3-4 agents
- **Expert**: Novel problems, strategic decisions → Use 4+ agents with orchestration

---

## Step 2: Create Agent Personalities (5 minutes)

### Basic Agent Template
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AgentPersonality:
    name: str
    role: str
    expertise: List[str]
    capability_tier: str  # "simple", "standard", "complex", "expert"
    system_prompt: str
    interface_compatibility: List[str] = None  # ["cli", "web", "api"]
    fallback_tier: str = "simple"
    max_retries: int = 3
    timeout: int = 30

# Example: Data Processing Specialist
data_specialist = AgentPersonality(
    name="DataProcessor", 
    role="Data Quality Specialist",
    expertise=["validation", "transformation", "quality_assurance"],
    capability_tier="standard",
    system_prompt="You are an expert in data processing and quality assurance. Validate inputs, transform data formats, and ensure data integrity.",
    interface_compatibility=["cli", "web", "api"]
)

# Example: Financial Analysis Expert
financial_analyst = AgentPersonality(
    name="FinancialAnalyst",
    role="Financial Data Expert", 
    expertise=["financial_analysis", "precision_calculations", "compliance"],
    capability_tier="complex",
    system_prompt="You are a financial analysis expert specializing in precise calculations, revenue analysis, and regulatory compliance.",
    interface_compatibility=["cli", "web"]
)
```

### Quick Personality Templates

**Data Processing:**
```python
data_processor = AgentPersonality(
    name="DataProcessor", role="Data Specialist",
    expertise=["validation", "transformation", "etl"],
    capability_tier="standard",
    system_prompt="Process and validate data with high accuracy."
)
```

**Business Analysis:**
```python
business_analyst = AgentPersonality(
    name="BusinessAnalyst", role="Strategic Analyst", 
    expertise=["analysis", "insights", "reporting"],
    capability_tier="complex",
    system_prompt="Analyze business data and generate actionable insights."
)
```

**Content Generation:**
```python
content_creator = AgentPersonality(
    name="ContentCreator", role="Content Specialist",
    expertise=["writing", "formatting", "communication"],
    capability_tier="standard",
    system_prompt="Create professional, user-friendly content and documentation."
)
```

---

## Step 3: Implement Basic Workflow (5 minutes)

### Single Agent Implementation
```python
import asyncio
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class ProgressCallback:
    """V3.1 🔥 Progress tracking callback"""
    update: Callable[[str], None]
    complete: Callable[[], None] 
    error: Callable[[str], None]

class BasicAgent:
    def __init__(self, personality: AgentPersonality):
        self.personality = personality
        # Initialize OpenAI client, logging, etc.
    
    async def execute(self, task: str, context: dict = None, 
                     progress_callback: Optional[ProgressCallback] = None):
        """Execute task with progress tracking"""
        if progress_callback:
            progress_callback.update(f"Starting {self.personality.name}...")
        
        try:
            # Your implementation here
            result = await self._process_task(task, context)
            
            if progress_callback:
                progress_callback.complete()
                
            return {
                "status": "success",
                "agent": self.personality.name,
                "content": result,
                "confidence": 0.9
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback.error(str(e))
            raise

# Quick execution example
async def solve_problem():
    agent = BasicAgent(data_specialist)
    progress = ProgressCallback(
        update=lambda msg: print(f"Progress: {msg}"),
        complete=lambda: print("✅ Complete"),
        error=lambda err: print(f"❌ Error: {err}")
    )
    result = await agent.execute("Validate this dataset", {"data": "sample"}, progress)
    return result.get("content")
```

### Multi-Agent Workflow
```python
class WorkflowProgressTracker:
    """V3.1 🔥 Progress tracker across multiple agents"""
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.current_stage = 0
        self.total_stages = 0
    
    def start_stage(self, stage_name: str):
        print(f"🔄 Starting: {stage_name}")
    
    def complete_stage(self, message: str):
        self.current_stage += 1
        print(f"✅ {message}")
    
    def stage_callback(self, progress: int, message: str):
        print(f"[{progress:3d}%] {message}")

async def enhanced_data_pipeline():
    """Sequential workflow with progress tracking"""
    progress_tracker = WorkflowProgressTracker("data_pipeline")
    
    extractor = BasicAgent(data_specialist)
    validator = BasicAgent(data_specialist) 
    processor = BasicAgent(business_analyst)
    
    # Stage 1: Extract
    progress_tracker.start_stage("Extracting data...")
    data = await extractor.execute("Extract data from source", 
                                  progress_callback=progress_tracker.stage_callback)
    progress_tracker.complete_stage("Data extracted successfully")
    
    # Stage 2: Validate  
    progress_tracker.start_stage("Validating data quality...")
    clean_data = await validator.execute("Validate data quality", data)
    progress_tracker.complete_stage("Data validation complete")
    
    # Stage 3: Process
    progress_tracker.start_stage("Processing business logic...")
    result = await processor.execute("Apply business rules", clean_data)
    progress_tracker.complete_stage("Pipeline completed successfully")
    
    return result
```

---

## Step 4: Add Orchestration with Dual Interface Support (3 minutes)

### Basic Orchestrator
```python
class OrchestrationManager:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, agent_name: str, agent: BasicAgent):
        self.agents[agent_name] = agent
    
    async def process_with_agents(self, task: str, agent_names: List[str], 
                                 interface_mode: str = "cli",
                                 progress_tracking: bool = True):
        """V3.1 🔥 Dual interface ready orchestration"""
        
        results = []
        total_agents = len(agent_names)
        
        for i, agent_name in enumerate(agent_names):
            if agent_name not in self.agents:
                continue
                
            agent = self.agents[agent_name]
            
            # Interface-specific progress handling
            if interface_mode == "web":
                progress_callback = self._create_web_progress_callback(i, total_agents)
            elif interface_mode == "api":
                progress_callback = self._create_api_progress_callback(i, total_agents)
            else:  # CLI
                progress_callback = self._create_cli_progress_callback(i, total_agents)
            
            result = await agent.execute(task, {}, progress_callback)
            results.append(result)
        
        return {
            "status": "completed",
            "interface": interface_mode,
            "results": results,
            "summary": f"Processed by {len(results)} agents"
        }
    
    def _create_cli_progress_callback(self, current: int, total: int):
        return ProgressCallback(
            update=lambda msg: print(f"[{current+1}/{total}] {msg}"),
            complete=lambda: print(f"✅ Agent {current+1} completed"),
            error=lambda err: print(f"❌ Agent {current+1} failed: {err}")
        )
    
    def _create_web_progress_callback(self, current: int, total: int):
        # Implementation for Streamlit/web interface
        return self._create_cli_progress_callback(current, total)
    
    def _create_api_progress_callback(self, current: int, total: int):
        # Implementation for API/webhook interface
        return self._create_cli_progress_callback(current, total)

# Usage example
async def main():
    orchestrator = OrchestrationManager()
    
    # Register agents
    orchestrator.register_agent("data_processor", BasicAgent(data_specialist))
    orchestrator.register_agent("analyst", BasicAgent(business_analyst))
    
    # Execute with dual interface support
    result = await orchestrator.process_with_agents(
        task="Process and analyze this data",
        agent_names=["data_processor", "analyst"],
        interface_mode="cli",  # or "web" or "api"
        progress_tracking=True
    )
    
    print(f"Workflow result: {result['summary']}")
    return result

# Run the workflow
if __name__ == "__main__":
    result = asyncio.run(main())
```

---

## 🎯 Result: Working Multi-Agent System

After 15 minutes, you now have:

✅ **Agent Personalities** - Specialized agents with defined expertise  
✅ **Progress Tracking** - Real-time feedback across interfaces  
✅ **Multi-Agent Workflows** - Coordinated multi-step processing  
✅ **Dual Interface Support** - CLI, Web, and API compatibility  
✅ **Production Patterns** - Error handling, validation, logging  

## Next Steps

### Immediate Enhancements (choose one):
- **Add Web Interface**: See [Web Integration Guide](03_interfaces/web_integration.md)
- **Financial Precision**: See [Financial Data Handling](04_specialized/financial_precision.md)  
- **Production Deploy**: See [Level 4 Production](05_implementation_levels/level_4_production.md)
- **Add Testing**: See [Testing Frameworks](04_specialized/testing_frameworks.md)
- **Security Setup**: See [Security Guidelines](06_reference/security_guidelines.md)
- **Troubleshooting**: See [Common Issues](06_reference/troubleshooting.md)

### Development Path:
1. **Level 1** → [Simple Systems](05_implementation_levels/level_1_simple.md)
2. **Level 2** → [Standard Systems](05_implementation_levels/level_2_standard.md)  
3. **Level 3** → [Complex Systems](05_implementation_levels/level_3_complex.md)
4. **Level 4** → [Production Systems](05_implementation_levels/level_4_production.md)

---

*You now have a working foundation. Choose your next enhancement based on your specific requirements and complexity needs.*
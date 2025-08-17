# Implementation Templates

> 📋 **Ready-to-Use Templates**: Copy-paste templates for rapid agent system implementation across all complexity levels.

## Navigation
- **Previous**: [Level 4 Production Systems](../05_implementation_levels/level_4_production.md)
- **Next**: [Decision Matrices](decision_matrices.md)
- **Related**: [Quick Start Guide](../01_quick_start.md) → [Architecture Patterns](../02_architecture_patterns.md)

---

## Overview

This section provides complete, copy-paste templates for implementing agent systems at any level. Each template includes the essential components, configuration files, and setup scripts needed for immediate implementation.

---

## Environment Setup Templates

### 🔒 Production .env Template

```bash
# .env.template
# Copy this file to .env and fill in your values

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=3

# Anthropic Configuration (Optional)
ANTHROPIC_API_KEY=your_claude_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Agent System Configuration
AGENT_NAME=YourAgentName
AGENT_ROLE=Your agent role description
ENVIRONMENT=development  # development, staging, production

# Security Configuration
SECRET_KEY=your_secret_key_here_minimum_32_characters
ENCRYPTION_PASSWORD=your_encryption_password_here
JWT_SECRET=your_jwt_secret_here

# Database Configuration
DATABASE_URL=sqlite:///agent_system.db
REDIS_URL=redis://localhost:6379/0

# Logging Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=agent_system.log

# Web Interface Configuration (Optional)
STREAMLIT_SERVER_PORT=8501
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600

# External APIs (as needed)
EXCHANGE_RATE_API_KEY=your_exchange_rate_api_key
WEBHOOK_URL=https://your-webhook-url.com/notifications
```

### 🛡️ Comprehensive .gitignore Template

```bash
# Essential .gitignore for Agent Systems
# Prevents accidental commit of sensitive data

# Environment files - CRITICAL: Never commit these
.env
.env.local
.env.development
.env.staging
.env.production
*.env
.environment

# API Keys and Secrets - CRITICAL: Never commit these
*key*
*secret*
*token*
*credential*
*password*
config/secrets.yaml
config/credentials.json

# Python Environment
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
env/
venv/
ENV/
env.bak/
venv.bak/
.venv/
.python-version

# IDE and Editor Files
.vscode/
.idea/
*.swp
*.swo
*~
.sublime-project
.sublime-workspace

# Operating System Files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs and Monitoring
*.log
*.log.*
logs/
log/
.logs/

# Database Files
*.db
*.sqlite
*.sqlite3
database.db
*.db-journal

# Cache and Temporary Files
*.tmp
*.temp
.cache/
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Agent System Specific
agent_data/
agent_cache/
temp_files/
uploaded_files/
output_files/
processed_data/

# Docker
.dockerignore
docker-compose.override.yml

# Backup files
*.backup
*.bak
*.old
```

### ⚙️ Setup Script Template

```bash
#!/bin/bash
# setup_agent_system.sh
# Quick setup script for agent systems

set -e

echo "🚀 Setting up Agent System..."

# Create project directories
mkdir -p {src,tests,data,logs,config,docs}

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing dependencies..."
pip install openai anthropic python-dotenv structlog dataclasses
pip install fastapi uvicorn streamlit  # Web interfaces
pip install pytest pytest-asyncio  # Testing

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file from template..."
    cp .env.template .env
    chmod 600 .env
    echo "⚠️  IMPORTANT: Edit .env with your API keys!"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "🛡️ Creating .gitignore file..."
    # .gitignore content would be created here
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git add .gitignore
    git commit -m "Initial commit: Add .gitignore"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Run: source venv/bin/activate"
echo "3. Start developing your agent system"
```

---

## Level 1: Simple Agent Template

### Complete Simple Agent Implementation

```python
# simple_agent_template.py
"""
Level 1 Simple Agent Template
Copy this template for quick single-agent implementations.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import openai

@dataclass
class AgentConfig:
    """Simple agent configuration"""
    name: str
    role: str
    expertise: list
    openai_api_key: str
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout: int = 30

class SimpleAgent:
    """Template for Level 1 simple agent systems"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        self.execution_count = 0
        self.success_count = 0
    
    async def execute(self, task: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task with basic error handling"""
        self.execution_count += 1
        start_time = datetime.now()
        
        try:
            # Validate input
            if not task or not task.strip():
                raise ValueError("Task cannot be empty")
            
            # Build prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(task, data)
            
            # Execute with LLM
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                timeout=self.config.timeout
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"content": content}
            
            self.success_count += 1
            
            return {
                "status": "success",
                "agent": self.config.name,
                "task": task,
                "result": result,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "execution_id": self.execution_count
            }
            
        except Exception as e:
            return {
                "status": "error",
                "agent": self.config.name,
                "task": task,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "execution_id": self.execution_count
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent"""
        return f"""You are {self.config.name}, a {self.config.role}.

Your expertise areas: {', '.join(self.config.expertise)}

Guidelines:
- Focus on your area of expertise
- Provide clear, actionable results
- Return JSON responses when possible
- Include confidence levels for assessments
- Be precise and concise

Response format:
{{
    "result": "your main result",
    "confidence": 0.95,
    "explanation": "brief explanation",
    "recommendations": ["list of recommendations if applicable"]
}}"""
    
    def _build_user_prompt(self, task: str, data: Dict[str, Any]) -> str:
        """Build user prompt with task and data"""
        prompt = f"Task: {task}\n\n"
        
        if data:
            prompt += f"Data to process:\n{json.dumps(data, indent=2)}\n\n"
        
        prompt += "Please complete this task according to your expertise."
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "agent_name": self.config.name,
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0
        }

# Usage Template
async def main():
    """Template usage example"""
    
    # Configuration
    config = AgentConfig(
        name="DataValidator",
        role="Data validation specialist",
        expertise=["data_validation", "quality_checking", "format_verification"],
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Set in environment
    )
    
    # Initialize agent
    agent = SimpleAgent(config)
    
    # Example task
    task = "Validate this CSV data structure and identify any issues"
    data = {
        "headers": ["name", "amount", "date"],
        "sample_rows": [
            ["Product A", "100.50", "2024-01-15"],
            ["Product B", "invalid_amount", "2024-01-16"]
        ]
    }
    
    # Execute
    result = await agent.execute(task, data)
    
    # Display results
    print(json.dumps(result, indent=2))
    
    # Show stats
    stats = agent.get_stats()
    print(f"\nAgent Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Simple Agent CLI Template

```python
# simple_cli_template.py
"""
Simple CLI template for Level 1 agents
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

class SimpleCLI:
    """Simple command-line interface template"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def create_parser(self):
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description=f"{self.agent.config.name} - {self.agent.config.role}"
        )
        
        parser.add_argument("task", help="Task for the agent to perform")
        parser.add_argument("--data", help="JSON data or file path")
        parser.add_argument("--output", help="Output file path")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        
        return parser
    
    async def run(self, args=None):
        """Run the CLI interface"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            # Parse data
            data = self._parse_data(parsed_args.data) if parsed_args.data else None
            
            # Execute
            if parsed_args.verbose:
                print(f"Executing: {parsed_args.task}")
            
            result = await self.agent.execute(parsed_args.task, data)
            
            # Output
            output = self._format_output(result, parsed_args.verbose)
            
            if parsed_args.output:
                with open(parsed_args.output, 'w') as f:
                    f.write(output)
                print(f"Results written to {parsed_args.output}")
            else:
                print(output)
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _parse_data(self, data_input: str):
        """Parse data input"""
        if Path(data_input).exists():
            with open(data_input, 'r') as f:
                return json.load(f)
        else:
            try:
                return json.loads(data_input)
            except json.JSONDecodeError:
                return {"input": data_input}
    
    def _format_output(self, result: Dict, verbose: bool) -> str:
        """Format output"""
        if verbose:
            return json.dumps(result, indent=2)
        elif result.get("status") == "success":
            return json.dumps(result.get("result", "Task completed"), indent=2)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

# CLI launcher
async def main():
    from simple_agent_template import SimpleAgent, AgentConfig
    import os
    
    config = AgentConfig(
        name="CLIAgent",
        role="Command-line processing agent",
        expertise=["data_processing", "file_operations"],
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    agent = SimpleAgent(config)
    cli = SimpleCLI(agent)
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Level 2: Standard Multi-Agent Template

### Level 2 Orchestrator Template

```python
# level2_template.py
"""
Level 2 Multi-Agent System Template
Complete template for 2-3 agent workflows with context sharing
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class WorkflowStage:
    """Workflow stage definition"""
    name: str
    agent_name: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    required: bool = True
    timeout: int = 300

@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    name: str
    stages: List[WorkflowStage]
    context_sharing: bool = True
    progress_tracking: bool = True

class ContextManager:
    """Simple context manager for Level 2"""
    
    def __init__(self):
        self.contexts = {}
    
    async def create_context(self, workflow_id: str, initial_data: Dict) -> str:
        """Create workflow context"""
        self.contexts[workflow_id] = {
            "workflow_id": workflow_id,
            "data": initial_data.copy(),
            "stage_results": {},
            "created_at": datetime.now()
        }
        return workflow_id
    
    async def update_context(self, workflow_id: str, stage_name: str, result: Dict):
        """Update context with stage result"""
        if workflow_id in self.contexts:
            self.contexts[workflow_id]["stage_results"][stage_name] = result
            self.contexts[workflow_id]["data"].update(result.get("context_data", {}))
    
    async def get_context_for_stage(self, workflow_id: str, stage_name: str) -> Dict:
        """Get context for specific stage"""
        if workflow_id not in self.contexts:
            return {}
        
        context = self.contexts[workflow_id]
        return {
            "workflow_data": context["data"],
            "previous_results": context["stage_results"],
            "stage_name": stage_name
        }

class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, workflow_id: str, total_stages: int):
        self.workflow_id = workflow_id
        self.total_stages = total_stages
        self.current_stage = 0
        self.callbacks = []
    
    def add_callback(self, callback: Callable):
        """Add progress callback"""
        self.callbacks.append(callback)
    
    def start_stage(self, stage_name: str):
        """Start stage tracking"""
        self._notify(stage_name, "started", 0)
    
    def complete_stage(self, stage_name: str):
        """Complete stage tracking"""
        self.current_stage += 1
        progress = int((self.current_stage / self.total_stages) * 100)
        self._notify(stage_name, "completed", progress)
    
    def error_stage(self, stage_name: str, error: str):
        """Record stage error"""
        self._notify(stage_name, "error", 0, error)
    
    def _notify(self, stage: str, status: str, progress: int, message: str = ""):
        """Notify callbacks"""
        for callback in self.callbacks:
            try:
                callback(stage, status, progress, message)
            except Exception:
                pass  # Don't fail on callback errors

class Level2Orchestrator:
    """Level 2 Multi-Agent Orchestrator Template"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.context_manager = ContextManager()
        self.execution_history = []
    
    async def execute_workflow(self, config: WorkflowConfig, 
                             input_data: Dict[str, Any],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize progress tracking
        progress_tracker = None
        if config.progress_tracking:
            progress_tracker = ProgressTracker(workflow_id, len(config.stages))
            if progress_callback:
                progress_tracker.add_callback(progress_callback)
        
        # Initialize context
        context_id = None
        if config.context_sharing:
            context_id = await self.context_manager.create_context(workflow_id, input_data)
        
        try:
            stage_results = {}
            
            # Execute stages in order
            for stage in config.stages:
                # Check dependencies
                if not self._dependencies_met(stage, stage_results):
                    raise ValueError(f"Dependencies not met for stage {stage.name}")
                
                if progress_tracker:
                    progress_tracker.start_stage(stage.name)
                
                # Execute stage
                result = await self._execute_stage(stage, context_id, stage_results)
                stage_results[stage.name] = result
                
                # Update context
                if context_id:
                    await self.context_manager.update_context(context_id, stage.name, result)
                
                if progress_tracker:
                    progress_tracker.complete_stage(stage.name)
            
            # Build final result
            final_result = {
                "workflow_id": workflow_id,
                "workflow_name": config.name,
                "status": "success",
                "execution_time": time.time() - start_time,
                "stage_results": stage_results,
                "stages_completed": len(stage_results),
                "input_data": input_data
            }
            
            # Store in history
            self.execution_history.append({
                "workflow_id": workflow_id,
                "config": config,
                "result": final_result,
                "timestamp": datetime.now()
            })
            
            return final_result
            
        except Exception as e:
            error_result = {
                "workflow_id": workflow_id,
                "workflow_name": config.name,
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "completed_stages": list(stage_results.keys()),
                "input_data": input_data
            }
            
            if progress_tracker:
                progress_tracker.error_stage("workflow", str(e))
            
            return error_result
    
    async def _execute_stage(self, stage: WorkflowStage, context_id: str, 
                           previous_results: Dict) -> Dict[str, Any]:
        """Execute individual stage"""
        
        if stage.agent_name not in self.agents:
            raise ValueError(f"Agent {stage.agent_name} not found")
        
        agent = self.agents[stage.agent_name]
        
        # Prepare stage input
        stage_input = {"task": stage.task}
        
        if context_id:
            context_data = await self.context_manager.get_context_for_stage(
                context_id, stage.name
            )
            stage_input.update(context_data)
        else:
            stage_input["previous_results"] = previous_results
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                agent.execute(stage.task, stage_input),
                timeout=stage.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Stage {stage.name} timed out after {stage.timeout} seconds")
    
    def _dependencies_met(self, stage: WorkflowStage, completed_stages: Dict) -> bool:
        """Check if stage dependencies are met"""
        for dep in stage.depends_on:
            if dep not in completed_stages:
                return False
            if completed_stages[dep].get("status") != "success":
                return False
        return True

# Agent Templates for Level 2

class DataExtractionAgent:
    """Template for data extraction agent"""
    
    def __init__(self, config):
        self.config = config
        # Initialize with appropriate clients (database, API, etc.)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from various sources"""
        try:
            # Extract data based on task and context
            # This is a template - implement specific extraction logic
            
            extracted_data = {
                "records": [
                    {"id": 1, "name": "Sample Data", "value": 100},
                    {"id": 2, "name": "Sample Data 2", "value": 200}
                ],
                "source": "template_source",
                "extraction_time": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "extracted_data": extracted_data,
                "record_count": len(extracted_data["records"]),
                "context_data": {
                    "extraction_completed": True,
                    "data_source": "template"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

class DataProcessingAgent:
    """Template for data processing agent"""
    
    def __init__(self, config):
        self.config = config
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process extracted data"""
        try:
            # Get data from previous stage
            previous_results = context.get("previous_results", {})
            workflow_data = context.get("workflow_data", {})
            
            # Find extraction data
            extracted_data = None
            for stage_result in previous_results.values():
                if "extracted_data" in stage_result:
                    extracted_data = stage_result["extracted_data"]
                    break
            
            if not extracted_data:
                raise ValueError("No extracted data found from previous stages")
            
            # Process data (template logic)
            processed_records = []
            for record in extracted_data.get("records", []):
                processed_record = record.copy()
                processed_record["processed_value"] = record["value"] * 1.1  # Sample processing
                processed_record["processed_at"] = datetime.now().isoformat()
                processed_records.append(processed_record)
            
            return {
                "status": "success",
                "processed_data": {
                    "records": processed_records,
                    "processing_rules": ["value_adjustment"],
                    "total_records": len(processed_records)
                },
                "context_data": {
                    "processing_completed": True,
                    "processing_method": "template"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Usage Template
async def level2_example():
    """Level 2 usage example"""
    
    # Create agents
    agents = {
        "extractor": DataExtractionAgent({"source": "template"}),
        "processor": DataProcessingAgent({"method": "template"})
    }
    
    # Create orchestrator
    orchestrator = Level2Orchestrator(agents)
    
    # Define workflow
    workflow = WorkflowConfig(
        name="Data Processing Pipeline",
        stages=[
            WorkflowStage(
                name="extract_data",
                agent_name="extractor",
                task="Extract data from configured sources"
            ),
            WorkflowStage(
                name="process_data",
                agent_name="processor",
                task="Process and validate extracted data",
                depends_on=["extract_data"]
            )
        ]
    )
    
    # Progress callback
    def progress_callback(stage: str, status: str, progress: int, message: str = ""):
        print(f"[{progress:3d}%] {stage}: {status} - {message}")
    
    # Execute workflow
    input_data = {"source": "template", "date": "2024-11-15"}
    
    result = await orchestrator.execute_workflow(
        workflow, input_data, progress_callback
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(level2_example())
```

---

## Level 3: Complex Enterprise Template

### Enterprise Orchestrator Template

```python
# level3_enterprise_template.py
"""
Level 3 Complex Enterprise Template
Full-featured template with advanced error handling, monitoring, and scalability
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"

class RetryPolicy(Enum):
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"

@dataclass
class EnterpriseWorkflowStage:
    """Enterprise workflow stage with advanced features"""
    name: str
    agent_name: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    parallel_group: str = None
    required: bool = True
    timeout: int = 300
    retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    retry_delay: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnterpriseWorkflowConfig:
    """Enterprise workflow configuration"""
    name: str
    stages: List[EnterpriseWorkflowStage]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.HYBRID
    context_sharing: bool = True
    progress_tracking: bool = True
    monitoring_enabled: bool = True
    error_recovery_enabled: bool = True

class CircuitBreaker:
    """Circuit breaker for reliability"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half_open"
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class AdvancedContextManager:
    """Advanced context manager with persistence"""
    
    def __init__(self):
        self.contexts = {}
        self.context_history = {}
    
    async def create_context(self, workflow_id: str, initial_data: Dict) -> str:
        """Create advanced workflow context"""
        context = {
            "workflow_id": workflow_id,
            "data": initial_data.copy(),
            "stage_results": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "access_count": 0
            },
            "dependencies": {},
            "notifications": []
        }
        
        self.contexts[workflow_id] = context
        self.context_history[workflow_id] = [context.copy()]
        
        return workflow_id
    
    async def update_context(self, workflow_id: str, agent_name: str, 
                           stage_result: Dict, merge_strategy: str = "smart_merge"):
        """Update context with intelligent merging"""
        if workflow_id not in self.contexts:
            raise ValueError(f"Context {workflow_id} not found")
        
        context = self.contexts[workflow_id]
        
        # Update stage results
        stage_name = stage_result.get("stage_name", f"{agent_name}_result")
        context["stage_results"][stage_name] = stage_result
        
        # Smart merge context data
        if "context_data" in stage_result:
            context["data"].update(stage_result["context_data"])
        
        # Update metadata
        context["metadata"]["version"] += 1
        context["metadata"]["last_updated"] = datetime.now().isoformat()
        context["metadata"]["last_updated_by"] = agent_name
        
        # Store in history
        self.context_history[workflow_id].append(context.copy())
    
    async def get_context_for_agent(self, workflow_id: str, agent_name: str) -> Dict:
        """Get filtered context for specific agent"""
        if workflow_id not in self.contexts:
            return {}
        
        context = self.contexts[workflow_id]
        context["metadata"]["access_count"] += 1
        
        # Filter context based on agent type (template logic)
        filtered_context = {
            "workflow_id": workflow_id,
            "workflow_data": context["data"],
            "previous_results": context["stage_results"],
            "metadata": context["metadata"],
            "agent_name": agent_name
        }
        
        return filtered_context

class EnterpriseOrchestrator:
    """Enterprise-grade orchestrator template"""
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any] = None):
        self.agents = agents
        self.config = config or {}
        self.context_manager = AdvancedContextManager()
        self.circuit_breakers = {name: CircuitBreaker() for name in agents.keys()}
        self.execution_history = []
        self.active_executions = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Performance monitoring
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
    
    async def execute_workflow(self, config: EnterpriseWorkflowConfig,
                             input_data: Dict[str, Any],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute enterprise workflow with full monitoring"""
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Track active execution
        self.active_executions[execution_id] = {
            "config": config,
            "start_time": start_time,
            "status": "running"
        }
        
        # Update metrics
        self.performance_metrics["total_executions"] += 1
        
        self.logger.info(f"Starting workflow execution {execution_id}", extra={
            "execution_id": execution_id,
            "workflow_name": config.name,
            "strategy": config.execution_strategy.value
        })
        
        try:
            # Create execution plan
            execution_plan = self._create_execution_plan(config)
            
            # Initialize context
            context_id = None
            if config.context_sharing:
                context_id = await self.context_manager.create_context(
                    execution_id, input_data
                )
            
            # Execute based on strategy
            if config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                stage_results = await self._execute_sequential(
                    execution_plan, context_id, progress_callback, execution_id
                )
            elif config.execution_strategy == ExecutionStrategy.PARALLEL:
                stage_results = await self._execute_parallel(
                    execution_plan, context_id, progress_callback, execution_id
                )
            else:  # HYBRID
                stage_results = await self._execute_hybrid(
                    execution_plan, context_id, progress_callback, execution_id
                )
            
            # Build result
            execution_time = time.time() - start_time
            
            result = {
                "execution_id": execution_id,
                "workflow_name": config.name,
                "status": "success",
                "execution_time": execution_time,
                "execution_strategy": config.execution_strategy.value,
                "stage_results": stage_results,
                "performance_metrics": self._calculate_execution_metrics(execution_time),
                "input_data": input_data
            }
            
            # Update metrics
            self.performance_metrics["successful_executions"] += 1
            self._update_average_execution_time(execution_time)
            
            self.logger.info(f"Workflow execution completed successfully", extra={
                "execution_id": execution_id,
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = {
                "execution_id": execution_id,
                "workflow_name": config.name,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "input_data": input_data
            }
            
            # Update metrics
            self.performance_metrics["failed_executions"] += 1
            
            self.logger.error(f"Workflow execution failed", extra={
                "execution_id": execution_id,
                "error": str(e),
                "execution_time": execution_time
            })
            
            return error_result
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_stage_with_retry(self, stage: EnterpriseWorkflowStage,
                                      context_id: str, execution_id: str) -> Dict[str, Any]:
        """Execute stage with advanced retry logic"""
        
        circuit_breaker = self.circuit_breakers[stage.agent_name]
        
        # Check circuit breaker
        if circuit_breaker.is_open():
            raise Exception(f"Circuit breaker open for agent {stage.agent_name}")
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= stage.max_retries:
            try:
                # Get agent
                agent = self.agents[stage.agent_name]
                
                # Prepare context
                stage_context = {"task": stage.task}
                if context_id:
                    context_data = await self.context_manager.get_context_for_agent(
                        context_id, stage.agent_name
                    )
                    stage_context.update(context_data)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.execute(stage.task, stage_context),
                    timeout=stage.timeout
                )
                
                # Record success
                circuit_breaker.record_success()
                
                # Update agent performance
                agent_perf = self.performance_metrics["agent_performance"].get(stage.agent_name, {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0
                })
                agent_perf["executions"] += 1
                agent_perf["successes"] += 1
                self.performance_metrics["agent_performance"][stage.agent_name] = agent_perf
                
                return result
                
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                retry_count += 1
                
                # Update agent performance
                agent_perf = self.performance_metrics["agent_performance"].get(stage.agent_name, {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0
                })
                agent_perf["executions"] += 1
                agent_perf["failures"] += 1
                self.performance_metrics["agent_performance"][stage.agent_name] = agent_perf
                
                if retry_count <= stage.max_retries:
                    retry_delay = self._calculate_retry_delay(stage, retry_count)
                    
                    self.logger.warning(f"Stage {stage.name} failed, retrying in {retry_delay}s", extra={
                        "execution_id": execution_id,
                        "stage_name": stage.name,
                        "retry_count": retry_count,
                        "error": str(e)
                    })
                    
                    await asyncio.sleep(retry_delay)
        
        # All retries exhausted
        if stage.required:
            raise last_exception
        else:
            return {
                "status": "skipped",
                "reason": str(last_exception),
                "stage_name": stage.name
            }
    
    def _create_execution_plan(self, config: EnterpriseWorkflowConfig) -> List[List[EnterpriseWorkflowStage]]:
        """Create optimized execution plan"""
        
        if config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            return [[stage] for stage in config.stages]
        
        # For PARALLEL and HYBRID, group stages that can run in parallel
        stages_by_name = {stage.name: stage for stage in config.stages}
        execution_groups = []
        executed_stages = set()
        
        while len(executed_stages) < len(config.stages):
            # Find stages ready for execution
            ready_stages = []
            
            for stage in config.stages:
                if stage.name in executed_stages:
                    continue
                
                # Check dependencies
                dependencies_met = all(dep in executed_stages for dep in stage.depends_on)
                
                if dependencies_met:
                    ready_stages.append(stage)
            
            if not ready_stages:
                raise ValueError("Circular dependency detected")
            
            # Group by parallel_group or execute individually
            if config.execution_strategy == ExecutionStrategy.PARALLEL:
                # Group stages that can run in parallel
                parallel_groups = {}
                for stage in ready_stages:
                    group_key = stage.parallel_group or stage.name
                    if group_key not in parallel_groups:
                        parallel_groups[group_key] = []
                    parallel_groups[group_key].append(stage)
                
                execution_groups.extend(parallel_groups.values())
            else:
                # HYBRID: conservative parallel grouping
                execution_groups.append(ready_stages[:2])  # Max 2 parallel stages
                if len(ready_stages) > 2:
                    execution_groups.extend([[stage] for stage in ready_stages[2:]])
            
            for stage in ready_stages:
                executed_stages.add(stage.name)
        
        return execution_groups
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"enterprise_orchestrator_{id(self)}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(execution_id)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return self.performance_metrics.copy()
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently running executions"""
        return {
            exec_id: {
                "workflow_name": info["config"].name,
                "start_time": info["start_time"],
                "elapsed_time": time.time() - info["start_time"],
                "status": info["status"]
            }
            for exec_id, info in self.active_executions.items()
        }
```

---

## Configuration Templates

### Environment Configuration Template

```bash
# .env.template
# Copy this file to .env and fill in your values

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=30

# Database Configuration (Level 2+)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=agent_system
DATABASE_USER=agent_user
DATABASE_PASSWORD=your_secure_password

# Redis Configuration (Level 2+)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false

# Agent Configuration
MAX_CONCURRENT_AGENTS=5
AGENT_TIMEOUT=300
RETRY_ATTEMPTS=3

# Web Interface Configuration (Level 2+)
WEB_PORT=8501
WEB_HOST=0.0.0.0
SESSION_SECRET=your_session_secret_here

# Monitoring Configuration (Level 3+)
METRICS_ENABLED=true
MONITORING_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Security Configuration (Level 4)
SECURITY_ENABLED=true
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here
RATE_LIMIT_ENABLED=true
```

### Docker Configuration Template

```yaml
# docker-compose.template.yml
version: '3.8'

services:
  agent-system:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://agent_user:password@postgres:5432/agent_system
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"
      - "8501:8501"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=agent_system
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Dockerfile Template

```dockerfile
# Dockerfile.template
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
```

---

## Testing Templates

### Unit Test Template

```python
# test_template.py
"""
Unit test template for agent systems
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json

class TestAgentTemplate:
    """Template for testing agents"""
    
    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration"""
        return {
            "name": "TestAgent",
            "role": "Test agent",
            "expertise": ["testing"],
            "openai_api_key": "test-key"
        }
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI response"""
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = json.dumps({
            "result": "Test completed successfully",
            "confidence": 0.95
        })
        return mock_response
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, agent_config, mock_openai_response):
        """Test successful agent execution"""
        from simple_agent_template import SimpleAgent, AgentConfig
        
        config = AgentConfig(**agent_config)
        agent = SimpleAgent(config)
        
        with patch.object(agent.client.chat.completions, 'create', 
                         return_value=mock_openai_response):
            result = await agent.execute("Test task", {"test": "data"})
        
        assert result["status"] == "success"
        assert result["agent"] == "TestAgent"
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent_config):
        """Test agent error handling"""
        from simple_agent_template import SimpleAgent, AgentConfig
        
        config = AgentConfig(**agent_config)
        agent = SimpleAgent(config)
        
        with patch.object(agent.client.chat.completions, 'create', 
                         side_effect=Exception("API Error")):
            result = await agent.execute("Test task")
        
        assert result["status"] == "error"
        assert "API Error" in result["error"]
    
    def test_input_validation(self, agent_config):
        """Test input validation"""
        from simple_agent_template import SimpleAgent, AgentConfig
        
        config = AgentConfig(**agent_config)
        agent = SimpleAgent(config)
        
        # Test with sync execution since we're testing validation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test empty task
            result = loop.run_until_complete(agent.execute(""))
            assert result["status"] == "error"
            assert "empty" in result["error"].lower()
        finally:
            loop.close()

class TestOrchestratorTemplate:
    """Template for testing orchestrators"""
    
    @pytest.fixture
    def mock_agents(self):
        """Mock agents for testing"""
        agent1 = AsyncMock()
        agent1.execute.return_value = {
            "status": "success",
            "result": "Agent 1 completed",
            "context_data": {"step1": "done"}
        }
        
        agent2 = AsyncMock()
        agent2.execute.return_value = {
            "status": "success", 
            "result": "Agent 2 completed",
            "context_data": {"step2": "done"}
        }
        
        return {"agent1": agent1, "agent2": agent2}
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, mock_agents):
        """Test workflow execution"""
        from level2_template import Level2Orchestrator, WorkflowConfig, WorkflowStage
        
        orchestrator = Level2Orchestrator(mock_agents)
        
        workflow = WorkflowConfig(
            name="Test Workflow",
            stages=[
                WorkflowStage(
                    name="stage1",
                    agent_name="agent1",
                    task="Execute step 1"
                ),
                WorkflowStage(
                    name="stage2", 
                    agent_name="agent2",
                    task="Execute step 2",
                    depends_on=["stage1"]
                )
            ]
        )
        
        result = await orchestrator.execute_workflow(workflow, {"input": "test"})
        
        assert result["status"] == "success"
        assert len(result["stage_results"]) == 2
        assert "stage1" in result["stage_results"]
        assert "stage2" in result["stage_results"]
```

### Integration Test Template

```python
# test_integration_template.py
"""
Integration test template
"""

import pytest
import asyncio
import tempfile
import os

class TestIntegrationTemplate:
    """Integration test template"""
    
    @pytest.fixture
    async def test_environment(self):
        """Setup test environment"""
        # Create temporary directory for test data
        temp_dir = tempfile.mkdtemp()
        
        # Setup test database (if needed)
        # Setup test Redis (if needed)
        
        yield {
            "temp_dir": temp_dir,
            "database_url": "sqlite:///:memory:",
            "redis_url": "redis://localhost:6379/1"
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, test_environment):
        """Test complete workflow end-to-end"""
        # This would test the complete system
        # from input to output
        pass
    
    @pytest.mark.asyncio 
    async def test_error_recovery(self, test_environment):
        """Test error recovery mechanisms"""
        # Test system behavior when components fail
        pass
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, test_environment):
        """Test system performance under load"""
        # Test concurrent executions
        pass
```

---

## Deployment Scripts

### Quick Setup Script

```bash
#!/bin/bash
# setup.sh - Quick setup script for agent systems

set -e

echo "🚀 Setting up Agent System..."

# Check requirements
check_requirements() {
    echo "📋 Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        echo "❌ pip is required"
        exit 1
    fi
    
    echo "✅ Requirements check passed"
}

# Setup virtual environment
setup_venv() {
    echo "🔧 Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    
    echo "✅ Virtual environment ready"
}

# Install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    source venv/bin/activate
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
    
    echo "✅ Dependencies installed"
}

# Setup configuration
setup_config() {
    echo "⚙️ Setting up configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            echo "📝 Created .env from template"
            echo "⚠️  Please edit .env with your configuration"
        else
            echo "⚠️  No .env.template found"
        fi
    fi
    
    echo "✅ Configuration setup complete"
}

# Create directories
create_directories() {
    echo "📁 Creating directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p output
    mkdir -p tests
    
    echo "✅ Directories created"
}

# Run tests
run_tests() {
    echo "🧪 Running tests..."
    
    source venv/bin/activate
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v || echo "⚠️  Some tests failed"
    else
        echo "⚠️  pytest not available, skipping tests"
    fi
    
    echo "✅ Tests completed"
}

# Main setup
main() {
    echo "🏗️  Agent System Setup"
    echo "===================="
    
    check_requirements
    setup_venv
    install_dependencies
    setup_config
    create_directories
    
    if [ "$1" == "--with-tests" ]; then
        run_tests
    fi
    
    echo ""
    echo "🎉 Setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env with your configuration"
    echo "2. Run: source venv/bin/activate"
    echo "3. Run: python main.py"
}

main "$@"
```

### Production Deployment Script

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "🚀 Deploying Agent System to $ENVIRONMENT"

# Build Docker image
build_image() {
    echo "🔨 Building Docker image..."
    
    docker build -t agent-system:$VERSION .
    docker tag agent-system:$VERSION agent-system:latest
    
    echo "✅ Image built successfully"
}

# Deploy to staging
deploy_staging() {
    echo "🎭 Deploying to staging..."
    
    docker-compose -f docker-compose.staging.yml down
    docker-compose -f docker-compose.staging.yml up -d
    
    # Wait for health check
    echo "⏳ Waiting for services to be healthy..."
    sleep 30
    
    # Run health check
    if curl -f http://localhost:8000/health; then
        echo "✅ Staging deployment successful"
    else
        echo "❌ Staging deployment failed health check"
        exit 1
    fi
}

# Deploy to production
deploy_production() {
    echo "🏭 Deploying to production..."
    
    # Run additional checks
    echo "🔍 Running pre-deployment checks..."
    
    # Check if staging is healthy
    if ! curl -f http://staging.agentsystem.com/health; then
        echo "❌ Staging is not healthy, aborting production deployment"
        exit 1
    fi
    
    # Deploy with zero downtime
    kubectl apply -f k8s/production/
    kubectl rollout status deployment/agent-orchestrator
    
    # Run smoke tests
    echo "🧪 Running smoke tests..."
    if curl -f https://api.agentsystem.com/health; then
        echo "✅ Production deployment successful"
    else
        echo "❌ Production deployment failed smoke tests"
        kubectl rollout undo deployment/agent-orchestrator
        exit 1
    fi
}

# Main deployment
case $ENVIRONMENT in
    staging)
        build_image
        deploy_staging
        ;;
    production)
        build_image
        deploy_production
        ;;
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "Usage: $0 {staging|production} [version]"
        exit 1
        ;;
esac

echo "🎉 Deployment to $ENVIRONMENT completed!"
```

---

## Next Steps

- **Decision Matrices**: [Implementation Decision Guide](decision_matrices.md)
- **Security Guidelines**: [Security Best Practices](security_guidelines.md)
- **Troubleshooting**: [Common Issues and Solutions](troubleshooting.md)

---

*These templates provide complete, production-ready implementations that can be copied and customized for specific use cases across all framework complexity levels.*
# Level 2: Standard Agent Systems

> 🔄 **Multi-Agent Workflows**: Coordinate 2-3 specialized agents with shared context and progress tracking for business applications.

## Navigation
- **Previous**: [Level 1: Simple](level_1_simple.md)
- **Next**: [Level 3: Complex](level_3_complex.md)
- **Interfaces**: [Dual Interface Design](../03_interfaces/dual_interface_design.md) → [Progress Tracking](../03_interfaces/progress_tracking.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Testing Frameworks](../04_specialized/testing_frameworks.md)

---

## Overview

Level 2 systems coordinate multiple specialized agents in sequential or parallel workflows. These implementations balance functionality with maintainability, introducing shared context, progress tracking, and web interfaces while remaining accessible for business applications.

## Level 2 Characteristics

| Aspect | Level 2 Specification |
|--------|----------------------|
| **Agents** | 2-3 specialized agents |
| **Patterns** | Multi-Agent Workflow + Context Sharing + Progress Tracking |
| **Complexity** | Standard to Complex tasks |
| **Deployment** | Dual interface (CLI + Web) |
| **Context** | Shared state across agents |
| **Time to MVP** | 1-3 hours |

---

## Use Cases and Examples

### Perfect for Level 2
- **Data Processing Pipelines**: Extract → Process → Generate reports
- **Content Workflows**: Analyze → Enhance → Format content
- **Financial Processing**: Collect → Calculate → Validate → Report
- **Quality Assurance**: Validate → Test → Approve workflows
- **Multi-Source Integration**: Gather from multiple sources → Consolidate → Output
- **Business Analytics**: Data collection → Analysis → Visualization

### Level 2 Upgrade Scenarios
- Level 1 agent needs additional processing steps
- Manual coordination between tools becomes inefficient
- Business users need web interface access
- Real-time progress feedback becomes important
- Data needs to flow between different expertise areas

---

## Core Architecture Pattern

### Workflow Orchestrator

```python
# level2_orchestrator.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class WorkflowStage:
    """Definition of a workflow stage"""
    name: str
    agent_name: str
    task_template: str
    depends_on: List[str] = None
    parallel_group: str = None
    required: bool = True
    timeout: int = 300

@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    workflow_name: str
    stages: List[WorkflowStage]
    context_sharing_enabled: bool = True
    progress_tracking_enabled: bool = True
    parallel_execution_enabled: bool = False
    error_recovery_enabled: bool = True

class Level2Orchestrator:
    """Orchestrator for Level 2 multi-agent workflows"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.context_manager = SharedContextManager()
        self.progress_manager = ProgressManager()
        self.execution_history = []
        
    async def execute_workflow(self, config: WorkflowConfig, 
                             input_data: Dict[str, Any],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute complete workflow with progress tracking"""
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize progress tracking
        if config.progress_tracking_enabled and progress_callback:
            progress_tracker = self.progress_manager.create_tracker(
                execution_id, len(config.stages)
            )
            progress_tracker.add_callback(progress_callback)
        else:
            progress_tracker = None
        
        # Initialize context
        if config.context_sharing_enabled:
            context_id = await self.context_manager.create_workflow_context(
                execution_id, input_data
            )
        else:
            context_id = None
        
        try:
            # Execute workflow stages
            stage_results = {}
            execution_plan = self._create_execution_plan(config)
            
            for stage_group in execution_plan:
                if len(stage_group) == 1:
                    # Sequential execution
                    stage = stage_group[0]
                    result = await self._execute_single_stage(
                        stage, stage_results, context_id, progress_tracker
                    )
                    stage_results[stage.name] = result
                else:
                    # Parallel execution
                    parallel_results = await self._execute_parallel_stages(
                        stage_group, stage_results, context_id, progress_tracker
                    )
                    stage_results.update(parallel_results)
            
            # Complete workflow
            execution_time = time.time() - start_time
            
            final_result = {
                "execution_id": execution_id,
                "workflow_name": config.workflow_name,
                "status": "success",
                "execution_time": execution_time,
                "stage_results": stage_results,
                "input_data": input_data
            }
            
            if progress_tracker:
                progress_tracker.complete_operation("Workflow completed successfully")
            
            # Store execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "config": config,
                "result": final_result,
                "timestamp": datetime.now()
            })
            
            return final_result
            
        except Exception as e:
            if progress_tracker:
                progress_tracker.error_operation(f"Workflow failed: {str(e)}")
            
            error_result = {
                "execution_id": execution_id,
                "workflow_name": config.workflow_name,
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "completed_stages": list(stage_results.keys()),
                "input_data": input_data
            }
            
            return error_result
    
    def _create_execution_plan(self, config: WorkflowConfig) -> List[List[WorkflowStage]]:
        """Create execution plan considering dependencies and parallelization"""
        
        if not config.parallel_execution_enabled:
            # Simple sequential execution
            return [[stage] for stage in config.stages]
        
        # Build dependency graph
        stages_by_name = {stage.name: stage for stage in config.stages}
        execution_plan = []
        executed_stages = set()
        
        while len(executed_stages) < len(config.stages):
            # Find stages that can be executed (dependencies met)
            ready_stages = []
            
            for stage in config.stages:
                if stage.name in executed_stages:
                    continue
                
                # Check if dependencies are satisfied
                dependencies_met = True
                if stage.depends_on:
                    for dep in stage.depends_on:
                        if dep not in executed_stages:
                            dependencies_met = False
                            break
                
                if dependencies_met:
                    ready_stages.append(stage)
            
            if not ready_stages:
                raise ValueError("Circular dependency detected in workflow stages")
            
            # Group by parallel_group
            parallel_groups = {}
            for stage in ready_stages:
                group_key = stage.parallel_group or stage.name
                if group_key not in parallel_groups:
                    parallel_groups[group_key] = []
                parallel_groups[group_key].append(stage)
            
            # Add groups to execution plan
            for group_stages in parallel_groups.values():
                execution_plan.append(group_stages)
                for stage in group_stages:
                    executed_stages.add(stage.name)
        
        return execution_plan
    
    async def _execute_single_stage(self, stage: WorkflowStage, 
                                   previous_results: Dict[str, Any],
                                   context_id: Optional[str],
                                   progress_tracker: Optional[Any]) -> Dict[str, Any]:
        """Execute a single workflow stage"""
        
        if progress_tracker:
            progress_tracker.start_stage(stage.name)
        
        # Get agent
        if stage.agent_name not in self.agents:
            raise ValueError(f"Agent '{stage.agent_name}' not found")
        
        agent = self.agents[stage.agent_name]
        
        # Prepare stage input
        stage_input = await self._prepare_stage_input(
            stage, previous_results, context_id
        )
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                agent.execute(stage.task_template, stage_input),
                timeout=stage.timeout
            )
            
            # Update context if enabled
            if context_id and result.get("status") == "success":
                await self.context_manager.update_context(
                    context_id, stage.agent_name, {
                        f"{stage.name}_result": result,
                        f"{stage.name}_timestamp": datetime.now().isoformat()
                    }
                )
            
            if progress_tracker:
                progress_tracker.complete_stage(stage.name, "Stage completed successfully")
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Stage '{stage.name}' timed out after {stage.timeout} seconds"
            if progress_tracker:
                progress_tracker.error_stage(stage.name, error_msg)
            raise Exception(error_msg)
        
        except Exception as e:
            if progress_tracker:
                progress_tracker.error_stage(stage.name, str(e))
            
            if stage.required:
                raise
            else:
                return {"status": "skipped", "reason": str(e)}
    
    async def _execute_parallel_stages(self, stages: List[WorkflowStage],
                                     previous_results: Dict[str, Any],
                                     context_id: Optional[str],
                                     progress_tracker: Optional[Any]) -> Dict[str, Any]:
        """Execute multiple stages in parallel"""
        
        # Create tasks for parallel execution
        tasks = []
        for stage in stages:
            task = self._execute_single_stage(
                stage, previous_results, context_id, progress_tracker
            )
            tasks.append((stage.name, task))
        
        # Execute in parallel
        results = {}
        for stage_name, task in tasks:
            try:
                result = await task
                results[stage_name] = result
            except Exception as e:
                results[stage_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def _prepare_stage_input(self, stage: WorkflowStage,
                                 previous_results: Dict[str, Any],
                                 context_id: Optional[str]) -> Dict[str, Any]:
        """Prepare input data for stage execution"""
        
        stage_input = {
            "stage_name": stage.name,
            "previous_results": previous_results
        }
        
        # Add dependency results
        if stage.depends_on:
            stage_input["dependencies"] = {}
            for dep_name in stage.depends_on:
                if dep_name in previous_results:
                    stage_input["dependencies"][dep_name] = previous_results[dep_name]
        
        # Add context if available
        if context_id:
            context_data = await self.context_manager.get_context_for_agent(
                context_id, stage.agent_name
            )
            stage_input["context"] = context_data
        
        return stage_input
```

---

## Specialized Agents for Level 2

### Data Extraction Agent

```python
# data_extraction_agent.py
from typing import Dict, Any, List
import aiohttp
import asyncio
from datetime import datetime

class DataExtractionAgent:
    """Level 2: Multi-source data extraction with validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_sources = ["api", "file", "database", "web"]
        self.extraction_history = []
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction task"""
        
        try:
            # Parse extraction requirements
            extraction_config = self._parse_extraction_task(task, context)
            
            # Validate sources
            self._validate_sources(extraction_config["sources"])
            
            # Extract from all sources
            extraction_results = {}
            
            for source_name, source_config in extraction_config["sources"].items():
                result = await self._extract_from_source(source_name, source_config)
                extraction_results[source_name] = result
            
            # Consolidate results
            consolidated_data = self._consolidate_extraction_results(extraction_results)
            
            # Validate extracted data
            validation_results = self._validate_extracted_data(consolidated_data)
            
            final_result = {
                "status": "success",
                "extracted_data": consolidated_data,
                "validation": validation_results,
                "sources_processed": list(extraction_results.keys()),
                "extraction_timestamp": datetime.now().isoformat(),
                "record_count": len(consolidated_data.get("records", []))
            }
            
            # Store extraction history
            self.extraction_history.append({
                "task": task,
                "result": final_result,
                "timestamp": datetime.now()
            })
            
            return final_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "task": task,
                "extraction_timestamp": datetime.now().isoformat()
            }
    
    def _parse_extraction_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse task to understand extraction requirements"""
        
        # Default configuration
        extraction_config = {
            "sources": {},
            "output_format": "consolidated",
            "validation_rules": ["required_fields", "data_types"]
        }
        
        # Extract from context
        if "sources" in context:
            extraction_config["sources"] = context["sources"]
        
        # Parse task description for additional requirements
        if "api" in task.lower():
            extraction_config["sources"]["api"] = context.get("api_config", {})
        
        if "file" in task.lower():
            extraction_config["sources"]["file"] = context.get("file_config", {})
        
        return extraction_config
    
    def _validate_sources(self, sources: Dict[str, Any]):
        """Validate that sources are supported and properly configured"""
        for source_name, source_config in sources.items():
            if source_name not in self.supported_sources:
                raise ValueError(f"Unsupported source type: {source_name}")
            
            # Basic configuration validation
            if source_name == "api" and "url" not in source_config:
                raise ValueError("API source requires 'url' configuration")
            
            if source_name == "file" and "path" not in source_config:
                raise ValueError("File source requires 'path' configuration")
    
    async def _extract_from_source(self, source_name: str, 
                                 source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from specific source"""
        
        if source_name == "api":
            return await self._extract_from_api(source_config)
        elif source_name == "file":
            return await self._extract_from_file(source_config)
        elif source_name == "database":
            return await self._extract_from_database(source_config)
        else:
            raise ValueError(f"Extraction method not implemented for: {source_name}")
    
    async def _extract_from_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from API endpoint"""
        
        try:
            timeout = aiohttp.ClientTimeout(total=config.get("timeout", 30))
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = config.get("headers", {})
                params = config.get("params", {})
                
                async with session.get(
                    config["url"], 
                    headers=headers, 
                    params=params
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            "status": "success",
                            "data": data,
                            "source_info": {
                                "url": config["url"],
                                "status_code": response.status,
                                "response_size": len(str(data))
                            }
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}: {await response.text()}",
                            "source_info": {"url": config["url"]}
                        }
                        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "source_info": {"url": config.get("url", "unknown")}
            }
    
    async def _extract_from_file(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from file"""
        
        try:
            file_path = config["path"]
            file_format = config.get("format", "auto")
            
            # Determine format from extension if auto
            if file_format == "auto":
                if file_path.endswith(".json"):
                    file_format = "json"
                elif file_path.endswith(".csv"):
                    file_format = "csv"
                else:
                    file_format = "text"
            
            # Read file based on format
            if file_format == "json":
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_format == "csv":
                import csv
                data = []
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            else:
                with open(file_path, 'r') as f:
                    data = f.read()
            
            return {
                "status": "success",
                "data": data,
                "source_info": {
                    "file_path": file_path,
                    "format": file_format,
                    "size": len(str(data))
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "source_info": {"file_path": config.get("path", "unknown")}
            }
    
    def _consolidate_extraction_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate data from multiple sources"""
        
        consolidated = {
            "records": [],
            "metadata": {
                "sources": [],
                "total_records": 0,
                "consolidation_strategy": "merge"
            }
        }
        
        for source_name, result in results.items():
            if result["status"] == "success":
                source_data = result["data"]
                
                # Convert to standard record format
                if isinstance(source_data, list):
                    consolidated["records"].extend(source_data)
                elif isinstance(source_data, dict):
                    consolidated["records"].append(source_data)
                
                consolidated["metadata"]["sources"].append({
                    "name": source_name,
                    "record_count": len(source_data) if isinstance(source_data, list) else 1,
                    "source_info": result.get("source_info", {})
                })
        
        consolidated["metadata"]["total_records"] = len(consolidated["records"])
        
        return consolidated
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consolidated extracted data"""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "record_count": len(data.get("records", [])),
            "validation_rules_applied": []
        }
        
        records = data.get("records", [])
        
        # Check if we have data
        if not records:
            validation_results["errors"].append("No records extracted")
            validation_results["valid"] = False
        
        # Basic data structure validation
        if records:
            first_record = records[0]
            required_fields = ["id", "name"] if isinstance(first_record, dict) else []
            
            for field in required_fields:
                missing_count = sum(1 for record in records if field not in record)
                if missing_count > 0:
                    validation_results["warnings"].append(
                        f"Field '{field}' missing in {missing_count} records"
                    )
        
        validation_results["validation_rules_applied"] = ["record_count", "required_fields"]
        
        return validation_results
```

### Data Processing Agent

```python
# data_processing_agent.py
from decimal import Decimal, InvalidOperation
import re
from datetime import datetime
from typing import Dict, Any, List

class DataProcessingAgent:
    """Level 2: Advanced data processing with validation and transformation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_rules = self._load_processing_rules()
        self.transformation_history = []
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing task"""
        
        try:
            # Get input data from previous stages
            input_data = self._extract_input_data(context)
            
            # Determine processing operations from task
            operations = self._parse_processing_operations(task, context)
            
            # Apply processing operations
            processed_data = input_data.copy()
            operation_results = {}
            
            for operation in operations:
                result = await self._apply_operation(operation, processed_data)
                processed_data = result["processed_data"]
                operation_results[operation["name"]] = result
            
            # Validate processed data
            validation_results = self._validate_processed_data(processed_data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                input_data, processed_data, operation_results
            )
            
            final_result = {
                "status": "success",
                "processed_data": processed_data,
                "operations_applied": list(operation_results.keys()),
                "operation_results": operation_results,
                "validation": validation_results,
                "quality_metrics": quality_metrics,
                "processing_timestamp": datetime.now().isoformat(),
                "input_record_count": len(input_data.get("records", [])),
                "output_record_count": len(processed_data.get("records", []))
            }
            
            # Store processing history
            self.transformation_history.append({
                "task": task,
                "operations": operations,
                "result": final_result,
                "timestamp": datetime.now()
            })
            
            return final_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "task": task,
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def _extract_input_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input data from context"""
        
        # Check for data from previous stages
        if "dependencies" in context:
            for dep_name, dep_result in context["dependencies"].items():
                if dep_result.get("status") == "success":
                    if "extracted_data" in dep_result:
                        return dep_result["extracted_data"]
                    elif "processed_data" in dep_result:
                        return dep_result["processed_data"]
        
        # Check for direct data input
        if "data" in context:
            return context["data"]
        
        # Check for previous results
        if "previous_results" in context:
            for stage_name, result in context["previous_results"].items():
                if result.get("status") == "success":
                    if "extracted_data" in result:
                        return result["extracted_data"]
        
        raise ValueError("No input data found in context")
    
    def _parse_processing_operations(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse task to determine required processing operations"""
        
        operations = []
        
        # Check explicit operations in context
        if "operations" in context:
            return context["operations"]
        
        # Parse task description
        task_lower = task.lower()
        
        if "clean" in task_lower or "validate" in task_lower:
            operations.append({
                "name": "data_cleaning",
                "type": "cleaning",
                "config": {"remove_duplicates": True, "validate_formats": True}
            })
        
        if "normalize" in task_lower or "standardize" in task_lower:
            operations.append({
                "name": "normalization",
                "type": "transformation",
                "config": {"normalize_names": True, "standardize_formats": True}
            })
        
        if "calculate" in task_lower or "compute" in task_lower:
            operations.append({
                "name": "calculations",
                "type": "calculation",
                "config": {"add_derived_fields": True}
            })
        
        if "filter" in task_lower:
            operations.append({
                "name": "filtering",
                "type": "filtering",
                "config": {"apply_business_rules": True}
            })
        
        # Default operations if none specified
        if not operations:
            operations = [
                {
                    "name": "basic_processing",
                    "type": "cleaning",
                    "config": {"basic_validation": True}
                }
            ]
        
        return operations
    
    async def _apply_operation(self, operation: Dict[str, Any], 
                             data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific processing operation"""
        
        operation_type = operation["type"]
        config = operation.get("config", {})
        
        if operation_type == "cleaning":
            return await self._apply_cleaning_operation(data, config)
        elif operation_type == "transformation":
            return await self._apply_transformation_operation(data, config)
        elif operation_type == "calculation":
            return await self._apply_calculation_operation(data, config)
        elif operation_type == "filtering":
            return await self._apply_filtering_operation(data, config)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    async def _apply_cleaning_operation(self, data: Dict[str, Any], 
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data cleaning operations"""
        
        cleaned_data = data.copy()
        cleaning_stats = {
            "records_processed": 0,
            "records_removed": 0,
            "fields_cleaned": 0,
            "issues_found": []
        }
        
        records = cleaned_data.get("records", [])
        cleaned_records = []
        
        for record in records:
            cleaning_stats["records_processed"] += 1
            cleaned_record = record.copy()
            record_valid = True
            
            # Remove duplicates based on ID
            if config.get("remove_duplicates", False):
                record_id = record.get("id")
                if record_id and any(r.get("id") == record_id for r in cleaned_records):
                    cleaning_stats["records_removed"] += 1
                    cleaning_stats["issues_found"].append(f"Duplicate record: {record_id}")
                    record_valid = False
            
            # Validate formats
            if record_valid and config.get("validate_formats", False):
                for field, value in record.items():
                    if field.endswith("_amount") or field == "revenue":
                        try:
                            # Validate monetary amounts
                            if isinstance(value, str):
                                cleaned_value = self._clean_monetary_value(value)
                                cleaned_record[field] = cleaned_value
                                cleaning_stats["fields_cleaned"] += 1
                        except ValueError as e:
                            cleaning_stats["issues_found"].append(f"Invalid amount in {field}: {value}")
                            record_valid = False
                    
                    elif field.endswith("_date") or field == "date":
                        try:
                            # Validate dates
                            if isinstance(value, str):
                                cleaned_value = self._clean_date_value(value)
                                cleaned_record[field] = cleaned_value
                                cleaning_stats["fields_cleaned"] += 1
                        except ValueError as e:
                            cleaning_stats["issues_found"].append(f"Invalid date in {field}: {value}")
            
            if record_valid:
                cleaned_records.append(cleaned_record)
            else:
                cleaning_stats["records_removed"] += 1
        
        cleaned_data["records"] = cleaned_records
        
        return {
            "processed_data": cleaned_data,
            "operation_stats": cleaning_stats,
            "records_before": len(records),
            "records_after": len(cleaned_records)
        }
    
    async def _apply_transformation_operation(self, data: Dict[str, Any],
                                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data transformation operations"""
        
        transformed_data = data.copy()
        transformation_stats = {
            "fields_normalized": 0,
            "records_transformed": 0,
            "transformations_applied": []
        }
        
        records = transformed_data.get("records", [])
        
        for record in records:
            transformation_stats["records_transformed"] += 1
            
            # Normalize names
            if config.get("normalize_names", False):
                if "name" in record or "podcast_name" in record:
                    name_field = "name" if "name" in record else "podcast_name"
                    original_name = record[name_field]
                    normalized_name = self._normalize_name(original_name)
                    record[name_field] = normalized_name
                    transformation_stats["fields_normalized"] += 1
                    transformation_stats["transformations_applied"].append(
                        f"Normalized {name_field}: {original_name} -> {normalized_name}"
                    )
            
            # Standardize formats
            if config.get("standardize_formats", False):
                for field, value in record.items():
                    if field == "currency":
                        record[field] = str(value).upper()
                        transformation_stats["fields_normalized"] += 1
        
        transformed_data["records"] = records
        
        return {
            "processed_data": transformed_data,
            "operation_stats": transformation_stats
        }
    
    async def _apply_calculation_operation(self, data: Dict[str, Any],
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation operations"""
        
        calculated_data = data.copy()
        calculation_stats = {
            "calculations_performed": 0,
            "derived_fields_added": 0,
            "calculation_errors": []
        }
        
        records = calculated_data.get("records", [])
        
        for record in records:
            try:
                # Add derived fields
                if config.get("add_derived_fields", False):
                    
                    # Calculate revenue per impression if both fields exist
                    if "revenue" in record and "impressions" in record:
                        try:
                            revenue = Decimal(str(record["revenue"]))
                            impressions = int(record["impressions"])
                            
                            if impressions > 0:
                                rpm = revenue / impressions * 1000  # Revenue per mille
                                record["rpm"] = str(rpm.quantize(Decimal('0.01')))
                                calculation_stats["derived_fields_added"] += 1
                                calculation_stats["calculations_performed"] += 1
                        except (ValueError, InvalidOperation) as e:
                            calculation_stats["calculation_errors"].append(
                                f"RPM calculation error: {str(e)}"
                            )
                    
                    # Add month-year field if date exists
                    if "date" in record:
                        try:
                            date_str = record["date"]
                            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            record["month_year"] = date_obj.strftime("%Y-%m")
                            calculation_stats["derived_fields_added"] += 1
                        except ValueError as e:
                            calculation_stats["calculation_errors"].append(
                                f"Date parsing error: {str(e)}"
                            )
                            
            except Exception as e:
                calculation_stats["calculation_errors"].append(str(e))
        
        calculated_data["records"] = records
        
        return {
            "processed_data": calculated_data,
            "operation_stats": calculation_stats
        }
    
    def _clean_monetary_value(self, value: str) -> str:
        """Clean and validate monetary values"""
        if not value:
            raise ValueError("Empty monetary value")
        
        # Remove currency symbols and formatting
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        
        # Validate as decimal
        try:
            decimal_value = Decimal(cleaned)
            return str(decimal_value.quantize(Decimal('0.01')))
        except InvalidOperation:
            raise ValueError(f"Invalid monetary format: {value}")
    
    def _clean_date_value(self, value: str) -> str:
        """Clean and validate date values"""
        if not value:
            raise ValueError("Empty date value")
        
        # Try to parse various date formats
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S"
        ]
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(value, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        raise ValueError(f"Invalid date format: {value}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize podcast/entity names"""
        if not name:
            return name
        
        # Basic normalization
        normalized = name.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = normalized.title()  # Title case
        
        return normalized
    
    def _validate_processed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processed data quality"""
        
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "quality_score": 0.0
        }
        
        records = data.get("records", [])
        
        if not records:
            validation["errors"].append("No records after processing")
            validation["valid"] = False
            return validation
        
        # Check data consistency
        sample_record = records[0]
        expected_fields = set(sample_record.keys())
        
        inconsistent_records = 0
        for record in records:
            if set(record.keys()) != expected_fields:
                inconsistent_records += 1
        
        if inconsistent_records > 0:
            validation["warnings"].append(
                f"{inconsistent_records} records have inconsistent field structure"
            )
        
        # Calculate quality score
        total_checks = 3
        passed_checks = 0
        
        # Check 1: Records exist
        if records:
            passed_checks += 1
        
        # Check 2: Structure consistency
        if inconsistent_records / len(records) < 0.1:  # Less than 10% inconsistent
            passed_checks += 1
        
        # Check 3: Required fields present
        required_fields = ["id", "name"]
        if any(field in sample_record for field in required_fields):
            passed_checks += 1
        
        validation["quality_score"] = passed_checks / total_checks
        
        return validation
    
    def _calculate_quality_metrics(self, input_data: Dict[str, Any],
                                 output_data: Dict[str, Any],
                                 operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        
        input_count = len(input_data.get("records", []))
        output_count = len(output_data.get("records", []))
        
        metrics = {
            "input_record_count": input_count,
            "output_record_count": output_count,
            "data_retention_rate": output_count / input_count if input_count > 0 else 0,
            "operations_successful": sum(1 for r in operation_results.values() 
                                       if "operation_stats" in r),
            "total_operations": len(operation_results),
            "processing_efficiency": 0.0
        }
        
        # Calculate processing efficiency
        total_issues = 0
        for result in operation_results.values():
            stats = result.get("operation_stats", {})
            total_issues += len(stats.get("issues_found", []))
            total_issues += len(stats.get("calculation_errors", []))
        
        if input_count > 0:
            metrics["processing_efficiency"] = max(0, 1 - (total_issues / input_count))
        
        return metrics
    
    def _load_processing_rules(self) -> Dict[str, Any]:
        """Load processing rules from configuration"""
        return {
            "monetary_precision": 2,
            "date_format": "%Y-%m-%d",
            "required_fields": ["id", "name"],
            "optional_fields": ["description", "metadata"]
        }
```

---

## Web Interface for Level 2

### Streamlit Dashboard

```python
# level2_web_interface.py
import streamlit as st
import asyncio
import json
import time
from typing import Dict, Any

class Level2WebInterface:
    """Web interface for Level 2 agent systems"""
    
    def __init__(self, orchestrator: Level2Orchestrator):
        self.orchestrator = orchestrator
        self.setup_page()
        
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Level 2 Agent System",
            page_icon="🔄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.title("🔄 Level 2 Multi-Agent System")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "Execute Workflow", "History", "Agent Status"]
        )
        
        if page == "Dashboard":
            self.render_main_dashboard()
        elif page == "Execute Workflow":
            self.render_workflow_execution()
        elif page == "History":
            self.render_execution_history()
        elif page == "Agent Status":
            self.render_agent_status()
    
    def render_main_dashboard(self):
        """Render main dashboard overview"""
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Agents", len(self.orchestrator.agents))
        
        with col2:
            st.metric("Executions Today", len(self.orchestrator.execution_history))
        
        with col3:
            success_rate = self._calculate_success_rate()
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col4:
            avg_time = self._calculate_average_execution_time()
            st.metric("Avg Execution Time", f"{avg_time:.1f}s")
        
        # Recent executions
        st.subheader("Recent Workflow Executions")
        
        if self.orchestrator.execution_history:
            for execution in self.orchestrator.execution_history[-5:]:
                with st.expander(f"Execution {execution['execution_id'][:8]}... - {execution['timestamp'].strftime('%H:%M:%S')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text(f"Workflow: {execution['config'].workflow_name}")
                        st.text(f"Status: {execution['result']['status']}")
                        st.text(f"Duration: {execution['result']['execution_time']:.2f}s")
                    
                    with col2:
                        st.text(f"Stages: {len(execution['config'].stages)}")
                        if execution['result']['status'] == 'success':
                            st.text(f"Completed: {len(execution['result']['stage_results'])}")
                        else:
                            st.text(f"Error: {execution['result'].get('error', 'Unknown')}")
        else:
            st.info("No workflow executions yet.")
    
    def render_workflow_execution(self):
        """Render workflow execution interface"""
        st.header("Execute Workflow")
        
        # Workflow configuration
        with st.form("workflow_config"):
            workflow_name = st.text_input("Workflow Name", "Data Processing Pipeline")
            
            st.subheader("Input Data")
            data_source = st.selectbox("Data Source", ["Manual Input", "File Upload", "API"])
            
            if data_source == "Manual Input":
                input_data = st.text_area(
                    "Input Data (JSON)",
                    value='{"sources": {"file": {"path": "./data/sample.csv", "format": "csv"}}}',
                    height=200
                )
            elif data_source == "File Upload":
                uploaded_file = st.file_uploader("Upload Data File", type=["json", "csv"])
                input_data = '{"file_uploaded": true}' if uploaded_file else '{}'
            else:
                api_url = st.text_input("API URL", "https://api.example.com/data")
                input_data = f'{{"sources": {{"api": {{"url": "{api_url}"}}}}}}'
            
            st.subheader("Workflow Configuration")
            
            # Agent selection
            available_agents = list(self.orchestrator.agents.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                extraction_agent = st.selectbox("Data Extraction Agent", available_agents)
            with col2:
                processing_agent = st.selectbox("Data Processing Agent", available_agents)
            
            # Processing options
            enable_progress = st.checkbox("Enable Progress Tracking", value=True)
            enable_context = st.checkbox("Enable Context Sharing", value=True)
            
            submitted = st.form_submit_button("Execute Workflow")
        
        if submitted:
            try:
                # Parse input data
                parsed_input = json.loads(input_data)
                
                # Create workflow configuration
                workflow_config = WorkflowConfig(
                    workflow_name=workflow_name,
                    stages=[
                        WorkflowStage(
                            name="data_extraction",
                            agent_name=extraction_agent,
                            task_template="Extract data from configured sources"
                        ),
                        WorkflowStage(
                            name="data_processing",
                            agent_name=processing_agent,
                            task_template="Process and validate extracted data",
                            depends_on=["data_extraction"]
                        )
                    ],
                    context_sharing_enabled=enable_context,
                    progress_tracking_enabled=enable_progress
                )
                
                # Execute workflow with progress tracking
                self._execute_workflow_with_progress(workflow_config, parsed_input)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON in input data")
            except Exception as e:
                st.error(f"Workflow execution failed: {str(e)}")
    
    def _execute_workflow_with_progress(self, config: WorkflowConfig, input_data: Dict[str, Any]):
        """Execute workflow with real-time progress updates"""
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_container = st.empty()
        results_container = st.empty()
        
        # Progress callback for Streamlit
        def progress_callback(stage: str, progress: int, message: str = ""):
            progress_bar.progress(progress / 100.0)
            status_container.markdown(f"**{stage}**: {message} ({progress}%)")
        
        # Execute workflow
        try:
            with st.spinner("Executing workflow..."):
                result = asyncio.run(
                    self.orchestrator.execute_workflow(
                        config, input_data, progress_callback
                    )
                )
            
            # Display results
            if result["status"] == "success":
                st.success("✅ Workflow completed successfully!")
                
                # Results summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Execution Time", f"{result['execution_time']:.2f}s")
                
                with col2:
                    st.metric("Stages Completed", len(result['stage_results']))
                
                with col3:
                    st.metric("Status", result['status'].title())
                
                # Detailed results
                with st.expander("Detailed Results", expanded=True):
                    st.json(result)
                
                # Download results
                st.download_button(
                    "Download Results",
                    data=json.dumps(result, indent=2),
                    file_name=f"workflow_results_{result['execution_id']}.json",
                    mime="application/json"
                )
                
            else:
                st.error(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
                with st.expander("Error Details"):
                    st.json(result)
                    
        except Exception as e:
            st.error(f"Execution error: {str(e)}")
    
    def render_execution_history(self):
        """Render execution history"""
        st.header("Execution History")
        
        if not self.orchestrator.execution_history:
            st.info("No executions found.")
            return
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Success", "Error"])
        
        with col2:
            limit = st.slider("Show Last N Executions", 5, 50, 10)
        
        # Display executions
        filtered_history = self.orchestrator.execution_history[-limit:]
        
        if status_filter != "All":
            filtered_history = [
                execution for execution in filtered_history
                if execution['result']['status'] == status_filter.lower()
            ]
        
        for execution in reversed(filtered_history):
            result = execution['result']
            config = execution['config']
            
            status_icon = "✅" if result['status'] == 'success' else "❌"
            
            with st.expander(f"{status_icon} {config.workflow_name} - {execution['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.text(f"ID: {result['execution_id'][:8]}...")
                    st.text(f"Status: {result['status']}")
                    st.text(f"Duration: {result['execution_time']:.2f}s")
                
                with col2:
                    st.text(f"Stages: {len(config.stages)}")
                    if result['status'] == 'success':
                        st.text(f"Completed: {len(result['stage_results'])}")
                    else:
                        st.text(f"Failed at: {result.get('completed_stages', [])}")
                
                with col3:
                    st.text(f"Context: {'Yes' if config.context_sharing_enabled else 'No'}")
                    st.text(f"Progress: {'Yes' if config.progress_tracking_enabled else 'No'}")
                
                # Detailed view
                if st.button(f"View Details", key=f"details_{result['execution_id']}"):
                    st.json(result)
    
    def render_agent_status(self):
        """Render agent status information"""
        st.header("Agent Status")
        
        for agent_name, agent in self.orchestrator.agents.items():
            with st.expander(f"🤖 {agent_name}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Type: {type(agent).__name__}")
                    
                    # Get agent stats if available
                    if hasattr(agent, 'get_stats'):
                        stats = agent.get_stats()
                        st.metric("Executions", stats.get('total_executions', 0))
                        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
                    else:
                        st.text("Stats not available")
                
                with col2:
                    if hasattr(agent, 'config'):
                        st.text(f"Role: {getattr(agent.config, 'agent_role', 'Unknown')}")
                        expertise = getattr(agent.config, 'expertise_areas', [])
                        if expertise:
                            st.text(f"Expertise: {', '.join(expertise)}")
                    
                    # Health check
                    try:
                        health_status = "🟢 Healthy"
                        st.text(f"Status: {health_status}")
                    except:
                        st.text("Status: 🔴 Error")
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.orchestrator.execution_history:
            return 0.0
        
        successful = sum(1 for execution in self.orchestrator.execution_history
                        if execution['result']['status'] == 'success')
        
        return successful / len(self.orchestrator.execution_history)
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self.orchestrator.execution_history:
            return 0.0
        
        total_time = sum(execution['result']['execution_time']
                        for execution in self.orchestrator.execution_history)
        
        return total_time / len(self.orchestrator.execution_history)

# Launch web interface
def launch_level2_web():
    """Launch Level 2 web interface"""
    
    # Initialize agents (would be loaded from configuration)
    agents = {
        "data_extractor": DataExtractionAgent({"source_types": ["api", "file"]}),
        "data_processor": DataProcessingAgent({"processing_rules": {}})
    }
    
    # Initialize orchestrator
    orchestrator = Level2Orchestrator(agents)
    
    # Create and render web interface
    web_interface = Level2WebInterface(orchestrator)
    web_interface.render_dashboard()

if __name__ == "__main__":
    launch_level2_web()
```

---

## Example Level 2 Implementation

### Complete Revenue Processing System

```python
# revenue_processing_system.py
import asyncio
import os
from decimal import Decimal

async def main():
    """Complete Level 2 revenue processing example"""
    
    # Configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "data_sources": {
            "megaphone": {"type": "api", "url": "https://api.megaphone.fm/data"},
            "audiomax": {"type": "file", "path": "./data/audiomax_data.csv"}
        }
    }
    
    # Initialize agents
    extraction_agent = DataExtractionAgent(config)
    processing_agent = DataProcessingAgent(config)
    
    agents = {
        "data_extractor": extraction_agent,
        "data_processor": processing_agent
    }
    
    # Initialize orchestrator
    orchestrator = Level2Orchestrator(agents)
    
    # Define workflow
    workflow_config = WorkflowConfig(
        workflow_name="Revenue Data Processing",
        stages=[
            WorkflowStage(
                name="extract_revenue_data",
                agent_name="data_extractor",
                task_template="Extract revenue data from Megaphone and Audiomax sources"
            ),
            WorkflowStage(
                name="process_revenue_data",
                agent_name="data_processor", 
                task_template="Clean, validate and calculate derived metrics for revenue data",
                depends_on=["extract_revenue_data"]
            )
        ],
        context_sharing_enabled=True,
        progress_tracking_enabled=True
    )
    
    # Input data
    input_data = {
        "sources": config["data_sources"],
        "processing_month": "2024-11",
        "validation_rules": {
            "min_revenue": 0.01,
            "max_revenue": 100000.00,
            "required_fields": ["podcast_name", "revenue", "date"]
        }
    }
    
    # Progress callback
    def progress_callback(stage: str, progress: int, message: str = ""):
        print(f"[{progress:3d}%] {stage}: {message}")
    
    # Execute workflow
    print("🚀 Starting revenue processing workflow...")
    
    result = await orchestrator.execute_workflow(
        workflow_config, input_data, progress_callback
    )
    
    # Display results
    if result["status"] == "success":
        print("\n✅ Workflow completed successfully!")
        print(f"   Execution ID: {result['execution_id']}")
        print(f"   Duration: {result['execution_time']:.2f} seconds")
        print(f"   Stages completed: {len(result['stage_results'])}")
        
        # Show stage results summary
        for stage_name, stage_result in result["stage_results"].items():
            print(f"\n📊 {stage_name}:")
            if stage_result.get("status") == "success":
                if "record_count" in stage_result:
                    print(f"   Records processed: {stage_result['record_count']}")
                if "quality_metrics" in stage_result:
                    metrics = stage_result["quality_metrics"]
                    print(f"   Data quality score: {metrics.get('processing_efficiency', 0):.1%}")
                    print(f"   Data retention rate: {metrics.get('data_retention_rate', 0):.1%}")
            else:
                print(f"   ❌ Error: {stage_result.get('error', 'Unknown')}")
    
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
        print(f"   Completed stages: {result.get('completed_stages', [])}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Integration for Level 2 Systems

### Simple FastAPI Wrapper

**Expose Level 2 workflows through REST API:**

```python
# level2_api_wrapper.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import uuid
from datetime import datetime

# Request/Response Models
class WorkflowRequest(BaseModel):
    """Request model for Level 2 workflow execution"""
    task: str = Field(..., description="Task description")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    workflow_config: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")
    callback_url: Optional[str] = Field(None, description="Webhook for completion notification")

class WorkflowResponse(BaseModel):
    """Response model for Level 2 workflow"""
    workflow_id: str
    status: str  # "processing", "completed", "failed"
    progress: Dict[str, Any] = Field(default_factory=dict)
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

class Level2APIWrapper:
    """FastAPI wrapper for Level 2 Standard Systems"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Level 2 Agent System API",
            description="Multi-agent workflow orchestration for business applications",
            version="1.0.0"
        )
        
        # Initialize Level 2 components
        self.orchestrator = WorkflowOrchestrator()
        self.active_workflows = {}
        
        # Setup agents
        self.setup_agents()
        self.setup_routes()
    
    def setup_agents(self):
        """Initialize and register Level 2 agents"""
        
        # Create Level 2 agents
        data_extractor = DataExtractionAgent(AgentPersonality(
            name="DataExtractor",
            role="Data extraction and preprocessing",
            expertise=["data_extraction", "preprocessing", "validation"],
            system_prompt="Extract and preprocess data from various sources.",
            model_preference="gpt-4o-mini"
        ))
        
        data_processor = DataProcessingAgent(AgentPersonality(
            name="DataProcessor", 
            role="Data processing and transformation",
            expertise=["data_processing", "transformation", "calculations"],
            system_prompt="Process and transform data with business logic.",
            model_preference="gpt-4o-mini"
        ))
        
        insights_generator = InsightsGeneratorAgent(AgentPersonality(
            name="InsightsGenerator",
            role="Generate insights and reports",
            expertise=["analysis", "insights", "reporting"],
            system_prompt="Generate actionable insights from processed data.",
            model_preference="gpt-4o-mini"
        ))
        
        # Register agents with orchestrator
        self.orchestrator.register_agent("data_extractor", data_extractor)
        self.orchestrator.register_agent("data_processor", data_processor)
        self.orchestrator.register_agent("insights_generator", insights_generator)
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/v1/workflows/execute", response_model=WorkflowResponse)
        async def execute_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
            """Execute a Level 2 workflow"""
            
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
            
            # Create workflow configuration
            workflow_config = request.workflow_config or self._create_default_workflow_config()
            
            # Initialize workflow tracking
            self.active_workflows[workflow_id] = {
                "workflow_id": workflow_id,
                "status": "processing",
                "progress": {"current_stage": "initialization", "completion": 0},
                "started_at": datetime.utcnow(),
                "request": request.dict()
            }
            
            # Execute workflow in background
            background_tasks.add_task(
                self._execute_workflow_background,
                workflow_id, request.task, request.input_data, workflow_config
            )
            
            return WorkflowResponse(
                workflow_id=workflow_id,
                status="processing",
                progress={"current_stage": "starting", "completion": 0}
            )
        
        @self.app.get("/v1/workflows/{workflow_id}/status", response_model=WorkflowResponse)
        async def get_workflow_status(workflow_id: str):
            """Get workflow execution status"""
            
            if workflow_id not in self.active_workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            workflow_data = self.active_workflows[workflow_id]
            
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=workflow_data["status"],
                progress=workflow_data.get("progress", {}),
                results=workflow_data.get("results"),
                error=workflow_data.get("error"),
                execution_time=workflow_data.get("execution_time")
            )
        
        @self.app.get("/v1/workflows", response_model=List[WorkflowResponse])
        async def list_workflows(status: Optional[str] = None, limit: int = 20):
            """List recent workflows"""
            
            workflows = []
            for wf_id, wf_data in list(self.active_workflows.items())[-limit:]:
                if status is None or wf_data["status"] == status:
                    workflows.append(WorkflowResponse(
                        workflow_id=wf_id,
                        status=wf_data["status"],
                        progress=wf_data.get("progress", {}),
                        results=wf_data.get("results"),
                        error=wf_data.get("error"),
                        execution_time=wf_data.get("execution_time")
                    ))
            
            return workflows
        
        @self.app.get("/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "active_workflows": len([
                    wf for wf in self.active_workflows.values() 
                    if wf["status"] == "processing"
                ]),
                "total_workflows": len(self.active_workflows),
                "agents": {
                    name: "ready" for name in self.orchestrator.agents.keys()
                }
            }
        
        @self.app.get("/v1/agents/capabilities")
        async def get_agent_capabilities():
            """Get available agents and their capabilities"""
            return {
                "agents": {
                    name: {
                        "role": agent.personality.role,
                        "expertise": agent.personality.expertise,
                        "model": agent.personality.model_preference
                    }
                    for name, agent in self.orchestrator.agents.items()
                }
            }
    
    async def _execute_workflow_background(self, workflow_id: str, task: str, 
                                         input_data: Dict[str, Any], config: Dict[str, Any]):
        """Execute workflow in background with progress tracking"""
        
        start_time = datetime.utcnow()
        
        try:
            # Create progress tracker
            def update_progress(stage: str, completion: int, message: str = ""):
                if workflow_id in self.active_workflows:
                    self.active_workflows[workflow_id]["progress"] = {
                        "current_stage": stage,
                        "completion": completion,
                        "message": message
                    }
            
            # Update progress: Starting extraction
            update_progress("data_extraction", 10, "Starting data extraction")
            
            # Stage 1: Data Extraction
            extraction_result = await self.orchestrator.execute_stage(
                "data_extractor", task, input_data
            )
            
            update_progress("data_processing", 40, "Data extraction completed, starting processing")
            
            # Stage 2: Data Processing
            processing_result = await self.orchestrator.execute_stage(
                "data_processor", task, extraction_result["result"]
            )
            
            update_progress("insights_generation", 70, "Processing completed, generating insights")
            
            # Stage 3: Insights Generation
            insights_result = await self.orchestrator.execute_stage(
                "insights_generator", task, processing_result["result"]
            )
            
            # Compile final results
            final_results = {
                "extraction": extraction_result,
                "processing": processing_result,
                "insights": insights_result,
                "workflow_summary": {
                    "total_stages": 3,
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "status": "completed"
                }
            }
            
            # Update final status
            self.active_workflows[workflow_id].update({
                "status": "completed",
                "results": final_results,
                "progress": {"current_stage": "completed", "completion": 100},
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            })
            
        except Exception as e:
            # Handle errors
            self.active_workflows[workflow_id].update({
                "status": "failed",
                "error": str(e),
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            })
    
    def _create_default_workflow_config(self) -> Dict[str, Any]:
        """Create default workflow configuration"""
        return {
            "stages": [
                {
                    "name": "extraction",
                    "agent": "data_extractor",
                    "timeout": 60,
                    "retry_count": 2
                },
                {
                    "name": "processing", 
                    "agent": "data_processor",
                    "timeout": 120,
                    "retry_count": 2,
                    "depends_on": ["extraction"]
                },
                {
                    "name": "insights",
                    "agent": "insights_generator", 
                    "timeout": 90,
                    "retry_count": 1,
                    "depends_on": ["processing"]
                }
            ],
            "global_timeout": 300,
            "enable_progress_tracking": True
        }

# Usage
def create_level2_api():
    """Create Level 2 API application"""
    api_wrapper = Level2APIWrapper()
    return api_wrapper.app

# Run the API
if __name__ == "__main__":
    import uvicorn
    
    app = create_level2_api()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### Client Usage Examples

**Simple Python client for Level 2 API:**

```python
# level2_client_example.py
import requests
import time
import json

class Level2Client:
    """Simple client for Level 2 API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def execute_workflow(self, task: str, input_data: dict, 
                        wait_for_completion: bool = True) -> dict:
        """Execute a workflow and optionally wait for completion"""
        
        # Submit workflow
        response = requests.post(
            f"{self.base_url}/v1/workflows/execute",
            json={
                "task": task,
                "input_data": input_data
            }
        )
        response.raise_for_status()
        
        workflow_data = response.json()
        workflow_id = workflow_data["workflow_id"]
        
        if not wait_for_completion:
            return {"workflow_id": workflow_id, "status": "submitted"}
        
        # Poll for completion
        while True:
            status_response = requests.get(
                f"{self.base_url}/v1/workflows/{workflow_id}/status"
            )
            status_response.raise_for_status()
            
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                return status_data["results"]
            elif status_data["status"] == "failed":
                raise Exception(f"Workflow failed: {status_data.get('error')}")
            
            # Show progress
            progress = status_data.get("progress", {})
            print(f"Progress: {progress.get('current_stage', 'unknown')} "
                  f"({progress.get('completion', 0)}%)")
            
            time.sleep(2)  # Poll every 2 seconds

# Usage examples
def main():
    client = Level2Client()
    
    # Example 1: Process financial data
    financial_data = {
        "records": [
            {"date": "2024-01-15", "revenue": "1500.75", "currency": "USD"},
            {"date": "2024-01-16", "revenue": "1200.50", "currency": "USD"}
        ]
    }
    
    result = client.execute_workflow(
        task="Process and analyze financial revenue data",
        input_data=financial_data
    )
    
    print("Financial analysis results:")
    print(json.dumps(result, indent=2))
    
    # Example 2: Process marketing data (async)
    marketing_data = {
        "campaigns": [
            {"name": "Campaign A", "impressions": 10000, "clicks": 150, "spend": 500},
            {"name": "Campaign B", "impressions": 8000, "clicks": 120, "spend": 400}
        ]
    }
    
    async_result = client.execute_workflow(
        task="Analyze marketing campaign performance", 
        input_data=marketing_data,
        wait_for_completion=False
    )
    
    print(f"Marketing analysis started: {async_result['workflow_id']}")

if __name__ == "__main__":
    main()
```

### cURL Examples

**API testing with cURL:**

```bash
# Start a workflow
curl -X POST "http://localhost:8000/v1/workflows/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Process customer data and generate insights",
    "input_data": {
      "customers": [
        {"id": 1, "name": "John Doe", "purchases": 5, "total_spent": 250.50},
        {"id": 2, "name": "Jane Smith", "purchases": 3, "total_spent": 180.25}
      ]
    }
  }'

# Get workflow status
curl "http://localhost:8000/v1/workflows/wf_12345678/status"

# List all workflows
curl "http://localhost:8000/v1/workflows?limit=10"

# Check system health
curl "http://localhost:8000/v1/health"

# Get agent capabilities
curl "http://localhost:8000/v1/agents/capabilities"
```

---

## Best Practices for Level 2

### System Design
1. **Clear Stage Separation**: Each agent should have a distinct, well-defined role
2. **Context Strategy**: Share relevant data without overwhelming agents
3. **Progress Granularity**: Provide meaningful progress updates at each stage
4. **Error Recovery**: Handle failures gracefully with clear error messages
5. **Resource Management**: Monitor memory and processing time across agents

### Development Workflow
1. **Start Simple**: Begin with 2 agents, add third when needed
2. **Test Stages Individually**: Validate each agent separately before orchestration
3. **Mock External Services**: Use test data for development and testing
4. **Monitor Performance**: Track execution time and success rates
5. **Documentation**: Document workflow configurations and dependencies

---

## Migration Paths

### From Level 1 to Level 2
1. **Add Orchestrator**: Wrap existing agent in workflow orchestrator
2. **Create Second Agent**: Add complementary processing agent
3. **Enable Context Sharing**: Allow agents to share relevant data
4. **Add Progress Tracking**: Implement real-time feedback
5. **Build Web Interface**: Upgrade from CLI to web dashboard

### To Level 3 Complex Systems
- Add more specialized agents (3-5 total)
- Implement advanced error recovery and retry logic
- Add parallel execution capabilities
- Integrate external systems and APIs
- Implement advanced monitoring and alerting

---

## Next Steps

- **Scale Up**: [Level 3 Complex Systems](level_3_complex.md) for enterprise workflows
- **Production**: [Level 4 Production Systems](level_4_production.md) for mission-critical deployments
- **Specialized Features**: [Financial Precision](../04_specialized/financial_precision.md) for financial applications

---

*Level 2 systems provide robust multi-agent coordination while maintaining simplicity and rapid development cycles, making them ideal for most business applications.*
# Testing Frameworks for Agent Systems

> 🧪 **Quality Assurance**: Comprehensive testing strategies for agent systems, from unit tests to end-to-end workflow validation.

## Navigation
- **Previous**: [Financial Precision](financial_precision.md)
- **Next**: [Context Management](context_management.md)
- **Implementation**: [Level 2: Standard](../05_implementation_levels/level_2_standard.md) → [Level 4: Production](../05_implementation_levels/level_4_production.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Troubleshooting](../06_reference/troubleshooting.md)

---

## Overview

Agent systems require specialized testing approaches due to their async nature, LLM integration, and complex workflows. This section provides proven testing patterns that ensure reliability, performance, and maintainability across all system complexity levels.

---

## Testing Strategy Overview

```
Unit Tests → Integration Tests → End-to-End Tests → Performance Tests
     ↓              ↓                   ↓                  ↓
Individual     Agent            Complete         Load &
Components     Interactions     Workflows        Scalability
```

| Test Type | Scope | Tools | Coverage Target |
|-----------|-------|-------|----------------|
| **Unit Tests** | Individual agents, components | pytest, asyncio | 90%+ |
| **Integration Tests** | Agent coordination, workflows | pytest, mocking | 80%+ |
| **End-to-End Tests** | Complete system workflows | pytest, test data | 100% scenarios |
| **Performance Tests** | Load, scalability, timing | pytest-benchmark | Key operations |

---

## Core Testing Infrastructure

### Test Configuration and Setup

```python
# conftest.py - Test configuration
import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal
import json

from core.orchestrator import UniversalOrchestrator
from core.base_agent import BaseRevenueAgent
from database.models import TestDatabase
from core.context_manager import MemoryContextManager

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_database():
    """Provide isolated test database"""
    db = TestDatabase()
    await db.initialize()
    yield db
    await db.cleanup()

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for consistent testing"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client

@pytest.fixture
def sample_revenue_data():
    """Provide realistic test data"""
    return {
        "megaphone_data": [
            {
                "podcast_name": "Tech Talk Daily",
                "month": 11,
                "year": 2024,
                "programmatic_revenue": Decimal("1250.75"),
                "impressions": 45000,
                "fill_rate": Decimal("0.85")
            }
        ],
        "audiomax_data": [
            {
                "podcast_name": "Tech Talk Daily",
                "month": 11,
                "year": 2024,
                "net_revenue": Decimal("850.50"),
                "impressions": 32000
            }
        ]
    }

@pytest.fixture
async def test_orchestrator(test_database, mock_openai_client):
    """Provide configured test orchestrator"""
    from agents.web_extraction_agent import WebExtractionAgent
    from agents.data_consolidation_agent import DataConsolidationAgent
    
    # Create mock agents
    agents = {
        "web_extractor": WebExtractionAgent(
            personality=create_test_personality("WebExtractor"),
            openai_client=mock_openai_client
        ),
        "data_consolidator": DataConsolidationAgent(
            personality=create_test_personality("DataConsolidator"),
            openai_client=mock_openai_client
        )
    }
    
    orchestrator = UniversalOrchestrator(agents)
    orchestrator.database = test_database
    
    return orchestrator

def create_test_personality(agent_type: str):
    """Create test agent personality"""
    from core.agent_personality import AgentPersonality
    
    personalities = {
        "WebExtractor": AgentPersonality(
            name="TestWebExtractor",
            role="Data extraction specialist",
            expertise=["web_scraping", "data_validation"],
            system_prompt="You are a test web extraction agent.",
            capability_tier="standard"
        ),
        "DataConsolidator": AgentPersonality(
            name="TestDataConsolidator", 
            role="Data consolidation specialist",
            expertise=["data_processing", "validation"],
            system_prompt="You are a test data consolidation agent.",
            capability_tier="standard"
        )
    }
    
    return personalities[agent_type]

class TestDataGenerator:
    """Generate realistic test data for various scenarios"""
    
    @staticmethod
    def create_revenue_dataset(num_podcasts: int = 10, num_months: int = 3):
        """Generate comprehensive revenue dataset"""
        import random
        from datetime import datetime, timedelta
        
        podcasts = [f"Podcast_{i}" for i in range(num_podcasts)]
        base_date = datetime(2024, 1, 1)
        
        dataset = {
            "megaphone": [],
            "audiomax": [],
            "audioserve": []
        }
        
        for month_offset in range(num_months):
            current_date = base_date + timedelta(days=30 * month_offset)
            
            for podcast in podcasts:
                # Megaphone data
                base_revenue = random.uniform(500, 5000)
                dataset["megaphone"].append({
                    "podcast_name": podcast,
                    "month": current_date.month,
                    "year": current_date.year,
                    "programmatic_revenue": Decimal(str(round(base_revenue, 2))),
                    "impressions": random.randint(10000, 100000),
                    "fill_rate": Decimal(str(round(random.uniform(0.7, 0.95), 3)))
                })
                
                # Audiomax data (typically lower than Megaphone)
                audiomax_revenue = base_revenue * random.uniform(0.6, 0.8)
                dataset["audiomax"].append({
                    "podcast_name": podcast,
                    "month": current_date.month,
                    "year": current_date.year,
                    "net_revenue": Decimal(str(round(audiomax_revenue, 2))),
                    "impressions": random.randint(8000, 80000)
                })
        
        return dataset
    
    @staticmethod
    def create_financial_test_cases():
        """Generate edge cases for financial calculations"""
        return [
            # Standard cases
            {"revenue": Decimal("1000.00"), "share": Decimal("0.70"), "expected_creator": Decimal("700.00")},
            {"revenue": Decimal("1234.56"), "share": Decimal("0.70"), "expected_creator": Decimal("864.19")},
            
            # Rounding edge cases
            {"revenue": Decimal("100.33"), "share": Decimal("0.70"), "expected_creator": Decimal("70.23")},
            {"revenue": Decimal("0.01"), "share": Decimal("0.70"), "expected_creator": Decimal("0.01")},
            
            # Large numbers
            {"revenue": Decimal("999999.99"), "share": Decimal("0.70"), "expected_creator": Decimal("699999.99")},
            
            # Edge percentages
            {"revenue": Decimal("1000.00"), "share": Decimal("0.00"), "expected_creator": Decimal("0.00")},
            {"revenue": Decimal("1000.00"), "share": Decimal("1.00"), "expected_creator": Decimal("1000.00")},
        ]
```

---

## Unit Testing Patterns

### Agent-Level Unit Tests

```python
# tests/unit/test_base_agent.py
import pytest
from unittest.mock import AsyncMock, patch
from decimal import Decimal

from core.base_agent import BaseRevenueAgent
from core.agent_personality import AgentPersonality

class TestBaseRevenueAgent:
    """Comprehensive tests for BaseRevenueAgent functionality"""
    
    @pytest.fixture
    def base_agent(self, mock_openai_client):
        personality = AgentPersonality(
            name="TestAgent",
            role="Test specialist",
            expertise=["testing", "validation"],
            system_prompt="You are a test agent",
            capability_tier="standard"
        )
        return BaseRevenueAgent(personality, mock_openai_client)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, base_agent):
        """Test agent initializes correctly"""
        assert base_agent.personality.name == "TestAgent"
        assert base_agent.performance_tracker is not None
        assert base_agent.validation_rules is not None
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, base_agent, mock_openai_client):
        """Test successful agent execution"""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = json.dumps({
            "status": "success",
            "result": "Test completed successfully",
            "confidence": 0.95
        })
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await base_agent.execute("Test task", {"context": "test"})
        
        assert result["status"] == "success"
        assert result["agent"] == "TestAgent"
        assert "content" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, base_agent, mock_openai_client):
        """Test agent error handling"""
        # Mock LLM failure
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            await base_agent.execute("Test task", {"context": "test"})
    
    @pytest.mark.asyncio
    async def test_validation_rules(self, base_agent):
        """Test input validation"""
        # Test with invalid input
        with pytest.raises(ValueError):
            await base_agent.execute("", {})  # Empty task
        
        # Test with valid input
        result = await base_agent.validate_input("Valid task", {"valid": "context"})
        assert result["task"] == "Valid task"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, base_agent, mock_openai_client):
        """Test performance metrics collection"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = json.dumps({
            "status": "success",
            "result": "Test result"
        })
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await base_agent.execute("Test task", {"context": "test"})
        
        # Check performance metrics
        metrics = base_agent.performance_tracker.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 1.0
        assert "average_execution_time" in metrics
    
    @pytest.mark.asyncio
    async def test_capability_tier_selection(self, base_agent):
        """Test model selection based on capability tier"""
        # Test simple task
        model = base_agent.select_model_for_task("Simple calculation", complexity="simple")
        assert model in ["gpt-4o-mini", "gpt-3.5-turbo"]
        
        # Test complex task
        model = base_agent.select_model_for_task("Complex analysis", complexity="complex")
        assert model in ["gpt-4o", "gpt-4"]
    
    def test_result_formatting(self, base_agent):
        """Test result formatting consistency"""
        raw_result = {
            "processed_data": {"value": 123},
            "metadata": {"confidence": 0.9}
        }
        
        formatted = base_agent.format_agent_response(raw_result)
        
        assert "status" in formatted
        assert "agent" in formatted
        assert "content" in formatted
        assert formatted["agent"] == base_agent.personality.name

class TestAgentPersonality:
    """Test agent personality configuration and behavior"""
    
    def test_personality_creation(self):
        """Test personality object creation"""
        personality = AgentPersonality(
            name="TestPersonality",
            role="Test role",
            expertise=["skill1", "skill2"],
            system_prompt="Test prompt",
            capability_tier="standard"
        )
        
        assert personality.name == "TestPersonality"
        assert len(personality.expertise) == 2
        assert personality.capability_tier == "standard"
    
    def test_capability_tier_validation(self):
        """Test capability tier validation"""
        # Valid tiers
        for tier in ["simple", "standard", "complex", "expert"]:
            personality = AgentPersonality(
                name="Test",
                role="Test",
                expertise=["test"],
                system_prompt="Test",
                capability_tier=tier
            )
            assert personality.capability_tier == tier
        
        # Invalid tier
        with pytest.raises(ValueError):
            AgentPersonality(
                name="Test",
                role="Test", 
                expertise=["test"],
                system_prompt="Test",
                capability_tier="invalid"
            )
    
    def test_prompt_generation(self):
        """Test system prompt generation"""
        personality = AgentPersonality(
            name="TestAgent",
            role="Data processor",
            expertise=["data_processing", "validation"],
            system_prompt="You are a data processing expert.",
            capability_tier="standard"
        )
        
        full_prompt = personality.get_full_system_prompt()
        
        assert "TestAgent" in full_prompt
        assert "data_processing" in full_prompt
        assert "You are a data processing expert." in full_prompt
```

### Financial Calculation Unit Tests

```python
# tests/unit/test_financial_calculations.py
import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

from agents.financial_calculation_agent import FinancialCalculationAgent
from core.financial_precision import PrecisionConfig, FinancialDataProcessor

class TestFinancialCalculations:
    """Test financial calculation accuracy and precision"""
    
    @pytest.fixture
    def precision_config(self):
        return PrecisionConfig(
            decimal_places=2,
            rounding_mode=ROUND_HALF_UP,
            currency="USD"
        )
    
    @pytest.fixture
    def financial_processor(self, precision_config):
        return FinancialDataProcessor(precision_config)
    
    def test_decimal_input_validation(self, financial_processor):
        """Test decimal input validation and conversion"""
        test_cases = [
            # (input, expected_output)
            ("100.50", Decimal("100.50")),
            (100.50, Decimal("100.50")),
            (100, Decimal("100.00")),
            ("0.01", Decimal("0.01")),
            ("999999.99", Decimal("999999.99"))
        ]
        
        for input_value, expected in test_cases:
            result = financial_processor.validate_input(input_value)
            assert result == expected, f"Failed for input {input_value}"
    
    def test_invalid_inputs(self, financial_processor):
        """Test handling of invalid financial inputs"""
        invalid_inputs = ["", "abc", "100.999", None, float('inf'), float('nan')]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                financial_processor.validate_input(invalid_input)
    
    def test_revenue_share_calculations(self, financial_processor):
        """Test revenue share calculations for accuracy"""
        test_cases = TestDataGenerator.create_financial_test_cases()
        
        for case in test_cases:
            result = financial_processor.calculate_revenue_share(
                case["revenue"], case["share"]
            )
            
            assert result["creator_share"] == case["expected_creator"]
            assert result["total"] == case["revenue"]
            
            # Verify total equals sum of parts
            calculated_total = result["creator_share"] + result["platform_share"]
            assert calculated_total == result["total"]
    
    def test_percentage_calculations(self, financial_processor):
        """Test percentage calculation precision"""
        # Test various percentage scenarios
        base_amount = Decimal("1000.00")
        
        test_percentages = [
            (Decimal("0.70"), Decimal("700.00")),
            (Decimal("0.333333"), Decimal("333.33")),
            (Decimal("0.666667"), Decimal("666.67")),
            (Decimal("0.125"), Decimal("125.00"))
        ]
        
        for percentage, expected in test_percentages:
            result = financial_processor.calculate_percentage(base_amount, percentage)
            assert result == expected
    
    def test_currency_conversion(self, financial_processor):
        """Test currency conversion accuracy"""
        usd_amount = Decimal("100.00")
        exchange_rate = Decimal("1.25")  # USD to EUR
        
        converted = financial_processor.convert_currency(
            usd_amount, "USD", "EUR", exchange_rate
        )
        
        assert converted["original_amount"] == usd_amount
        assert converted["converted_amount"] == Decimal("80.00")
        assert converted["exchange_rate"] == exchange_rate
    
    def test_rounding_consistency(self, financial_processor):
        """Test consistent rounding behavior"""
        # Test borderline rounding cases
        test_values = [
            (Decimal("100.505"), Decimal("100.51")),  # Round up
            (Decimal("100.504"), Decimal("100.50")),  # Round down
            (Decimal("100.515"), Decimal("100.52")),  # Round up
        ]
        
        for input_value, expected in test_values:
            result = financial_processor.apply_rounding(input_value)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_financial_agent_integration(self, test_database, mock_openai_client):
        """Test financial agent with realistic workflow"""
        personality = create_test_personality("FinancialCalculator")
        agent = FinancialCalculationAgent(personality, mock_openai_client)
        
        # Mock LLM response for financial calculations
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = json.dumps({
            "status": "success",
            "calculations": {
                "total_revenue": "1500.00",
                "creator_share": "1050.00",
                "platform_share": "450.00"
            },
            "confidence": 0.98
        })
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        context = {
            "revenue_data": [
                {"amount": "1000.00", "share_rate": "0.70"},
                {"amount": "500.00", "share_rate": "0.70"}
            ]
        }
        
        result = await agent.execute("Calculate revenue shares", context)
        
        assert result["status"] == "success"
        assert "calculations" in result["content"]
    
    def test_audit_trail_creation(self, financial_processor):
        """Test financial audit trail generation"""
        financial_processor.config.audit_trail_enabled = True
        
        # Perform operations that should be audited
        financial_processor.validate_input("100.50")
        financial_processor.calculate_percentage(Decimal("100.00"), Decimal("0.70"))
        
        audit_trail = financial_processor.audit_trail.get_trail()
        
        assert len(audit_trail) >= 2
        assert all("operation" in entry for entry in audit_trail)
        assert all("timestamp" in entry for entry in audit_trail)
```

---

## Integration Testing Patterns

### Agent Coordination Tests

```python
# tests/integration/test_workflow_orchestration.py
import pytest
from unittest.mock import AsyncMock, patch
import asyncio

from core.orchestrator import UniversalOrchestrator
from core.context_manager import SharedContextManager

class TestWorkflowOrchestration:
    """Test agent coordination and workflow execution"""
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, test_orchestrator, sample_revenue_data):
        """Test basic workflow coordination"""
        workflow_config = {
            "stages": [
                {
                    "name": "data_extraction",
                    "agent_name": "web_extractor",
                    "task": "Extract revenue data from Megaphone",
                    "context": {"source": "megaphone"}
                },
                {
                    "name": "data_consolidation", 
                    "agent_name": "data_consolidator",
                    "task": "Consolidate and validate revenue data",
                    "context": {"validation_rules": "standard"}
                }
            ]
        }
        
        # Mock agent responses
        extraction_result = {
            "status": "success",
            "data": sample_revenue_data["megaphone_data"],
            "confidence": 0.95
        }
        
        consolidation_result = {
            "status": "success", 
            "consolidated_data": sample_revenue_data["megaphone_data"],
            "quality_score": 0.96
        }
        
        # Configure mocks
        test_orchestrator.agents["web_extractor"].execute = AsyncMock(return_value=extraction_result)
        test_orchestrator.agents["data_consolidator"].execute = AsyncMock(return_value=consolidation_result)
        
        # Execute workflow
        result = await test_orchestrator.execute_workflow(workflow_config)
        
        assert result["status"] == "success"
        assert len(result["stage_results"]) == 2
        assert result["stage_results"][0] == extraction_result
        assert result["stage_results"][1] == consolidation_result
    
    @pytest.mark.asyncio
    async def test_context_sharing_between_agents(self, test_orchestrator):
        """Test context sharing across workflow stages"""
        workflow_config = {
            "context_sharing_enabled": True,
            "stages": [
                {
                    "name": "stage_1",
                    "agent_name": "web_extractor", 
                    "task": "Process initial data",
                    "share_context": True
                },
                {
                    "name": "stage_2",
                    "agent_name": "data_consolidator",
                    "task": "Use data from stage 1",
                    "requires_context": ["stage_1"]
                }
            ]
        }
        
        # Mock first agent to return context data
        stage_1_result = {
            "status": "success",
            "data": {"extracted_value": 123},
            "context_data": {"shared_info": "important_value"}
        }
        
        stage_2_result = {
            "status": "success",
            "processed_data": {"final_value": 456}
        }
        
        test_orchestrator.agents["web_extractor"].execute = AsyncMock(return_value=stage_1_result)
        test_orchestrator.agents["data_consolidator"].execute = AsyncMock(return_value=stage_2_result)
        
        result = await test_orchestrator.execute_workflow(workflow_config)
        
        # Verify context was passed to second agent
        second_agent_call = test_orchestrator.agents["data_consolidator"].execute.call_args
        passed_context = second_agent_call[0][1]  # Second argument (context)
        
        assert "stage_1" in passed_context
        assert passed_context["stage_1"]["shared_info"] == "important_value"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_orchestrator):
        """Test workflow error handling and recovery"""
        workflow_config = {
            "error_recovery_enabled": True,
            "stages": [
                {
                    "name": "failing_stage",
                    "agent_name": "web_extractor",
                    "task": "This will fail",
                    "retry_config": {"max_retries": 2, "delay": 0.1}
                },
                {
                    "name": "recovery_stage",
                    "agent_name": "data_consolidator", 
                    "task": "Handle the failure",
                    "depends_on": ["failing_stage"],
                    "run_on_failure": True
                }
            ]
        }
        
        # Mock failing agent
        test_orchestrator.agents["web_extractor"].execute = AsyncMock(
            side_effect=Exception("Simulated failure")
        )
        
        # Mock recovery agent
        recovery_result = {
            "status": "success",
            "message": "Handled failure gracefully"
        }
        test_orchestrator.agents["data_consolidator"].execute = AsyncMock(
            return_value=recovery_result
        )
        
        result = await test_orchestrator.execute_workflow(workflow_config)
        
        # Verify recovery was executed
        assert result["status"] == "partial_success"
        assert "failing_stage" in result["failed_stages"]
        assert result["stage_results"][-1] == recovery_result
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, test_orchestrator):
        """Test parallel stage execution"""
        workflow_config = {
            "parallel_execution_enabled": True,
            "stages": [
                {
                    "name": "parallel_stage_1",
                    "agent_name": "web_extractor",
                    "task": "Extract data source 1",
                    "can_run_parallel": True,
                    "parallel_group": "data_extraction"
                },
                {
                    "name": "parallel_stage_2", 
                    "agent_name": "data_consolidator",
                    "task": "Extract data source 2",
                    "can_run_parallel": True,
                    "parallel_group": "data_extraction"
                }
            ]
        }
        
        # Mock both agents with delays to verify parallel execution
        async def mock_stage_1(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "source_1"}
        
        async def mock_stage_2(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "source_2"}
        
        test_orchestrator.agents["web_extractor"].execute = mock_stage_1
        test_orchestrator.agents["data_consolidator"].execute = mock_stage_2
        
        start_time = asyncio.get_event_loop().time()
        result = await test_orchestrator.execute_workflow(workflow_config)
        end_time = asyncio.get_event_loop().time()
        
        # Verify parallel execution (should be faster than sequential)
        execution_time = end_time - start_time
        assert execution_time < 0.15  # Should be ~0.1s, not ~0.2s
        
        assert result["status"] == "success"
        assert len(result["stage_results"]) == 2

class TestDatabaseOperations:
    """Test database operations across agents"""
    
    @pytest.mark.asyncio
    async def test_database_transaction_consistency(self, test_database):
        """Test database transaction handling"""
        # Test data insertion and rollback
        async with test_database.transaction():
            await test_database.insert_revenue_record({
                "podcast_name": "Test Podcast",
                "month": 11,
                "year": 2024,
                "revenue": Decimal("100.00")
            })
            
            # Verify data exists within transaction
            records = await test_database.get_revenue_records(11, 2024)
            assert len(records) == 1
            
            # Simulate error to trigger rollback
            raise Exception("Simulated error")
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, test_database):
        """Test concurrent database access"""
        async def insert_records(start_id: int, count: int):
            for i in range(count):
                await test_database.insert_revenue_record({
                    "id": start_id + i,
                    "podcast_name": f"Podcast_{start_id + i}",
                    "month": 11,
                    "year": 2024,
                    "revenue": Decimal("100.00")
                })
        
        # Run concurrent insertions
        tasks = [
            insert_records(0, 10),
            insert_records(10, 10),
            insert_records(20, 10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all records were inserted
        records = await test_database.get_revenue_records(11, 2024)
        assert len(records) == 30
```

---

## End-to-End Testing Patterns

### Complete Workflow Tests

```python
# tests/e2e/test_complete_workflow.py
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
from decimal import Decimal

class TestCompleteWorkflow:
    """End-to-end workflow testing"""
    
    @pytest.mark.asyncio
    async def test_full_revenue_reporting_workflow(self, test_orchestrator, sample_revenue_data):
        """Test complete revenue reporting workflow from start to finish"""
        
        # Mock external services
        with patch('selenium.webdriver.Chrome') as mock_chrome, \
             patch('smtplib.SMTP') as mock_smtp, \
             patch('requests.get') as mock_requests:
            
            # Configure Chrome mock for web scraping
            mock_driver = MagicMock()
            mock_driver.find_elements.return_value = self._create_mock_web_elements()
            mock_chrome.return_value.__enter__.return_value = mock_driver
            
            # Configure email mock
            mock_smtp_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
            
            # Configure exchange rate API mock
            mock_requests.return_value.json.return_value = {
                "rates": {"USD": 1.0, "EUR": 0.85}
            }
            
            # Execute complete workflow
            workflow_config = {
                "operation": "generate_monthly_report",
                "parameters": {
                    "month": 11,
                    "year": 2024,
                    "output_formats": ["pdf", "excel"],
                    "email_recipients": ["test@example.com"]
                }
            }
            
            result = await test_orchestrator.execute_complete_workflow(workflow_config)
            
            # Verify workflow completion
            assert result["status"] == "success"
            assert "execution_time" in result
            assert len(result["generated_files"]) == 2
            
            # Verify each stage completed
            expected_stages = [
                "data_extraction",
                "data_processing", 
                "financial_calculations",
                "report_generation",
                "email_distribution"
            ]
            
            for stage in expected_stages:
                assert stage in result["completed_stages"]
                assert result["stage_results"][stage]["status"] == "success"
    
    def _create_mock_web_elements(self):
        """Create mock web elements for Selenium testing"""
        elements = []
        
        # Mock revenue data elements
        for i in range(3):
            element = MagicMock()
            element.text = f"Podcast_{i}: $1,{i*100+500}.{i*10+25}"
            elements.append(element)
        
        return elements
    
    @pytest.mark.asyncio
    async def test_workflow_with_data_quality_issues(self, test_orchestrator):
        """Test workflow handling of data quality issues"""
        
        # Inject data quality issues
        problematic_data = {
            "megaphone_data": [
                {
                    "podcast_name": "Podcast With Issues",
                    "month": 11,
                    "year": 2024,
                    "programmatic_revenue": "invalid_number",  # Quality issue
                    "impressions": -1000  # Invalid impressions
                }
            ]
        }
        
        # Mock agents to return problematic data
        with patch.object(test_orchestrator.agents["web_extractor"], 'execute') as mock_extract:
            mock_extract.return_value = {
                "status": "success",
                "data": problematic_data["megaphone_data"],
                "quality_warnings": ["Invalid revenue format", "Negative impressions"]
            }
            
            workflow_config = {
                "operation": "process_data_with_validation",
                "parameters": {"strict_validation": True}
            }
            
            result = await test_orchestrator.execute_workflow_with_validation(workflow_config)
            
            # Verify quality issues were detected and handled
            assert "data_quality_issues" in result
            assert len(result["data_quality_issues"]) > 0
            assert result["status"] in ["partial_success", "failed"]
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, test_orchestrator):
        """Test system performance under load"""
        import time
        
        # Generate large dataset
        large_dataset = TestDataGenerator.create_revenue_dataset(
            num_podcasts=100, num_months=12
        )
        
        start_time = time.time()
        
        # Process large dataset
        workflow_config = {
            "operation": "process_large_dataset", 
            "parameters": {
                "dataset": large_dataset,
                "performance_monitoring": True
            }
        }
        
        result = await test_orchestrator.execute_workflow(workflow_config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert result["status"] == "success"
        assert result["records_processed"] == 1200  # 100 podcasts * 12 months
        
        # Check memory usage didn't exceed limits
        if "memory_usage" in result:
            assert result["memory_usage"]["peak_mb"] < 500  # Keep under 500MB
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, test_orchestrator):
        """Test various error recovery scenarios"""
        
        error_scenarios = [
            {
                "name": "database_connection_failure",
                "error": "Database connection lost",
                "recovery_expected": True
            },
            {
                "name": "external_api_timeout", 
                "error": "Exchange rate API timeout",
                "recovery_expected": True
            },
            {
                "name": "file_system_error",
                "error": "Disk space full",
                "recovery_expected": False
            }
        ]
        
        for scenario in error_scenarios:
            with patch('asyncio.sleep', return_value=None):  # Speed up retries
                
                # Inject specific error
                with patch.object(test_orchestrator, '_inject_error') as mock_inject:
                    mock_inject.return_value = scenario["error"]
                    
                    workflow_config = {
                        "operation": "test_error_recovery",
                        "parameters": {
                            "error_scenario": scenario["name"],
                            "recovery_enabled": True
                        }
                    }
                    
                    result = await test_orchestrator.execute_workflow_with_recovery(workflow_config)
                    
                    if scenario["recovery_expected"]:
                        assert result["status"] in ["success", "partial_success"]
                        assert "recovery_actions" in result
                    else:
                        assert result["status"] == "failed"
                        assert scenario["error"] in str(result.get("error", ""))

class TestReportGeneration:
    """Test report generation and output validation"""
    
    @pytest.mark.asyncio
    async def test_pdf_report_generation(self, test_orchestrator, sample_revenue_data):
        """Test PDF report generation with real data"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_config = {
                "operation": "generate_pdf_report",
                "parameters": {
                    "output_directory": temp_dir,
                    "data": sample_revenue_data,
                    "include_charts": True
                }
            }
            
            result = await test_orchestrator.execute_report_generation(workflow_config)
            
            # Verify PDF was created
            assert result["status"] == "success"
            pdf_path = result["generated_files"]["pdf"]
            assert os.path.exists(pdf_path)
            
            # Basic PDF validation
            file_size = os.path.getsize(pdf_path)
            assert file_size > 1000  # PDF should be substantial
            
            # Verify PDF contains expected content (if PDF reader available)
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                    
                    assert "Tech Talk Daily" in text_content
                    assert "Revenue Report" in text_content
            except ImportError:
                # PDF reader not available, skip content validation
                pass
    
    @pytest.mark.asyncio
    async def test_excel_report_generation(self, test_orchestrator, sample_revenue_data):
        """Test Excel report generation and validation"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_config = {
                "operation": "generate_excel_report",
                "parameters": {
                    "output_directory": temp_dir,
                    "data": sample_revenue_data,
                    "include_formulas": True
                }
            }
            
            result = await test_orchestrator.execute_report_generation(workflow_config)
            
            # Verify Excel file was created
            assert result["status"] == "success"
            excel_path = result["generated_files"]["excel"]
            assert os.path.exists(excel_path)
            
            # Validate Excel content
            try:
                import pandas as pd
                
                # Read main data sheet
                df = pd.read_excel(excel_path, sheet_name="Revenue Summary")
                assert len(df) > 0
                assert "Podcast Name" in df.columns
                assert "Revenue" in df.columns
                
                # Verify calculations
                total_revenue = df["Revenue"].sum()
                assert total_revenue > 0
                
            except ImportError:
                # Pandas not available, skip content validation
                pass
```

---

## Performance and Load Testing

### Performance Test Patterns

```python
# tests/performance/test_performance.py
import pytest
import asyncio
import time
import psutil
import memory_profiler
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance and scalability testing"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, test_orchestrator):
        """Test system performance under concurrent load"""
        
        # Configuration for concurrent tests
        num_concurrent_workflows = 10
        workflow_duration_limit = 30  # seconds
        
        async def execute_workflow(workflow_id: int):
            start_time = time.time()
            
            workflow_config = {
                "operation": "standard_workflow",
                "parameters": {
                    "workflow_id": workflow_id,
                    "data_size": "medium"
                }
            }
            
            result = await test_orchestrator.execute_workflow(workflow_config)
            
            execution_time = time.time() - start_time
            
            return {
                "workflow_id": workflow_id,
                "execution_time": execution_time,
                "status": result["status"],
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
            }
        
        # Execute concurrent workflows
        start_time = time.time()
        
        tasks = [execute_workflow(i) for i in range(num_concurrent_workflows)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_workflows = [r for r in results if r["status"] == "success"]
        average_execution_time = sum(r["execution_time"] for r in results) / len(results)
        max_memory_usage = max(r["memory_usage"] for r in results)
        
        # Performance assertions
        assert len(successful_workflows) == num_concurrent_workflows, "All workflows should succeed"
        assert total_time < workflow_duration_limit, f"Total time {total_time}s exceeded limit"
        assert average_execution_time < 10, f"Average execution time {average_execution_time}s too high"
        assert max_memory_usage < 500, f"Memory usage {max_memory_usage}MB too high"
    
    @pytest.mark.performance
    def test_memory_efficiency(self, test_orchestrator):
        """Test memory usage and garbage collection"""
        
        @memory_profiler.profile
        def memory_test_function():
            # Simulate memory-intensive operations
            large_datasets = []
            
            for i in range(100):
                dataset = TestDataGenerator.create_revenue_dataset(
                    num_podcasts=50, num_months=12
                )
                large_datasets.append(dataset)
                
                # Process dataset
                processed = test_orchestrator.process_dataset_synchronous(dataset)
                
                # Clean up to test garbage collection
                if i % 10 == 0:
                    large_datasets.clear()
            
            return len(large_datasets)
        
        # Monitor memory during execution
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = memory_test_function()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_increase < 200, f"Memory increased by {memory_increase}MB, too high"
        assert result >= 0, "Function should complete successfully"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_database_performance(self, test_database):
        """Test database operation performance"""
        
        # Prepare large dataset
        num_records = 10000
        large_dataset = []
        
        for i in range(num_records):
            record = {
                "id": i,
                "podcast_name": f"Podcast_{i % 100}",  # 100 unique podcasts
                "month": (i % 12) + 1,
                "year": 2024,
                "revenue": Decimal(f"{100 + (i % 1000)}.{i % 100:02d}"),
                "impressions": 1000 + (i % 10000)
            }
            large_dataset.append(record)
        
        # Test bulk insert performance
        start_time = time.time()
        
        await test_database.bulk_insert_revenue_records(large_dataset)
        
        insert_time = time.time() - start_time
        
        # Test query performance
        query_start = time.time()
        
        results = await test_database.get_revenue_records_with_aggregation(
            start_month=1, end_month=12, year=2024
        )
        
        query_time = time.time() - query_start
        
        # Performance assertions
        assert insert_time < 30, f"Bulk insert took {insert_time}s, too slow"
        assert query_time < 5, f"Query took {query_time}s, too slow"
        assert len(results) > 0, "Query should return results"
        
        # Verify data integrity
        total_inserted = await test_database.count_records(year=2024)
        assert total_inserted == num_records, "All records should be inserted"
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_financial_calculation_performance(self, benchmark, financial_processor):
        """Benchmark financial calculation performance"""
        
        def calculate_large_batch():
            results = []
            
            # Process 1000 revenue calculations
            for i in range(1000):
                revenue = Decimal(f"{100 + i}.{i % 100:02d}")
                share_rate = Decimal("0.70")
                
                result = financial_processor.calculate_revenue_share(revenue, share_rate)
                results.append(result)
            
            return results
        
        # Benchmark the calculation function
        results = benchmark(calculate_large_batch)
        
        # Verify results
        assert len(results) == 1000
        assert all(r["total"] > 0 for r in results)
        
        # Performance should be under 1 second for 1000 calculations
        assert benchmark.stats["mean"] < 1.0, "Calculations taking too long"

class TestScalability:
    """Test system scalability and resource management"""
    
    @pytest.mark.scalability
    @pytest.mark.parametrize("dataset_size", [10, 100, 1000, 5000])
    @pytest.mark.asyncio
    async def test_scalability_by_dataset_size(self, test_orchestrator, dataset_size):
        """Test how system scales with increasing dataset sizes"""
        
        # Generate dataset of specified size
        dataset = TestDataGenerator.create_revenue_dataset(
            num_podcasts=dataset_size // 10, num_months=10
        )
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process dataset
        workflow_config = {
            "operation": "process_scalability_test",
            "parameters": {
                "dataset": dataset,
                "dataset_size": dataset_size
            }
        }
        
        result = await test_orchestrator.execute_workflow(workflow_config)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Record scalability metrics
        metrics = {
            "dataset_size": dataset_size,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "throughput": dataset_size / execution_time if execution_time > 0 else 0
        }
        
        # Scalability assertions (should scale reasonably)
        if dataset_size <= 100:
            assert execution_time < 10, f"Small dataset taking too long: {execution_time}s"
            assert memory_used < 50, f"Small dataset using too much memory: {memory_used}MB"
        elif dataset_size <= 1000:
            assert execution_time < 60, f"Medium dataset taking too long: {execution_time}s"
            assert memory_used < 200, f"Medium dataset using too much memory: {memory_used}MB"
        else:
            assert execution_time < 300, f"Large dataset taking too long: {execution_time}s"
            assert memory_used < 500, f"Large dataset using too much memory: {memory_used}MB"
        
        assert result["status"] == "success", f"Processing failed for dataset size {dataset_size}"
```

---

## Test Automation and CI/CD

### Test Runner Configuration

```python
# run_tests.py - Custom test runner
#!/usr/bin/env python3
import subprocess
import sys
import argparse
import json
import time
from pathlib import Path

class TestRunner:
    """Custom test runner with reporting and analysis"""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
    
    def run_test_suite(self, test_type="all", coverage=False, verbose=False):
        """Run specified test suite with options"""
        
        commands = {
            "unit": ["pytest", "tests/unit/", "-v"],
            "integration": ["pytest", "tests/integration/", "-v"],
            "e2e": ["pytest", "tests/e2e/", "-v", "-s"],
            "performance": ["pytest", "tests/performance/", "-v", "-m", "performance"],
            "all": ["pytest", "tests/", "-v"]
        }
        
        if test_type not in commands:
            raise ValueError(f"Invalid test type: {test_type}")
        
        cmd = commands[test_type]
        
        # Add coverage if requested
        if coverage:
            cmd.extend(["--cov=core", "--cov=agents", "--cov-report=html", "--cov-report=term"])
        
        # Add junit XML output for CI
        cmd.extend(["--junit-xml=test_results.xml"])
        
        print(f"🚀 Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            self.results[test_type] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": time.time() - self.start_time
            }
            
            if result.returncode == 0:
                print(f"✅ {test_type} tests passed!")
            else:
                print(f"❌ {test_type} tests failed!")
                if verbose:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_type} tests timed out!")
            return False
    
    def run_quality_checks(self):
        """Run code quality checks"""
        
        quality_commands = {
            "lint": ["flake8", "core/", "agents/", "tests/"],
            "type_check": ["mypy", "core/", "agents/"],
            "security": ["bandit", "-r", "core/", "agents/"]
        }
        
        all_passed = True
        
        for check_name, cmd in quality_commands.items():
            print(f"🔍 Running {check_name}...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {check_name} passed!")
                else:
                    print(f"❌ {check_name} failed!")
                    print(result.stdout)
                    all_passed = False
                    
            except FileNotFoundError:
                print(f"⚠️ {check_name} tool not found, skipping...")
        
        return all_passed
    
    def generate_report(self, output_file="test_report.json"):
        """Generate comprehensive test report"""
        
        report = {
            "timestamp": time.time(),
            "total_execution_time": time.time() - self.start_time,
            "test_results": self.results,
            "summary": self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Test report saved to {output_file}")
    
    def _generate_summary(self):
        """Generate test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["returncode"] == 0)
        
        return {
            "total_test_suites": total_tests,
            "passed_test_suites": passed_tests,
            "failed_test_suites": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }

def main():
    parser = argparse.ArgumentParser(description="Run agent system tests")
    parser.add_argument("--type", choices=["unit", "integration", "e2e", "performance", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--quality", action="store_true", help="Run quality checks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--report", default="test_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Run tests
    test_success = runner.run_test_suite(args.type, args.coverage, args.verbose)
    
    # Run quality checks if requested
    quality_success = True
    if args.quality:
        quality_success = runner.run_quality_checks()
    
    # Generate report
    runner.generate_report(args.report)
    
    # Exit with appropriate code
    if test_success and quality_success:
        print("🎉 All tests and checks passed!")
        sys.exit(0)
    else:
        print("💥 Some tests or checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### GitHub Actions CI Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_sonoro_revenue
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install Chrome (for Selenium tests)
      uses: browser-actions/setup-chrome@latest
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_sonoro_revenue
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python run_tests.py --type unit --coverage --verbose
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_sonoro_revenue
      run: |
        python run_tests.py --type integration --verbose
    
    - name: Run quality checks
      run: |
        python run_tests.py --quality
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          test_results.xml
          test_report.json
          htmlcov/
```

---

## Best Practices Summary

### Testing Strategy Guidelines

1. **Test Pyramid Structure**: More unit tests, fewer integration tests, minimal E2E tests
2. **Async Testing**: Use proper async fixtures and patterns throughout
3. **Mock External Services**: Never depend on external APIs or services in tests
4. **Realistic Test Data**: Use representative data that matches production scenarios
5. **Performance Monitoring**: Include performance assertions in all test levels
6. **Error Scenario Coverage**: Test failure modes and recovery mechanisms
7. **Continuous Integration**: Automate testing with comprehensive CI pipelines

### Test Organization Principles

- **Isolation**: Each test runs independently with fresh state
- **Repeatability**: Tests produce consistent results across runs
- **Speed**: Unit tests run in milliseconds, integration tests in seconds
- **Clarity**: Test names and assertions clearly indicate intent
- **Coverage**: Achieve high coverage while focusing on critical paths

---

## Next Steps

- **Context Management**: [Shared State Patterns](context_management.md)
- **Implementation**: [Progressive Complexity Levels](../05_implementation_levels/)
- **Production**: [Level 4 Production Systems](../05_implementation_levels/level_4_production.md)

---

*This comprehensive testing framework ensures reliability, performance, and maintainability of agent systems across all complexity levels and deployment scenarios.*
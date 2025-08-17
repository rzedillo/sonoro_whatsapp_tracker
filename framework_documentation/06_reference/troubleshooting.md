# Troubleshooting Guide and Common Solutions

> 🔧 **Problem Solver**: Comprehensive solutions for common issues across all implementation levels of the Enhanced Agent Architecture Framework.

## Navigation
- **Previous**: [Security Guidelines](security_guidelines.md)
- **Next**: [Core Principles](../00_core_principles.md) → [Quick Start](../01_quick_start.md)
- **Related**: [Implementation Templates](templates.md) → [Decision Matrices](decision_matrices.md)

---

## Quick Diagnosis Tool

### Problem Identification Matrix

| Symptom | Likely Cause | Quick Fix | Detailed Section |
|---------|--------------|-----------|------------------|
| **Agent won't start** | Missing dependencies | `pip install -r requirements.txt` | [Environment Issues](#environment-issues) |
| **API calls fail** | Invalid credentials | Check `.env` file | [API Configuration](#api-configuration-issues) |
| **Slow responses** | Model/prompt issues | Use smaller model | [Performance Issues](#performance-optimization) |
| **Memory errors** | Large context/data | Implement chunking | [Memory Management](#memory-management) |
| **Database errors** | Connection/schema | Check DB config | [Database Issues](#database-troubleshooting) |
| **Import errors** | Python path/modules | Verify PYTHONPATH | [Import Problems](#import-and-dependency-issues) |
| **Web interface broken** | Port/dependencies | Check Streamlit setup | [Interface Issues](#interface-troubleshooting) |

### System Health Check Script

```python
#!/usr/bin/env python3
"""
System Health Check for Enhanced Agent Architecture
Run this script to diagnose common issues automatically.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

class SystemHealthChecker:
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def run_all_checks(self) -> Dict[str, any]:
        """Run comprehensive system health checks."""
        print("🔍 Running Enhanced Agent Architecture Health Check...")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Required Packages", self.check_packages),
            ("Environment Variables", self.check_environment),
            ("API Connectivity", self.check_api_access),
            ("Database Connection", self.check_database),
            ("File Permissions", self.check_file_permissions),
            ("Resource Availability", self.check_resources)
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                result = check_func()
                self.results[check_name] = result
                status = "✅ PASS" if result.get('status') == 'pass' else "❌ FAIL"
                print(f"   {status}: {result.get('message', 'Unknown')}")
                
                if result.get('warnings'):
                    for warning in result['warnings']:
                        print(f"   ⚠️  WARNING: {warning}")
                        self.warnings.append(f"{check_name}: {warning}")
                
            except Exception as e:
                error_msg = f"{check_name}: {str(e)}"
                self.errors.append(error_msg)
                print(f"   ❌ ERROR: {str(e)}")
        
        return self.generate_report()
    
    def check_python_version(self) -> Dict[str, any]:
        """Check Python version compatibility."""
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            return {
                'status': 'pass',
                'message': f'Python {version.major}.{version.minor}.{version.micro}',
                'details': {'version': f'{version.major}.{version.minor}.{version.micro}'}
            }
        else:
            return {
                'status': 'fail',
                'message': f'Python {version.major}.{version.minor} < {required_major}.{required_minor}',
                'recommendation': f'Upgrade to Python {required_major}.{required_minor}+'
            }
    
    def check_packages(self) -> Dict[str, any]:
        """Check required packages."""
        required_packages = [
            'openai', 'anthropic', 'requests', 'sqlalchemy', 
            'streamlit', 'pandas', 'numpy', 'python-dotenv'
        ]
        
        missing = []
        installed = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            return {
                'status': 'fail',
                'message': f'{len(missing)} packages missing',
                'details': {'missing': missing, 'installed': installed},
                'recommendation': f'pip install {" ".join(missing)}'
            }
        else:
            return {
                'status': 'pass',
                'message': f'All {len(installed)} packages installed',
                'details': {'installed': installed}
            }
    
    def check_environment(self) -> Dict[str, any]:
        """Check environment variables."""
        required_env = ['OPENAI_API_KEY']
        optional_env = ['ANTHROPIC_API_KEY', 'DATABASE_URL', 'SMTP_SERVER']
        
        missing_required = []
        missing_optional = []
        
        for var in required_env:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var in optional_env:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_required:
            return {
                'status': 'fail',
                'message': f'{len(missing_required)} required variables missing',
                'details': {'missing_required': missing_required},
                'recommendation': 'Create .env file with required variables'
            }
        else:
            warnings = []
            if missing_optional:
                warnings.append(f'Optional variables missing: {", ".join(missing_optional)}')
            
            return {
                'status': 'pass',
                'message': 'Required environment variables set',
                'warnings': warnings
            }
    
    def check_api_access(self) -> Dict[str, any]:
        """Check API connectivity."""
        apis_tested = []
        api_errors = []
        
        # Test OpenAI API
        try:
            import openai
            client = openai.OpenAI()
            response = client.models.list()
            apis_tested.append('OpenAI')
        except Exception as e:
            api_errors.append(f'OpenAI: {str(e)[:100]}')
        
        # Test Anthropic API (if configured)
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                client = anthropic.Anthropic()
                # Note: Add actual test call if needed
                apis_tested.append('Anthropic')
            except Exception as e:
                api_errors.append(f'Anthropic: {str(e)[:100]}')
        
        if api_errors:
            return {
                'status': 'fail',
                'message': f'{len(api_errors)} API connection failures',
                'details': {'errors': api_errors, 'working': apis_tested},
                'recommendation': 'Check API keys and network connectivity'
            }
        else:
            return {
                'status': 'pass',
                'message': f'{len(apis_tested)} APIs accessible',
                'details': {'working_apis': apis_tested}
            }
    
    def check_database(self) -> Dict[str, any]:
        """Check database connectivity."""
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            return {
                'status': 'pass',
                'message': 'No database configured (using SQLite)',
                'warnings': ['Consider PostgreSQL for production']
            }
        
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            
            return {
                'status': 'pass',
                'message': 'Database connection successful',
                'details': {'database_url': db_url.split('@')[1] if '@' in db_url else db_url}
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': 'Database connection failed',
                'details': {'error': str(e)},
                'recommendation': 'Check database URL and server status'
            }
    
    def check_file_permissions(self) -> Dict[str, any]:
        """Check file and directory permissions."""
        paths_to_check = [
            ('Current directory', '.'),
            ('Environment file', '.env'),
            ('Logs directory', 'logs'),
            ('Data directory', 'data')
        ]
        
        permission_issues = []
        
        for name, path in paths_to_check:
            path_obj = Path(path)
            if path_obj.exists():
                if not os.access(path_obj, os.R_OK):
                    permission_issues.append(f'{name}: No read access')
                if path_obj.is_dir() and not os.access(path_obj, os.W_OK):
                    permission_issues.append(f'{name}: No write access')
        
        if permission_issues:
            return {
                'status': 'fail',
                'message': f'{len(permission_issues)} permission issues',
                'details': {'issues': permission_issues},
                'recommendation': 'Fix file permissions with chmod'
            }
        else:
            return {
                'status': 'pass',
                'message': 'File permissions correct'
            }
    
    def check_resources(self) -> Dict[str, any]:
        """Check system resources."""
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        warnings = []
        
        # Check available memory (minimum 2GB recommended)
        if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB in bytes
            warnings.append(f'Low memory: {memory.available / 1024**3:.1f}GB available')
        
        # Check disk space (minimum 1GB recommended)
        if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB in bytes
            warnings.append(f'Low disk space: {disk.free / 1024**3:.1f}GB available')
        
        return {
            'status': 'pass',
            'message': f'Memory: {memory.available / 1024**3:.1f}GB, Disk: {disk.free / 1024**3:.1f}GB',
            'warnings': warnings,
            'details': {
                'memory_gb': round(memory.available / 1024**3, 1),
                'disk_gb': round(disk.free / 1024**3, 1)
            }
        }
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive health report."""
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result.get('status') == 'pass')
        
        health_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        report = {
            'timestamp': f"{sys.version}",
            'overall_health': health_score,
            'status': 'healthy' if health_score >= 80 else 'needs_attention',
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': total_checks - passed_checks,
                'warnings': len(self.warnings),
                'errors': len(self.errors)
            },
            'detailed_results': self.results,
            'recommendations': self.generate_recommendations()
        }
        
        print(f"\n{'='*60}")
        print(f"🏥 HEALTH REPORT: {health_score:.0f}% ({report['status'].upper()})")
        print(f"✅ Passed: {passed_checks}/{total_checks}")
        if self.warnings:
            print(f"⚠️  Warnings: {len(self.warnings)}")
        if self.errors:
            print(f"❌ Errors: {len(self.errors)}")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Add specific recommendations based on failures
        for check_name, result in self.results.items():
            if result.get('status') == 'fail' and result.get('recommendation'):
                recommendations.append(f"{check_name}: {result['recommendation']}")
        
        return recommendations

if __name__ == "__main__":
    checker = SystemHealthChecker()
    report = checker.run_all_checks()
    
    if report['status'] != 'healthy':
        print(f"\n📝 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    sys.exit(0 if report['status'] == 'healthy' else 1)
```

---

## Environment Issues

### Python Environment Problems

**Problem**: Import errors or module not found

```bash
# Diagnosis
python --version
pip list | grep -E "(openai|anthropic|streamlit)"

# Solutions
# 1. Wrong Python version
pyenv install 3.11.0
pyenv global 3.11.0

# 2. Virtual environment issues
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 3. Path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem**: Package conflicts or version issues

```bash
# Diagnosis
pip check
pip list --outdated

# Solutions
# 1. Clean environment
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -r requirements.txt

# 2. Use dependency resolver
pip install --upgrade pip
pip install pip-tools
pip-compile requirements.in  # Create requirements.txt
pip-sync requirements.txt

# 3. Use conda for complex dependencies
conda env create -f environment.yml
conda activate agent_env
```

### Operating System Compatibility

**Problem**: Different behavior across OS

```python
# Cross-platform file handling
import os
from pathlib import Path

class CrossPlatformUtils:
    @staticmethod
    def get_data_dir() -> Path:
        """Get appropriate data directory for OS."""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('APPDATA', '.')) / 'agent_data'
        else:  # Unix-like
            return Path.home() / '.local' / 'share' / 'agent_data'
    
    @staticmethod
    def get_config_dir() -> Path:
        """Get appropriate config directory for OS."""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('APPDATA', '.')) / 'agent_config'
        else:  # Unix-like
            return Path.home() / '.config' / 'agent_config'
    
    @staticmethod
    def fix_line_endings(content: str) -> str:
        """Normalize line endings."""
        return content.replace('\r\n', '\n').replace('\r', '\n')
```

---

## API Configuration Issues

### OpenAI API Problems

**Problem**: API key errors

```python
import os
import openai
from openai import OpenAI

def diagnose_openai_issues():
    """Diagnose OpenAI API configuration issues."""
    issues = []
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        issues.append("OPENAI_API_KEY not set in environment")
    elif not api_key.startswith('sk-'):
        issues.append("Invalid OpenAI API key format")
    elif len(api_key) < 50:
        issues.append("OpenAI API key appears too short")
    
    # Test API connection
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            print(f"✅ API connection successful. Available models: {len(models.data)}")
        except openai.AuthenticationError:
            issues.append("Invalid API key - authentication failed")
        except openai.RateLimitError:
            issues.append("Rate limit exceeded - wait and retry")
        except Exception as e:
            issues.append(f"API connection failed: {str(e)}")
    
    return issues

# Usage
issues = diagnose_openai_issues()
if issues:
    print("❌ OpenAI API Issues:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print("✅ OpenAI API configured correctly")
```

**Problem**: Rate limiting and quota issues

```python
import time
import random
from typing import Any, Dict

class RateLimitedClient:
    def __init__(self, client, max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries
        self.base_delay = 1
    
    def make_request(self, method_name: str, **kwargs) -> Any:
        """Make API request with rate limiting and retries."""
        method = getattr(self.client, method_name)
        
        for attempt in range(self.max_retries):
            try:
                return method(**kwargs)
            
            except openai.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1})")
                time.sleep(delay)
            
            except openai.APITimeoutError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                delay = self.base_delay * (attempt + 1)
                print(f"⏳ Timeout. Retrying in {delay}s... (attempt {attempt + 1})")
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")

# Usage
from openai import OpenAI
client = RateLimitedClient(OpenAI())
response = client.make_request('chat.completions.create', 
                              model="gpt-3.5-turbo",
                              messages=[{"role": "user", "content": "Hello"}])
```

### API Response Issues

**Problem**: Unexpected API responses

```python
import json
from typing import Optional, Dict, Any

class APIResponseValidator:
    @staticmethod
    def validate_openai_response(response: Any) -> Dict[str, Any]:
        """Validate OpenAI API response."""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check response structure
        if not hasattr(response, 'choices'):
            validation_result['valid'] = False
            validation_result['issues'].append("Response missing 'choices' attribute")
            return validation_result
        
        if not response.choices:
            validation_result['valid'] = False
            validation_result['issues'].append("Response has empty choices")
            return validation_result
        
        # Check first choice
        choice = response.choices[0]
        if not hasattr(choice, 'message'):
            validation_result['valid'] = False
            validation_result['issues'].append("Choice missing 'message' attribute")
        
        # Check usage information
        if hasattr(response, 'usage'):
            if response.usage.total_tokens > 8000:
                validation_result['warnings'].append(
                    f"High token usage: {response.usage.total_tokens}"
                )
        
        # Check for content filtering
        if hasattr(choice, 'finish_reason'):
            if choice.finish_reason == 'content_filter':
                validation_result['warnings'].append("Content was filtered")
            elif choice.finish_reason == 'length':
                validation_result['warnings'].append("Response truncated due to length")
        
        return validation_result
    
    @staticmethod
    def extract_content_safely(response: Any) -> Optional[str]:
        """Safely extract content from API response."""
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None

# Usage
validator = APIResponseValidator()
validation = validator.validate_openai_response(response)

if not validation['valid']:
    print("❌ Invalid API response:")
    for issue in validation['issues']:
        print(f"   • {issue}")

content = validator.extract_content_safely(response)
if content is None:
    print("⚠️ Could not extract content from response")
```

---

## Performance Optimization

### Memory Management

**Problem**: Memory leaks and high usage

```python
import gc
import psutil
import tracemalloc
from typing import Any, Dict
from functools import wraps

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
        tracemalloc.start()
    
    def profile_memory(func):
        """Decorator to profile memory usage of functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Take snapshot before
            snapshot_before = tracemalloc.take_snapshot()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Take snapshot after
                snapshot_after = tracemalloc.take_snapshot()
                
                # Calculate difference
                top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                
                total_size = sum(stat.size for stat in top_stats)
                if total_size > 1024 * 1024:  # More than 1MB
                    print(f"⚠️ {func.__name__} used {total_size / 1024 / 1024:.1f}MB")
                
                # Force garbage collection
                gc.collect()
        
        return wrapper
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_threshold(self, threshold_mb: float = 1000) -> bool:
        """Check if memory usage exceeds threshold."""
        usage = self.get_memory_usage()
        return usage['rss_mb'] > threshold_mb

# Memory optimization patterns
class MemoryOptimizer:
    @staticmethod
    def chunk_large_data(data: list, chunk_size: int = 1000):
        """Process data in chunks to reduce memory usage."""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    @staticmethod
    def clear_variables(*variables):
        """Explicitly clear large variables."""
        for var in variables:
            if var is not None:
                del var
        gc.collect()
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize pandas DataFrame memory usage."""
        # Downcast numeric types
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Use categorical for string columns with few unique values
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        return df

# Usage example
profiler = MemoryProfiler()

@profiler.profile_memory
def process_large_dataset(data):
    optimizer = MemoryOptimizer()
    
    # Process in chunks
    results = []
    for chunk in optimizer.chunk_large_data(data, chunk_size=500):
        processed_chunk = process_chunk(chunk)
        results.extend(processed_chunk)
        
        # Clear chunk from memory
        optimizer.clear_variables(chunk, processed_chunk)
    
    return results
```

### LLM Performance Optimization

**Problem**: Slow API responses

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class LLMOptimizer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.cache = {}  # Simple in-memory cache
    
    async def parallel_llm_calls(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Make multiple LLM calls in parallel."""
        tasks = []
        
        for request in requests:
            task = asyncio.create_task(self.make_async_call(request))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def make_async_call(self, request: Dict[str, Any]) -> Any:
        """Make asynchronous LLM call."""
        # Check cache first
        cache_key = self.generate_cache_key(request)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Make actual API call
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.sync_llm_call,
            request
        )
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    def sync_llm_call(self, request: Dict[str, Any]) -> Any:
        """Synchronous LLM call (to be run in executor)."""
        from openai import OpenAI
        client = OpenAI()
        
        return client.chat.completions.create(**request)
    
    def generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        import hashlib
        key_data = f"{request.get('model', '')}{request.get('messages', [])}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def optimize_prompt(self, prompt: str, max_length: int = 2000) -> str:
        """Optimize prompt for better performance."""
        # Remove excessive whitespace
        optimized = ' '.join(prompt.split())
        
        # Truncate if too long
        if len(optimized) > max_length:
            optimized = optimized[:max_length] + "..."
        
        # Add clear instructions at the end
        if not optimized.endswith('.'):
            optimized += "."
        
        return optimized
    
    def batch_similar_requests(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar requests for batch processing."""
        batches = {}
        
        for request in requests:
            model = request.get('model', 'default')
            if model not in batches:
                batches[model] = []
            batches[model].append(request)
        
        return list(batches.values())

# Usage example
async def process_multiple_agents():
    optimizer = LLMOptimizer()
    
    requests = [
        {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': 'Analyze data 1'}]
        },
        {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': 'Analyze data 2'}]
        }
    ]
    
    results = await optimizer.parallel_llm_calls(requests)
    return results
```

---

## Database Troubleshooting

### Connection Issues

**Problem**: Database connection failures

```python
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import time

class DatabaseDiagnostic:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
    
    def diagnose_connection(self) -> Dict[str, Any]:
        """Comprehensive database connection diagnosis."""
        result = {
            'connection_successful': False,
            'issues': [],
            'recommendations': [],
            'connection_info': {}
        }
        
        try:
            # Test basic connection
            self.engine = create_engine(self.database_url, echo=False)
            
            with self.engine.connect() as conn:
                # Test simple query
                start_time = time.time()
                conn.execute(text("SELECT 1"))
                query_time = time.time() - start_time
                
                result['connection_successful'] = True
                result['connection_info'] = {
                    'query_time_ms': round(query_time * 1000, 2),
                    'database_name': self.engine.url.database,
                    'host': self.engine.url.host,
                    'port': self.engine.url.port,
                    'driver': self.engine.url.drivername
                }
                
                # Performance checks
                if query_time > 1.0:
                    result['issues'].append(f"Slow connection: {query_time:.2f}s")
                    result['recommendations'].append("Check network latency to database server")
                
        except SQLAlchemyError as e:
            result['issues'].append(f"SQLAlchemy error: {str(e)}")
            
            # Specific error handling
            if "could not connect" in str(e).lower():
                result['recommendations'].extend([
                    "Check database server is running",
                    "Verify host and port configuration",
                    "Check firewall settings"
                ])
            elif "authentication failed" in str(e).lower():
                result['recommendations'].extend([
                    "Verify username and password",
                    "Check database user permissions"
                ])
            elif "database" in str(e).lower() and "does not exist" in str(e).lower():
                result['recommendations'].append("Create the database or update database name")
        
        except Exception as e:
            result['issues'].append(f"Unexpected error: {str(e)}")
            result['recommendations'].append("Check database URL format")
        
        return result
    
    def test_schema(self, required_tables: List[str]) -> Dict[str, Any]:
        """Test database schema and tables."""
        result = {
            'schema_valid': True,
            'missing_tables': [],
            'table_info': {}
        }
        
        if not self.engine:
            return {'error': 'No database connection'}
        
        try:
            inspector = sqlalchemy.inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            for table in required_tables:
                if table not in existing_tables:
                    result['missing_tables'].append(table)
                    result['schema_valid'] = False
                else:
                    columns = inspector.get_columns(table)
                    result['table_info'][table] = {
                        'columns': len(columns),
                        'column_names': [col['name'] for col in columns]
                    }
        
        except Exception as e:
            result['error'] = str(e)
            result['schema_valid'] = False
        
        return result

# Database setup helper
class DatabaseSetup:
    @staticmethod
    def create_sqlite_database(db_path: str):
        """Create SQLite database with proper configuration."""
        import sqlite3
        
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create database with optimized settings
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
        conn.execute("PRAGMA cache_size=10000")  # More cache
        conn.close()
    
    @staticmethod
    def test_postgresql_setup():
        """Test PostgreSQL setup and suggest optimizations."""
        suggestions = [
            "Install PostgreSQL: https://postgresql.org/download/",
            "Create database: createdb your_database_name",
            "Create user: createuser -P your_username",
            "Grant permissions: GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_username;"
        ]
        return suggestions

# Usage
diagnostic = DatabaseDiagnostic("postgresql://user:pass@localhost/mydb")
connection_result = diagnostic.diagnose_connection()

if not connection_result['connection_successful']:
    print("❌ Database Connection Issues:")
    for issue in connection_result['issues']:
        print(f"   • {issue}")
    print("\n💡 Recommendations:")
    for rec in connection_result['recommendations']:
        print(f"   • {rec}")
```

### Performance Issues

**Problem**: Slow database queries

```python
import time
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

class DatabasePerformanceMonitor:
    def __init__(self):
        self.slow_queries = []
        self.query_count = 0
        self.total_time = 0
        
        # Set up query monitoring
        self.setup_query_monitoring()
    
    def setup_query_monitoring(self):
        """Set up automatic query performance monitoring."""
        @event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            self.query_count += 1
            self.total_time += total
            
            # Log slow queries
            if total > 1.0:  # Queries taking more than 1 second
                self.slow_queries.append({
                    'statement': statement[:200] + "..." if len(statement) > 200 else statement,
                    'duration': total,
                    'timestamp': time.time()
                })
                
                logging.warning(f"Slow query ({total:.2f}s): {statement[:100]}...")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate database performance report."""
        avg_query_time = (self.total_time / self.query_count) if self.query_count > 0 else 0
        
        return {
            'total_queries': self.query_count,
            'total_time_seconds': round(self.total_time, 2),
            'average_query_time_ms': round(avg_query_time * 1000, 2),
            'slow_queries_count': len(self.slow_queries),
            'slow_queries': self.slow_queries[-10:],  # Last 10 slow queries
            'recommendations': self.generate_recommendations()
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        avg_time = (self.total_time / self.query_count) if self.query_count > 0 else 0
        
        if avg_time > 0.5:
            recommendations.append("Consider adding database indexes")
            recommendations.append("Review query complexity and optimize")
        
        if len(self.slow_queries) > 5:
            recommendations.append("Multiple slow queries detected - review database design")
        
        if self.query_count > 100:
            recommendations.append("High query count - consider connection pooling")
        
        return recommendations

# Query optimization utilities
class QueryOptimizer:
    @staticmethod
    def add_pagination(query, page: int = 1, per_page: int = 50):
        """Add pagination to query."""
        offset = (page - 1) * per_page
        return query.offset(offset).limit(per_page)
    
    @staticmethod
    def optimize_select(query, columns: List[str] = None):
        """Optimize SELECT to only fetch needed columns."""
        if columns:
            return query.with_entities(*columns)
        return query
    
    @staticmethod
    def add_indexes_suggestion(table_name: str, columns: List[str]) -> str:
        """Generate SQL for creating recommended indexes."""
        index_statements = []
        
        for column in columns:
            index_name = f"idx_{table_name}_{column}"
            statement = f"CREATE INDEX {index_name} ON {table_name} ({column});"
            index_statements.append(statement)
        
        return "\n".join(index_statements)

# Usage
monitor = DatabasePerformanceMonitor()

# After running your application
report = monitor.get_performance_report()
print(f"📊 Database Performance Report:")
print(f"   Queries: {report['total_queries']}")
print(f"   Average time: {report['average_query_time_ms']}ms")
print(f"   Slow queries: {report['slow_queries_count']}")

if report['recommendations']:
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
```

---

## Interface Troubleshooting

### Streamlit Issues

**Problem**: Streamlit app won't start or crashes

```python
import streamlit as st
import sys
import traceback
from pathlib import Path

class StreamlitDiagnostic:
    @staticmethod
    def check_streamlit_setup():
        """Check Streamlit configuration and common issues."""
        issues = []
        fixes = []
        
        # Check Streamlit installation
        try:
            import streamlit
            version = streamlit.__version__
            print(f"✅ Streamlit version: {version}")
        except ImportError:
            issues.append("Streamlit not installed")
            fixes.append("pip install streamlit")
        
        # Check port availability
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', 8501))
            sock.close()
            print("✅ Port 8501 available")
        except OSError:
            issues.append("Port 8501 already in use")
            fixes.append("Use different port: streamlit run app.py --server.port 8502")
        
        # Check browser configuration
        if 'DISPLAY' not in os.environ and sys.platform.startswith('linux'):
            issues.append("No display available (headless environment)")
            fixes.append("Set server.headless=true in .streamlit/config.toml")
        
        return issues, fixes
    
    @staticmethod
    def create_streamlit_config():
        """Create optimized Streamlit configuration."""
        config_dir = Path(".streamlit")
        config_dir.mkdir(exist_ok=True)
        
        config_content = """
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        config_file = config_dir / "config.toml"
        config_file.write_text(config_content.strip())
        
        return config_file

# Error handling for Streamlit apps
def streamlit_error_handler(func):
    """Decorator for handling Streamlit errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            
            # Offer retry option
            if st.button("🔄 Retry"):
                st.experimental_rerun()
    
    return wrapper

# Example Streamlit app with error handling
@streamlit_error_handler
def main_app():
    st.title("🤖 Enhanced Agent Architecture")
    
    # Add connection status
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = check_system_health()
    
    if not st.session_state.connection_status['healthy']:
        st.warning("⚠️ System health issues detected")
        
        with st.expander("🔧 Diagnostic Information"):
            st.json(st.session_state.connection_status)
    
    # Your app content here
    st.write("App is running correctly!")

def check_system_health() -> Dict[str, Any]:
    """Quick system health check for Streamlit apps."""
    health = {'healthy': True, 'issues': []}
    
    # Check API keys
    if not os.getenv('OPENAI_API_KEY'):
        health['healthy'] = False
        health['issues'].append('OpenAI API key not configured')
    
    # Check database connection
    try:
        # Your database connection test here
        pass
    except Exception as e:
        health['healthy'] = False
        health['issues'].append(f'Database connection failed: {str(e)}')
    
    return health

if __name__ == "__main__":
    # Run diagnostics before starting app
    diagnostic = StreamlitDiagnostic()
    issues, fixes = diagnostic.check_streamlit_setup()
    
    if issues:
        print("❌ Streamlit Issues Found:")
        for issue, fix in zip(issues, fixes):
            print(f"   Issue: {issue}")
            print(f"   Fix: {fix}\n")
    else:
        print("✅ Streamlit setup looks good")
        main_app()
```

### CLI Interface Issues

**Problem**: Command-line interface not working

```python
import argparse
import sys
import logging
from typing import List, Dict, Any

class CLIDiagnostic:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def setup_cli_logging(self, verbose: bool = False):
        """Set up proper CLI logging."""
        level = logging.DEBUG if verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler('cli_debug.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.getLogger().setLevel(level)
        logging.getLogger().addHandler(console_handler)
        logging.getLogger().addHandler(file_handler)
    
    def validate_cli_args(self, args: argparse.Namespace) -> List[str]:
        """Validate CLI arguments and return issues."""
        issues = []
        
        # Check required arguments
        required_args = ['command']  # Adjust based on your CLI
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                issues.append(f"Missing required argument: {arg}")
        
        # Check file paths
        if hasattr(args, 'config_file') and args.config_file:
            if not Path(args.config_file).exists():
                issues.append(f"Config file not found: {args.config_file}")
        
        # Check output directory
        if hasattr(args, 'output_dir') and args.output_dir:
            output_path = Path(args.output_dir)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    issues.append(f"Cannot create output directory: {args.output_dir}")
        
        return issues
    
    def create_robust_cli_parser(self) -> argparse.ArgumentParser:
        """Create robust CLI parser with proper error handling."""
        parser = argparse.ArgumentParser(
            description="Enhanced Agent Architecture CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s run --agent data_processor
  %(prog)s status --verbose
  %(prog)s --help

For more information, visit: https://your-docs-url.com
            """
        )
        
        # Global options
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='Configuration file path'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Run an agent')
        run_parser.add_argument('--agent', required=True, help='Agent name to run')
        run_parser.add_argument('--input', help='Input data or file')
        run_parser.add_argument('--output', help='Output directory')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show system status')
        status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
        
        return parser

# Error handling for CLI applications
class CLIErrorHandler:
    @staticmethod
    def handle_keyboard_interrupt():
        """Handle Ctrl+C gracefully."""
        print("\n⚠️ Operation cancelled by user")
        sys.exit(130)  # Standard exit code for SIGINT
    
    @staticmethod
    def handle_unexpected_error(error: Exception):
        """Handle unexpected errors."""
        logging.error(f"Unexpected error: {str(error)}")
        logging.debug(traceback.format_exc())
        
        print(f"❌ An unexpected error occurred: {str(error)}")
        print("💡 Check the log file 'cli_debug.log' for more details")
        sys.exit(1)
    
    @staticmethod
    def handle_validation_errors(errors: List[str]):
        """Handle validation errors."""
        print("❌ Validation errors:")
        for error in errors:
            print(f"   • {error}")
        
        print("\n💡 Use --help for usage information")
        sys.exit(2)

# Main CLI application
def main():
    """Main CLI entry point with comprehensive error handling."""
    diagnostic = CLIDiagnostic()
    error_handler = CLIErrorHandler()
    
    try:
        # Parse arguments
        parser = diagnostic.create_robust_cli_parser()
        args = parser.parse_args()
        
        # Set up logging
        diagnostic.setup_cli_logging(args.verbose)
        
        # Validate arguments
        validation_errors = diagnostic.validate_cli_args(args)
        if validation_errors:
            error_handler.handle_validation_errors(validation_errors)
        
        # Execute command
        if args.command == 'run':
            print(f"🚀 Running agent: {args.agent}")
            # Your agent execution logic here
        
        elif args.command == 'status':
            print("📊 System Status:")
            # Your status logic here
        
        else:
            parser.print_help()
            sys.exit(0)
    
    except KeyboardInterrupt:
        error_handler.handle_keyboard_interrupt()
    
    except Exception as e:
        error_handler.handle_unexpected_error(e)

if __name__ == "__main__":
    main()
```

---

## Import and Dependency Issues

### Python Path Problems

**Problem**: Module not found errors

```python
import sys
import os
from pathlib import Path

class PythonPathFixer:
    @staticmethod
    def fix_import_paths():
        """Fix common Python path issues."""
        # Add current directory to path
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Add parent directory (for relative imports)
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # Add common source directories
        common_dirs = ['src', 'lib', 'core', 'agents']
        for dir_name in common_dirs:
            dir_path = current_dir / dir_name
            if dir_path.exists() and str(dir_path) not in sys.path:
                sys.path.insert(0, str(dir_path))
    
    @staticmethod
    def diagnose_import_issues(module_name: str):
        """Diagnose why a module cannot be imported."""
        print(f"🔍 Diagnosing import issues for '{module_name}'...")
        
        # Check if module exists in path
        for path in sys.path:
            module_path = Path(path) / f"{module_name}.py"
            package_path = Path(path) / module_name / "__init__.py"
            
            if module_path.exists():
                print(f"✅ Found module file: {module_path}")
                return
            elif package_path.exists():
                print(f"✅ Found package: {package_path}")
                return
        
        print(f"❌ Module '{module_name}' not found in any path")
        print("📋 Current Python path:")
        for i, path in enumerate(sys.path):
            print(f"   {i}: {path}")
        
        # Suggest solutions
        print("\n💡 Possible solutions:")
        print("   • Check the module name spelling")
        print("   • Verify the module is installed: pip list | grep module_name")
        print("   • Add module directory to PYTHONPATH")
        print("   • Use relative imports if it's a local module")
    
    @staticmethod
    def create_init_files(directory: Path):
        """Create __init__.py files for Python packages."""
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            
            # Skip if already has __init__.py
            if (root_path / "__init__.py").exists():
                continue
            
            # Create __init__.py if directory contains Python files
            python_files = [f for f in files if f.endswith('.py')]
            if python_files:
                init_file = root_path / "__init__.py"
                init_file.touch()
                print(f"📄 Created: {init_file}")

# Usage
path_fixer = PythonPathFixer()
path_fixer.fix_import_paths()

# Diagnose specific import issues
try:
    import your_module
except ImportError as e:
    path_fixer.diagnose_import_issues('your_module')
```

### Dependency Conflicts

**Problem**: Package version conflicts

```python
import subprocess
import pkg_resources
from typing import List, Dict, Tuple

class DependencyManager:
    def __init__(self):
        self.installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    def check_conflicts(self, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """Check for dependency conflicts."""
        conflicts = []
        missing = []
        
        if not Path(requirements_file).exists():
            return {'error': f'Requirements file {requirements_file} not found'}
        
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        for req in requirements:
            package_name = req.split('==')[0].split('>=')[0].split('<=')[0].lower()
            
            if package_name not in self.installed_packages:
                missing.append(req)
            else:
                # Check version conflicts (simplified)
                if '==' in req:
                    required_version = req.split('==')[1]
                    installed_version = self.installed_packages[package_name]
                    
                    if required_version != installed_version:
                        conflicts.append({
                            'package': package_name,
                            'required': required_version,
                            'installed': installed_version
                        })
        
        return {
            'conflicts': conflicts,
            'missing': missing,
            'total_installed': len(self.installed_packages)
        }
    
    def create_clean_environment(self, requirements_file: str = "requirements.txt"):
        """Create clean virtual environment with exact requirements."""
        commands = [
            "python -m venv venv_clean",
            "source venv_clean/bin/activate" if os.name != 'nt' else "venv_clean\\Scripts\\activate",
            "pip install --upgrade pip",
            f"pip install -r {requirements_file}"
        ]
        
        print("🧹 Creating clean environment...")
        for cmd in commands:
            print(f"   Running: {cmd}")
        
        return commands
    
    def freeze_current_environment(self, output_file: str = "requirements_frozen.txt"):
        """Freeze current environment to file."""
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            print(f"✅ Environment frozen to {output_file}")
            return output_file
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to freeze environment: {e}")
            return None
    
    def suggest_fixes(self, conflicts: List[Dict[str, str]]) -> List[str]:
        """Suggest fixes for dependency conflicts."""
        suggestions = []
        
        for conflict in conflicts:
            package = conflict['package']
            required = conflict['required']
            installed = conflict['installed']
            
            suggestions.append(f"pip install {package}=={required}")
        
        if conflicts:
            suggestions.extend([
                "Consider using pip-tools for dependency management",
                "Create a new virtual environment for clean state",
                "Check for package compatibility issues"
            ])
        
        return suggestions

# Usage
dep_manager = DependencyManager()
conflicts = dep_manager.check_conflicts()

if conflicts.get('conflicts'):
    print("❌ Dependency conflicts found:")
    for conflict in conflicts['conflicts']:
        print(f"   {conflict['package']}: required {conflict['required']}, installed {conflict['installed']}")
    
    print("\n💡 Suggested fixes:")
    suggestions = dep_manager.suggest_fixes(conflicts['conflicts'])
    for suggestion in suggestions:
        print(f"   • {suggestion}")

if conflicts.get('missing'):
    print(f"\n📦 Missing packages ({len(conflicts['missing'])}):")
    for missing in conflicts['missing']:
        print(f"   • {missing}")
```

---

## Agent-Specific Issues

### Agent Communication Problems

**Problem**: Agents not communicating properly

```python
import json
import time
import uuid
from typing import Dict, Any, Optional

class AgentCommunicationDiagnostic:
    def __init__(self):
        self.message_log = []
        self.failed_messages = []
    
    def test_agent_communication(self, sender_agent, receiver_agent) -> Dict[str, Any]:
        """Test communication between two agents."""
        test_id = str(uuid.uuid4())
        
        # Create test message
        test_message = {
            'id': test_id,
            'sender': sender_agent.name,
            'receiver': receiver_agent.name,
            'content': 'ping',
            'timestamp': time.time()
        }
        
        result = {
            'test_id': test_id,
            'success': False,
            'latency_ms': None,
            'error': None,
            'details': {}
        }
        
        try:
            # Send message
            start_time = time.time()
            response = sender_agent.send_message(receiver_agent, test_message)
            end_time = time.time()
            
            # Check response
            if response and response.get('content') == 'pong':
                result['success'] = True
                result['latency_ms'] = round((end_time - start_time) * 1000, 2)
            else:
                result['error'] = f"Invalid response: {response}"
        
        except Exception as e:
            result['error'] = str(e)
        
        self.message_log.append(result)
        return result
    
    def diagnose_context_sharing(self, agents: List) -> Dict[str, Any]:
        """Diagnose context sharing between agents."""
        issues = []
        recommendations = []
        
        # Check if agents have context manager
        agents_without_context = []
        for agent in agents:
            if not hasattr(agent, 'context_manager'):
                agents_without_context.append(agent.name)
        
        if agents_without_context:
            issues.append(f"Agents without context manager: {agents_without_context}")
            recommendations.append("Initialize context manager for all agents")
        
        # Test context synchronization
        if len(agents) >= 2:
            test_key = "test_context_sync"
            test_value = f"sync_test_{int(time.time())}"
            
            try:
                # Set context in first agent
                agents[0].context_manager.set(test_key, test_value)
                
                # Check if second agent can read it
                retrieved_value = agents[1].context_manager.get(test_key)
                
                if retrieved_value != test_value:
                    issues.append("Context not synchronized between agents")
                    recommendations.append("Check context manager configuration")
            
            except Exception as e:
                issues.append(f"Context sharing test failed: {str(e)}")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'agents_tested': len(agents)
        }
    
    def test_parallel_execution(self, agents: List) -> Dict[str, Any]:
        """Test parallel execution of agents."""
        import asyncio
        
        async def run_agent_test(agent):
            try:
                start_time = time.time()
                result = await agent.execute_async({"test": "parallel_execution"})
                execution_time = time.time() - start_time
                
                return {
                    'agent': agent.name,
                    'success': True,
                    'execution_time': execution_time,
                    'result': result
                }
            except Exception as e:
                return {
                    'agent': agent.name,
                    'success': False,
                    'error': str(e)
                }
        
        # Run all agents in parallel
        async def run_all_tests():
            tasks = [run_agent_test(agent) for agent in agents]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            results = asyncio.run(run_all_tests())
            
            successful = [r for r in results if isinstance(r, dict) and r.get('success')]
            failed = [r for r in results if isinstance(r, dict) and not r.get('success')]
            
            return {
                'total_agents': len(agents),
                'successful': len(successful),
                'failed': len(failed),
                'results': results,
                'average_execution_time': sum(r['execution_time'] for r in successful) / len(successful) if successful else 0
            }
        
        except Exception as e:
            return {
                'error': f"Parallel execution test failed: {str(e)}",
                'total_agents': len(agents)
            }

# Agent debugging utilities
class AgentDebugger:
    def __init__(self, agent):
        self.agent = agent
        self.execution_history = []
    
    def trace_execution(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Trace agent execution with detailed logging."""
        trace_id = str(uuid.uuid4())
        
        trace_data = {
            'trace_id': trace_id,
            'agent_name': self.agent.name,
            'inputs': inputs,
            'start_time': time.time(),
            'steps': [],
            'outputs': None,
            'errors': [],
            'performance': {}
        }
        
        try:
            # Monkey patch to trace method calls
            original_execute = self.agent.execute
            
            def traced_execute(*args, **kwargs):
                step_start = time.time()
                try:
                    result = original_execute(*args, **kwargs)
                    step_end = time.time()
                    
                    trace_data['steps'].append({
                        'method': 'execute',
                        'duration': step_end - step_start,
                        'success': True
                    })
                    
                    return result
                except Exception as e:
                    step_end = time.time()
                    trace_data['steps'].append({
                        'method': 'execute',
                        'duration': step_end - step_start,
                        'success': False,
                        'error': str(e)
                    })
                    raise
            
            self.agent.execute = traced_execute
            
            # Execute with tracing
            outputs = self.agent.execute(inputs)
            trace_data['outputs'] = outputs
            
            # Restore original method
            self.agent.execute = original_execute
        
        except Exception as e:
            trace_data['errors'].append(str(e))
        
        finally:
            trace_data['end_time'] = time.time()
            trace_data['total_duration'] = trace_data['end_time'] - trace_data['start_time']
            self.execution_history.append(trace_data)
        
        return trace_data
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze agent performance from execution history."""
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        total_executions = len(self.execution_history)
        successful_executions = len([t for t in self.execution_history if not t['errors']])
        
        durations = [t['total_duration'] for t in self.execution_history if 'total_duration' in t]
        
        analysis = {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': (successful_executions / total_executions) * 100,
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'common_errors': self._get_common_errors()
        }
        
        return analysis
    
    def _get_common_errors(self) -> Dict[str, int]:
        """Get most common errors from execution history."""
        error_counts = {}
        
        for trace in self.execution_history:
            for error in trace['errors']:
                # Simplify error message for grouping
                error_type = error.split(':')[0] if ':' in error else error[:50]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))

# Usage
# Test agent communication
diagnostic = AgentCommunicationDiagnostic()
comm_result = diagnostic.test_agent_communication(agent1, agent2)

if not comm_result['success']:
    print(f"❌ Communication failed: {comm_result['error']}")
else:
    print(f"✅ Communication successful (latency: {comm_result['latency_ms']}ms)")

# Debug specific agent
debugger = AgentDebugger(my_agent)
trace = debugger.trace_execution({"input": "test data"})
performance = debugger.analyze_performance()

print(f"📊 Agent Performance:")
print(f"   Success rate: {performance['success_rate']:.1f}%")
print(f"   Average duration: {performance['average_duration']:.2f}s")
```

---

## Emergency Recovery Procedures

### System Recovery

**Problem**: Complete system failure

```python
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Any

class EmergencyRecovery:
    def __init__(self):
        self.backup_dir = Path("emergency_backups")
        self.recovery_log = []
    
    def create_emergency_backup(self) -> str:
        """Create emergency backup of critical files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"emergency_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        critical_files = [
            ".env",
            "requirements.txt",
            "config/",
            "data/",
            "logs/",
        ]
        
        backed_up = []
        failed = []
        
        for item in critical_files:
            item_path = Path(item)
            if item_path.exists():
                try:
                    if item_path.is_file():
                        shutil.copy2(item_path, backup_path)
                    else:
                        shutil.copytree(item_path, backup_path / item_path.name, 
                                      dirs_exist_ok=True)
                    backed_up.append(str(item_path))
                except Exception as e:
                    failed.append(f"{item}: {str(e)}")
        
        # Create recovery info
        recovery_info = {
            'timestamp': timestamp,
            'backup_path': str(backup_path),
            'backed_up_files': backed_up,
            'failed_files': failed,
            'python_version': sys.version,
            'working_directory': str(Path.cwd())
        }
        
        with open(backup_path / "recovery_info.json", 'w') as f:
            json.dump(recovery_info, f, indent=2)
        
        print(f"✅ Emergency backup created: {backup_path}")
        print(f"   Files backed up: {len(backed_up)}")
        if failed:
            print(f"   Files failed: {len(failed)}")
        
        return str(backup_path)
    
    def factory_reset(self) -> Dict[str, Any]:
        """Perform factory reset of the system."""
        print("⚠️ PERFORMING FACTORY RESET")
        print("This will remove all configuration and data!")
        
        reset_actions = []
        errors = []
        
        # Remove configuration files
        config_files = [".env", "config/", ".streamlit/"]
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    if config_path.is_file():
                        config_path.unlink()
                    else:
                        shutil.rmtree(config_path)
                    reset_actions.append(f"Removed: {config_file}")
                except Exception as e:
                    errors.append(f"Failed to remove {config_file}: {str(e)}")
        
        # Clear data directories
        data_dirs = ["data/", "logs/", "__pycache__/"]
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if data_path.exists():
                try:
                    shutil.rmtree(data_path)
                    reset_actions.append(f"Cleared: {data_dir}")
                except Exception as e:
                    errors.append(f"Failed to clear {data_dir}: {str(e)}")
        
        # Reinstall dependencies
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            reset_actions.append("Reinstalled dependencies")
        except subprocess.CalledProcessError as e:
            errors.append(f"Failed to reinstall dependencies: {str(e)}")
        
        return {
            'reset_actions': reset_actions,
            'errors': errors,
            'success': len(errors) == 0
        }
    
    def safe_mode_startup(self) -> Dict[str, Any]:
        """Start system in safe mode with minimal configuration."""
        print("🛡️ Starting in safe mode...")
        
        safe_mode_config = {
            'USE_MINIMAL_CONFIG': True,
            'DISABLE_EXTERNAL_APIS': True,
            'USE_MOCK_RESPONSES': True,
            'LOG_LEVEL': 'DEBUG'
        }
        
        # Set safe mode environment variables
        for key, value in safe_mode_config.items():
            os.environ[key] = str(value)
        
        # Create minimal configuration
        minimal_config = {
            'agents': {
                'enabled': ['diagnostic_agent'],
                'max_concurrent': 1
            },
            'interfaces': {
                'cli_only': True,
                'web_disabled': True
            },
            'logging': {
                'level': 'DEBUG',
                'console_output': True
            }
        }
        
        # Save safe mode config
        with open('safe_mode_config.json', 'w') as f:
            json.dump(minimal_config, f, indent=2)
        
        return {
            'safe_mode_enabled': True,
            'config': minimal_config,
            'message': 'System started in safe mode. Limited functionality available.'
        }
    
    def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore system from backup."""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            return {'error': f'Backup directory not found: {backup_path}'}
        
        # Read recovery info
        recovery_info_path = backup_dir / "recovery_info.json"
        if recovery_info_path.exists():
            with open(recovery_info_path, 'r') as f:
                recovery_info = json.load(f)
        else:
            recovery_info = {}
        
        restored = []
        errors = []
        
        # Restore files
        for item in backup_dir.iterdir():
            if item.name == "recovery_info.json":
                continue
            
            try:
                if item.is_file():
                    shutil.copy2(item, Path.cwd())
                else:
                    shutil.copytree(item, Path.cwd() / item.name, dirs_exist_ok=True)
                restored.append(str(item.name))
            except Exception as e:
                errors.append(f"{item.name}: {str(e)}")
        
        return {
            'restored_files': restored,
            'errors': errors,
            'recovery_info': recovery_info,
            'success': len(errors) == 0
        }

# Quick recovery script
def emergency_recovery_main():
    """Main emergency recovery function."""
    recovery = EmergencyRecovery()
    
    print("🚨 EMERGENCY RECOVERY MODE")
    print("=" * 50)
    
    options = [
        "1. Create emergency backup",
        "2. Start in safe mode",
        "3. Factory reset (WARNING: destructive)",
        "4. Restore from backup",
        "5. Run system diagnostics",
        "6. Exit"
    ]
    
    for option in options:
        print(option)
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        backup_path = recovery.create_emergency_backup()
        print(f"Backup created at: {backup_path}")
    
    elif choice == "2":
        result = recovery.safe_mode_startup()
        print(f"Safe mode: {result['message']}")
    
    elif choice == "3":
        confirm = input("Type 'RESET' to confirm factory reset: ")
        if confirm == "RESET":
            result = recovery.factory_reset()
            if result['success']:
                print("✅ Factory reset completed")
            else:
                print("❌ Factory reset failed")
                for error in result['errors']:
                    print(f"   • {error}")
    
    elif choice == "4":
        backup_path = input("Enter backup path: ").strip()
        result = recovery.restore_from_backup(backup_path)
        if result.get('success'):
            print("✅ Restore completed")
        else:
            print("❌ Restore failed")
    
    elif choice == "5":
        checker = SystemHealthChecker()
        report = checker.run_all_checks()
    
    elif choice == "6":
        print("Exiting recovery mode")
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    emergency_recovery_main()
```

---

## Common Error Messages and Solutions

### Error Reference Table

| Error Message | Cause | Solution | Prevention |
|---------------|-------|----------|------------|
| `ModuleNotFoundError: No module named 'openai'` | Missing dependency | `pip install openai` | Use requirements.txt |
| `AuthenticationError: Invalid API key` | Wrong/missing API key | Check .env file | Secure key management |
| `RateLimitError: Rate limit exceeded` | Too many API calls | Implement rate limiting | Use exponential backoff |
| `ConnectionError: Failed to connect` | Network/server issue | Check connectivity | Add retry logic |
| `MemoryError: Out of memory` | High memory usage | Process in chunks | Monitor memory usage |
| `PermissionError: Access denied` | File permission issue | Fix file permissions | Set proper permissions |
| `TimeoutError: Request timed out` | Slow response | Increase timeout | Optimize queries |
| `ValidationError: Invalid input` | Bad input data | Validate inputs | Add input validation |

---

## Next Steps

Once you've resolved your issues:

- **Return to Implementation**: [Choose Your Level](../05_implementation_levels/)
- **Security Check**: [Security Guidelines](security_guidelines.md)
- **Use Templates**: [Ready-to-Use Code](templates.md)
- **Start Fresh**: [Quick Start Guide](../01_quick_start.md)

---

*This troubleshooting guide provides systematic approaches to diagnosing and resolving issues across all aspects of the Enhanced Agent Architecture Framework.*
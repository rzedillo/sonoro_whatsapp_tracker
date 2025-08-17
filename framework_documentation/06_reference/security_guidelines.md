# Security Guidelines and Implementation Patterns

> 🔒 **Security First**: Comprehensive security practices and patterns for building secure agent systems across all implementation levels.

## Navigation
- **Previous**: [Decision Matrices](decision_matrices.md)
- **Next**: [Troubleshooting Guide](troubleshooting.md)
- **Related**: [Architecture Patterns](../02_architecture_patterns.md) → [Implementation Levels](../05_implementation_levels/)

---

## Security by Implementation Level

### 🔒 Foundation Security (ALL LEVELS) - MANDATORY

**Environment File Security**: Before any development, secure configuration management is CRITICAL.

#### Environment File Requirements

**✅ DO:**
```bash
# 1. Always use .env files for configuration
# Create .env with secure permissions
cat > .env << 'EOF'
OPENAI_API_KEY=your_key_here
SECRET_KEY=your_32_char_secret_here
EOF

# 2. Set restrictive permissions (owner read/write only)
chmod 600 .env

# 3. Verify permissions
ls -la .env
# Should show: -rw------- (600 permissions)

# 4. Always add to .gitignore
echo ".env" >> .gitignore
echo "*.env" >> .gitignore
echo "*secret*" >> .gitignore
echo "*key*" >> .gitignore
```

**❌ NEVER:**
```bash
# ❌ Never hardcode credentials in source code
OPENAI_API_KEY = "sk-1234567890abcdef"  # ❌ NEVER DO THIS

# ❌ Never commit .env files
git add .env  # ❌ NEVER DO THIS

# ❌ Never use weak permissions
chmod 644 .env  # ❌ Too permissive - others can read
chmod 777 .env  # ❌ Extremely dangerous
```

#### Environment Variable Validation

```python
import os
from pathlib import Path

class SecureEnvironmentManager:
    """Secure environment management for all agent systems"""
    
    def __init__(self):
        self.required_vars = ["OPENAI_API_KEY"]
        self.optional_vars = ["ANTHROPIC_API_KEY", "SECRET_KEY"]
        self.env_file = Path(".env")
        
    def validate_environment(self) -> bool:
        """Validate environment setup and security"""
        # Check .env file exists and has secure permissions
        if not self.env_file.exists():
            raise SecurityError("❌ .env file not found. Run setup first.")
        
        # Check file permissions (must be 600)
        file_stat = self.env_file.stat()
        if oct(file_stat.st_mode)[-3:] != '600':
            raise SecurityError(
                f"❌ .env file has insecure permissions: {oct(file_stat.st_mode)[-3:]}. "
                "Run: chmod 600 .env"
            )
        
        # Validate required environment variables
        missing_vars = []
        for var in self.required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise SecurityError(
                f"❌ Missing required environment variables: {missing_vars}"
            )
        
        # Validate API key format (basic check)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key.startswith("sk-") or len(openai_key) < 20:
            raise SecurityError("❌ Invalid OpenAI API key format")
        
        return True
    
    def secure_get_env(self, key: str, default: str = None) -> str:
        """Securely retrieve environment variable"""
        value = os.getenv(key, default)
        if not value:
            raise SecurityError(f"❌ Environment variable {key} not set")
        
        # Log access (without revealing value)
        print(f"✅ Environment variable {key} accessed securely")
        return value

# Usage in all agent systems
env_manager = SecureEnvironmentManager()
env_manager.validate_environment()  # Run this before any agent operations
```

#### Git Security Verification

```bash
# Always verify .env is not tracked by git
git_check_env_security() {
    echo "🔍 Checking git security..."
    
    # Check if .env is in .gitignore
    if ! grep -q "\.env" .gitignore 2>/dev/null; then
        echo "❌ ERROR: .env not in .gitignore"
        echo "Fix: echo '.env' >> .gitignore"
        return 1
    fi
    
    # Check if .env is being tracked
    if git ls-files --error-unmatch .env >/dev/null 2>&1; then
        echo "❌ CRITICAL: .env is being tracked by git!"
        echo "Fix: git rm --cached .env && git commit -m 'Remove .env from tracking'"
        return 1
    fi
    
    # Check for any committed secrets (basic scan)
    if git log --all --full-history -- "*.env" | head -1; then
        echo "⚠️  WARNING: .env files found in git history"
        echo "Consider using git-filter-branch to remove sensitive data"
    fi
    
    echo "✅ Git security check passed"
    return 0
}

# Run security check
git_check_env_security
```

**This foundation security is MANDATORY for ALL implementation levels and must be completed before any agent development.**

---

### Level 1: Basic Security (Development/Prototype)

**Essential Security Practices:**

```python
# Input Validation and Sanitization
from typing import Any, Dict
import re
import html

class BasicSecurityAgent(BaseAgent):
    def validate_input(self, user_input: str) -> str:
        """Basic input validation and sanitization."""
        # Remove potential script injections
        sanitized = html.escape(user_input)
        
        # Length limits
        if len(sanitized) > 10000:
            raise ValueError("Input too long")
        
        # Basic pattern validation
        if re.search(r'<script|javascript:|data:|vbscript:', sanitized, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected")
        
        return sanitized
    
    def secure_file_operations(self, filepath: str) -> bool:
        """Secure file operations with path validation."""
        # Prevent directory traversal
        if '..' in filepath or filepath.startswith('/'):
            raise ValueError("Invalid file path")
        
        # Whitelist allowed extensions
        allowed_extensions = {'.txt', '.json', '.csv', '.pdf'}
        if not any(filepath.endswith(ext) for ext in allowed_extensions):
            raise ValueError("File type not allowed")
        
        return True
```

**Environment Security:**

```python
# Secure Configuration Management
import os
from pathlib import Path

class SecureConfig:
    def __init__(self):
        self.config_file = Path('.env')
        self.ensure_secure_permissions()
    
    def ensure_secure_permissions(self):
        """Set secure file permissions for config files."""
        if self.config_file.exists():
            os.chmod(self.config_file, 0o600)  # Owner read/write only
    
    def get_secret(self, key: str) -> str:
        """Securely retrieve secrets from environment."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required secret {key} not found")
        return value
    
    def validate_api_key(self, api_key: str) -> bool:
        """Basic API key validation."""
        if not api_key or len(api_key) < 20:
            return False
        if api_key.startswith('sk-') and len(api_key) < 50:
            return False
        return True
```

### Level 2: Standard Security (Business Applications)

**Authentication and Authorization:**

```python
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps

class AuthenticationSystem:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=24)
    
    def hash_password(self, password: str) -> str:
        """Securely hash passwords."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_token(self, user_id: str, roles: list = None) -> str:
        """Generate JWT token with expiration."""
        payload = {
            'user_id': user_id,
            'roles': roles or [],
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

def require_auth(roles: list = None):
    """Decorator for requiring authentication."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract token from request
            token = kwargs.get('token') or args[0].get_header('Authorization')
            if not token:
                raise ValueError("Authentication required")
            
            # Verify token
            auth_system = AuthenticationSystem(os.getenv('SECRET_KEY'))
            payload = auth_system.verify_token(token)
            
            # Check roles
            if roles and not any(role in payload.get('roles', []) for role in roles):
                raise ValueError("Insufficient permissions")
            
            kwargs['user_info'] = payload
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

**Data Encryption:**

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: str = None):
        if password:
            self.key = self._derive_key(password.encode())
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: bytes) -> bytes:
        """Derive encryption key from password."""
        salt = os.getenv('ENCRYPTION_SALT', 'default_salt').encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()

class SecureDataHandler:
    def __init__(self):
        self.encryption = DataEncryption(os.getenv('ENCRYPTION_PASSWORD'))
    
    def store_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store data with sensitive fields encrypted."""
        sensitive_fields = {'password', 'api_key', 'token', 'secret'}
        
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                data[key] = self.encryption.encrypt_data(str(value))
        
        return data
    
    def retrieve_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data with sensitive fields decrypted."""
        sensitive_fields = {'password', 'api_key', 'token', 'secret'}
        
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                data[key] = self.encryption.decrypt_data(value)
        
        return data
```

### Level 3: Advanced Security (Enterprise)

**Role-Based Access Control (RBAC):**

```python
from enum import Enum
from typing import Set, List, Optional
from dataclasses import dataclass

class Permission(Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    ADMIN_ACCESS = "admin_access"
    EXECUTE_AGENT = "execute_agent"
    VIEW_LOGS = "view_logs"
    MANAGE_USERS = "manage_users"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    description: str = ""

@dataclass
class User:
    user_id: str
    username: str
    email: str
    roles: Set[str]
    active: bool = True

class RBACSystem:
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default role hierarchy."""
        self.roles = {
            'viewer': Role(
                'viewer',
                {Permission.READ_DATA, Permission.VIEW_LOGS},
                'Read-only access to data and logs'
            ),
            'operator': Role(
                'operator',
                {Permission.READ_DATA, Permission.EXECUTE_AGENT, Permission.VIEW_LOGS},
                'Can execute agents and view results'
            ),
            'admin': Role(
                'admin',
                {Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA,
                 Permission.EXECUTE_AGENT, Permission.VIEW_LOGS, Permission.MANAGE_USERS},
                'Full administrative access'
            ),
            'super_admin': Role(
                'super_admin',
                set(Permission),
                'Complete system access'
            )
        }
    
    def check_permission(self, user_id: str, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        user = self.users.get(user_id)
        if not user or not user.active:
            return False
        
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and required_permission in role.permissions:
                return True
        
        return False
    
    def require_permission(self, permission: Permission):
        """Decorator for requiring specific permissions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                user_id = kwargs.get('user_id') or getattr(args[0], 'current_user_id', None)
                if not user_id or not self.check_permission(user_id, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

**Audit Logging and Monitoring:**

```python
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, error
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

class SecurityAuditLogger:
    def __init__(self, log_file: str = "security_audit.log"):
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_event(self, event: AuditEvent):
        """Log security audit event."""
        log_data = asdict(event)
        log_data['timestamp'] = event.timestamp.isoformat()
        
        self.logger.info(json.dumps(log_data))
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempts."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action="authentication",
            resource="login",
            result="success" if success else "failure",
            details={"method": "jwt_token"},
            ip_address=ip_address
        )
        self.log_event(event)
    
    def log_data_access(self, user_id: str, resource: str, action: str, success: bool):
        """Log data access events."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            result="success" if success else "failure",
            details={"access_type": "data_operation"}
        )
        self.log_event(event)
    
    def log_agent_execution(self, user_id: str, agent_name: str, execution_id: str, success: bool):
        """Log agent execution events."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action="execute_agent",
            resource=agent_name,
            result="success" if success else "failure",
            details={"execution_id": execution_id}
        )
        self.log_event(event)

class SecurityMonitor:
    def __init__(self, audit_logger: SecurityAuditLogger):
        self.audit_logger = audit_logger
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def check_rate_limiting(self, user_id: str, ip_address: str) -> bool:
        """Check for suspicious activity patterns."""
        now = datetime.utcnow()
        
        # Clean old attempts
        if user_id in self.failed_attempts:
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if now - attempt < self.lockout_duration
            ]
        
        # Check if user is locked out
        if user_id in self.failed_attempts:
            if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
                self.audit_logger.log_event(AuditEvent(
                    timestamp=now,
                    user_id=user_id,
                    action="rate_limit_triggered",
                    resource="authentication",
                    result="blocked",
                    details={"reason": "too_many_failed_attempts"},
                    ip_address=ip_address
                ))
                return False
        
        return True
    
    def record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(datetime.utcnow())
```

### Level 4: Production Security (Mission-Critical)

**Zero Trust Architecture:**

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import hmac
import hashlib

class ZeroTrustValidator:
    def __init__(self):
        self.device_registry = {}
        self.session_store = {}
        self.risk_threshold = 0.7
    
    def register_device(self, device_id: str, public_key: bytes) -> bool:
        """Register trusted device."""
        self.device_registry[device_id] = {
            'public_key': public_key,
            'registered_at': datetime.utcnow(),
            'trusted': True
        }
        return True
    
    def validate_device(self, device_id: str, signature: bytes, message: str) -> bool:
        """Validate device using cryptographic signature."""
        if device_id not in self.device_registry:
            return False
        
        device_info = self.device_registry[device_id]
        public_key = serialization.load_pem_public_key(device_info['public_key'])
        
        try:
            public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def calculate_risk_score(self, user_id: str, context: Dict[str, Any]) -> float:
        """Calculate risk score for access request."""
        risk_factors = {
            'unusual_location': 0.3,
            'new_device': 0.4,
            'unusual_time': 0.2,
            'high_privilege_request': 0.5,
            'multiple_failed_attempts': 0.6
        }
        
        total_risk = 0.0
        for factor, weight in risk_factors.items():
            if context.get(factor, False):
                total_risk += weight
        
        return min(total_risk, 1.0)
    
    def evaluate_access_request(self, user_id: str, resource: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate access request using zero trust principles."""
        risk_score = self.calculate_risk_score(user_id, context)
        
        decision = {
            'allowed': risk_score < self.risk_threshold,
            'risk_score': risk_score,
            'requires_mfa': risk_score > 0.3,
            'requires_approval': risk_score > 0.5,
            'context': context
        }
        
        return decision
```

**Data Loss Prevention (DLP):**

```python
import re
from typing import List, Tuple

class DataLossPreventionScanner:
    def __init__(self):
        self.patterns = {
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b'
        }
        
        self.sensitivity_levels = {
            'credit_card': 'HIGH',
            'ssn': 'HIGH',
            'api_key': 'HIGH',
            'email': 'MEDIUM',
            'phone': 'MEDIUM'
        }
    
    def scan_content(self, content: str) -> List[Tuple[str, str, str]]:
        """Scan content for sensitive data patterns."""
        findings = []
        
        for data_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append((
                    data_type,
                    self.sensitivity_levels[data_type],
                    match.group()
                ))
        
        return findings
    
    def sanitize_content(self, content: str) -> str:
        """Remove or mask sensitive data from content."""
        sanitized = content
        
        for data_type, pattern in self.patterns.items():
            if self.sensitivity_levels[data_type] == 'HIGH':
                sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
            else:
                # Partial masking for medium sensitivity
                sanitized = re.sub(
                    pattern,
                    lambda m: self._partial_mask(m.group()),
                    sanitized,
                    flags=re.IGNORECASE
                )
        
        return sanitized
    
    def _partial_mask(self, text: str) -> str:
        """Apply partial masking to sensitive data."""
        if len(text) <= 4:
            return '*' * len(text)
        return text[:2] + '*' * (len(text) - 4) + text[-2:]
    
    def validate_output(self, content: str, allow_sensitive: bool = False) -> bool:
        """Validate output content for sensitive data."""
        findings = self.scan_content(content)
        
        if not allow_sensitive and findings:
            high_severity = [f for f in findings if f[1] == 'HIGH']
            if high_severity:
                return False
        
        return True
```

**Compliance and Governance:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

@dataclass
class ComplianceRequirement:
    framework: ComplianceFramework
    requirement_id: str
    description: str
    implementation_status: str
    evidence: List[str]
    last_assessed: Optional[datetime] = None

class ComplianceManager:
    def __init__(self):
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.audit_logger = SecurityAuditLogger("compliance_audit.log")
        self._initialize_requirements()
    
    def _initialize_requirements(self):
        """Initialize compliance requirements based on frameworks."""
        gdpr_requirements = [
            ComplianceRequirement(
                ComplianceFramework.GDPR,
                "GDPR-7.1",
                "Right to erasure (right to be forgotten)",
                "implemented",
                ["user_deletion_endpoint", "data_purge_process"]
            ),
            ComplianceRequirement(
                ComplianceFramework.GDPR,
                "GDPR-32",
                "Security of processing",
                "implemented",
                ["encryption_at_rest", "access_controls", "audit_logs"]
            )
        ]
        
        for req in gdpr_requirements:
            self.requirements[req.requirement_id] = req
    
    def assess_compliance(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Assess compliance status for a framework."""
        framework_reqs = [
            req for req in self.requirements.values()
            if req.framework == framework
        ]
        
        total_reqs = len(framework_reqs)
        implemented = len([req for req in framework_reqs if req.implementation_status == "implemented"])
        
        compliance_score = (implemented / total_reqs * 100) if total_reqs > 0 else 0
        
        assessment = {
            'framework': framework.value,
            'total_requirements': total_reqs,
            'implemented': implemented,
            'compliance_score': compliance_score,
            'status': 'compliant' if compliance_score >= 90 else 'non_compliant',
            'assessment_date': datetime.utcnow()
        }
        
        self.audit_logger.log_event(AuditEvent(
            timestamp=datetime.utcnow(),
            user_id="system",
            action="compliance_assessment",
            resource=framework.value,
            result="success",
            details=assessment
        ))
        
        return assessment
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate detailed compliance report."""
        assessment = self.assess_compliance(framework)
        
        framework_reqs = [
            req for req in self.requirements.values()
            if req.framework == framework
        ]
        
        report = {
            'summary': assessment,
            'requirements': [
                {
                    'id': req.requirement_id,
                    'description': req.description,
                    'status': req.implementation_status,
                    'evidence': req.evidence,
                    'last_assessed': req.last_assessed.isoformat() if req.last_assessed else None
                }
                for req in framework_reqs
            ],
            'recommendations': self._generate_recommendations(framework_reqs)
        }
        
        return report
    
    def _generate_recommendations(self, requirements: List[ComplianceRequirement]) -> List[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []
        
        not_implemented = [req for req in requirements if req.implementation_status != "implemented"]
        
        for req in not_implemented:
            recommendations.append(
                f"Implement {req.requirement_id}: {req.description}"
            )
        
        return recommendations
```

---

## Security Architecture Patterns

### Defense in Depth

```python
class DefenseInDepthArchitecture:
    def __init__(self):
        self.layers = {
            'perimeter': PerimeterSecurity(),
            'network': NetworkSecurity(),
            'application': ApplicationSecurity(),
            'data': DataSecurity(),
            'endpoint': EndpointSecurity()
        }
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate request through all security layers."""
        for layer_name, layer in self.layers.items():
            try:
                if not layer.validate(request):
                    self._log_security_violation(layer_name, request)
                    return False
            except Exception as e:
                self._log_security_error(layer_name, str(e))
                return False
        
        return True
    
    def _log_security_violation(self, layer: str, request: Dict[str, Any]):
        """Log security violations."""
        audit_logger = SecurityAuditLogger()
        audit_logger.log_event(AuditEvent(
            timestamp=datetime.utcnow(),
            user_id=request.get('user_id', 'unknown'),
            action="security_violation",
            resource=layer,
            result="blocked",
            details={'request_summary': str(request)[:200]}
        ))
```

### Secure Agent Communication

```python
import asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

class SecureAgentCommunication:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        self.trusted_agents = {}
        self.message_signatures = {}
    
    def register_agent(self, agent_id: str, public_key: bytes):
        """Register trusted agent."""
        self.trusted_agents[agent_id] = public_key
    
    def sign_message(self, message: str) -> bytes:
        """Sign message with private key."""
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_agent_message(self, agent_id: str, message: str, signature: bytes) -> bool:
        """Verify message from another agent."""
        if agent_id not in self.trusted_agents:
            return False
        
        try:
            public_key = serialization.load_pem_public_key(self.trusted_agents[agent_id])
            public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    async def secure_agent_call(self, agent_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make secure call to another agent."""
        # Sign the request
        message = json.dumps(request, sort_keys=True)
        signature = self.sign_message(message)
        
        # Add authentication headers
        secure_request = {
            'payload': request,
            'signature': base64.b64encode(signature).decode(),
            'sender_id': self.agent_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Make the call (implementation depends on communication method)
        response = await self._make_agent_call(agent_id, secure_request)
        
        # Verify response signature
        if not self.verify_agent_message(agent_id, json.dumps(response['payload'], sort_keys=True), 
                                       base64.b64decode(response['signature'])):
            raise SecurityError("Invalid response signature")
        
        return response['payload']
```

---

## Security Testing Framework

### Security Test Suite

```python
import pytest
from unittest.mock import MagicMock, patch

class SecurityTestSuite:
    def __init__(self):
        self.test_cases = []
        self.vulnerability_scanner = VulnerabilityScanner()
    
    def test_input_validation(self):
        """Test input validation against common attacks."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert('xss')",
            "\x00\x01\x02\x03",  # Null bytes
            "A" * 10000,  # Buffer overflow
        ]
        
        agent = BaseAgent("test")
        
        for malicious_input in malicious_inputs:
            with pytest.raises((ValueError, SecurityError)):
                agent.validate_input(malicious_input)
    
    def test_authentication_bypass(self):
        """Test authentication bypass attempts."""
        auth_system = AuthenticationSystem("test_secret")
        
        # Test invalid tokens
        invalid_tokens = [
            "invalid_token",
            "",
            None,
            "Bearer fake_token",
            {"fake": "token"}
        ]
        
        for token in invalid_tokens:
            with pytest.raises((ValueError, jwt.InvalidTokenError)):
                auth_system.verify_token(token)
    
    def test_authorization_escalation(self):
        """Test privilege escalation attempts."""
        rbac = RBACSystem()
        
        # Create low-privilege user
        rbac.users["test_user"] = User("test_user", "test", "test@example.com", {"viewer"})
        
        # Try to access admin-only functions
        assert not rbac.check_permission("test_user", Permission.DELETE_DATA)
        assert not rbac.check_permission("test_user", Permission.MANAGE_USERS)
    
    def test_data_encryption(self):
        """Test data encryption/decryption."""
        encryption = DataEncryption("test_password")
        
        sensitive_data = "secret_api_key_12345"
        encrypted = encryption.encrypt_data(sensitive_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert encrypted != sensitive_data
        assert decrypted == sensitive_data
    
    def test_audit_logging(self):
        """Test audit logging completeness."""
        audit_logger = SecurityAuditLogger("test_audit.log")
        
        # Test various security events
        audit_logger.log_authentication("test_user", True, "192.168.1.1")
        audit_logger.log_data_access("test_user", "sensitive_data", "read", True)
        audit_logger.log_agent_execution("test_user", "test_agent", "exec_123", False)
        
        # Verify logs are written
        with open("test_audit.log", "r") as f:
            logs = f.readlines()
            assert len(logs) == 3
    
    def test_rate_limiting(self):
        """Test rate limiting mechanisms."""
        monitor = SecurityMonitor(SecurityAuditLogger())
        
        # Simulate multiple failed attempts
        for _ in range(6):
            monitor.record_failed_attempt("test_user")
        
        # Should be rate limited
        assert not monitor.check_rate_limiting("test_user", "192.168.1.1")
    
    def test_data_loss_prevention(self):
        """Test DLP scanning and sanitization."""
        dlp = DataLossPreventionScanner()
        
        test_content = "Credit card: 4532-1234-5678-9012, SSN: 123-45-6789"
        findings = dlp.scan_content(test_content)
        
        assert len(findings) == 2
        assert any(finding[0] == 'credit_card' for finding in findings)
        assert any(finding[0] == 'ssn' for finding in findings)
        
        sanitized = dlp.sanitize_content(test_content)
        assert "4532-1234-5678-9012" not in sanitized
        assert "123-45-6789" not in sanitized

@pytest.fixture
def security_test_environment():
    """Set up secure test environment."""
    # Use test-specific secrets
    os.environ['SECRET_KEY'] = 'test_secret_key_12345'
    os.environ['ENCRYPTION_PASSWORD'] = 'test_encryption_pass'
    
    yield
    
    # Clean up
    if 'SECRET_KEY' in os.environ:
        del os.environ['SECRET_KEY']
    if 'ENCRYPTION_PASSWORD' in os.environ:
        del os.environ['ENCRYPTION_PASSWORD']

def run_security_tests():
    """Run comprehensive security test suite."""
    test_suite = SecurityTestSuite()
    
    # Run all security tests
    pytest.main([
        "-v",
        "--tb=short",
        "test_security.py"
    ])
```

---

## Security Configuration Templates

### Environment-Specific Security

**Development Security Configuration:**

```bash
# .env.development
SECURITY_LEVEL=development
ENABLE_DEBUG=true
ENABLE_CORS=true
SESSION_TIMEOUT=24h
PASSWORD_MIN_LENGTH=8
ENABLE_RATE_LIMITING=false
AUDIT_LOG_LEVEL=INFO
```

**Production Security Configuration:**

```bash
# .env.production
SECURITY_LEVEL=production
ENABLE_DEBUG=false
ENABLE_CORS=false
SESSION_TIMEOUT=1h
PASSWORD_MIN_LENGTH=12
ENABLE_RATE_LIMITING=true
AUDIT_LOG_LEVEL=WARNING
FORCE_HTTPS=true
ENABLE_HSTS=true
CONTENT_SECURITY_POLICY=strict
```

### Security Headers Configuration

```python
class SecurityHeaders:
    @staticmethod
    def get_security_headers(environment: str = "production") -> Dict[str, str]:
        """Get security headers based on environment."""
        base_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        if environment == "production":
            base_headers.update({
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
            })
        
        return base_headers
```

---

## Incident Response Framework

### Security Incident Handling

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityIncident:
    incident_id: str
    severity: IncidentSeverity
    description: str
    detected_at: datetime
    affected_systems: List[str]
    status: str = "open"
    assigned_to: str = ""

class IncidentResponseSystem:
    def __init__(self):
        self.incidents: Dict[str, SecurityIncident] = {}
        self.response_procedures = self._initialize_procedures()
        self.notification_system = NotificationSystem()
    
    def _initialize_procedures(self) -> Dict[IncidentSeverity, List[str]]:
        """Initialize incident response procedures."""
        return {
            IncidentSeverity.CRITICAL: [
                "Immediately isolate affected systems",
                "Notify security team and management",
                "Activate incident response team",
                "Document all actions",
                "Preserve evidence",
                "Implement containment measures"
            ],
            IncidentSeverity.HIGH: [
                "Assess and contain the incident",
                "Notify security team",
                "Document incident details",
                "Implement mitigation measures"
            ],
            IncidentSeverity.MEDIUM: [
                "Investigate and document",
                "Implement preventive measures",
                "Update security policies if needed"
            ],
            IncidentSeverity.LOW: [
                "Log incident for tracking",
                "Schedule regular review"
            ]
        }
    
    def report_incident(self, description: str, severity: IncidentSeverity, 
                       affected_systems: List[str]) -> str:
        """Report a new security incident."""
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            severity=severity,
            description=description,
            detected_at=datetime.utcnow(),
            affected_systems=affected_systems
        )
        
        self.incidents[incident_id] = incident
        
        # Trigger response procedures
        self._execute_response_procedures(incident)
        
        # Send notifications
        self.notification_system.notify_incident(incident)
        
        return incident_id
    
    def _execute_response_procedures(self, incident: SecurityIncident):
        """Execute response procedures based on severity."""
        procedures = self.response_procedures.get(incident.severity, [])
        
        for procedure in procedures:
            # Log the execution of each procedure
            audit_logger = SecurityAuditLogger()
            audit_logger.log_event(AuditEvent(
                timestamp=datetime.utcnow(),
                user_id="incident_response_system",
                action="execute_procedure",
                resource=incident.incident_id,
                result="success",
                details={"procedure": procedure, "severity": incident.severity.value}
            ))
```

---

## Security Metrics and KPIs

### Security Monitoring Dashboard

```python
class SecurityMetrics:
    def __init__(self):
        self.metrics_store = {}
        self.alert_thresholds = {
            'failed_logins_per_hour': 10,
            'data_access_anomalies': 5,
            'privilege_escalation_attempts': 1,
            'unusual_agent_executions': 3
        }
    
    def calculate_security_score(self) -> Dict[str, Any]:
        """Calculate overall security score."""
        metrics = {
            'authentication_success_rate': self._get_auth_success_rate(),
            'incident_response_time': self._get_avg_incident_response_time(),
            'compliance_score': self._get_compliance_score(),
            'vulnerability_count': self._get_open_vulnerabilities(),
            'patch_level': self._get_patch_level()
        }
        
        # Calculate weighted security score
        weights = {
            'authentication_success_rate': 0.2,
            'incident_response_time': 0.3,
            'compliance_score': 0.2,
            'vulnerability_count': 0.2,
            'patch_level': 0.1
        }
        
        security_score = sum(metrics[key] * weights[key] for key in weights)
        
        return {
            'overall_score': security_score,
            'metrics': metrics,
            'grade': self._calculate_grade(security_score)
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate security grade based on score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "F"
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        security_score = self.calculate_security_score()
        
        report = {
            'report_date': datetime.utcnow().isoformat(),
            'security_score': security_score,
            'alerts': self._get_active_alerts(),
            'incidents': self._get_recent_incidents(),
            'recommendations': self._generate_security_recommendations(),
            'trends': self._calculate_security_trends()
        }
        
        return report
```

---

## Next Steps

- **Troubleshooting Guide**: [Common Security Issues and Solutions](troubleshooting.md)
- **Implementation Templates**: [Ready-to-Use Security Code](templates.md)
- **Decision Matrices**: [Security Pattern Selection](decision_matrices.md)

---

*These security guidelines provide comprehensive protection patterns that scale from development prototypes to production-grade systems, ensuring your agent architecture remains secure at every level.*
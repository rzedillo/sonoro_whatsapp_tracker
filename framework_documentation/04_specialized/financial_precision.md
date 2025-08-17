# Financial & Precision Data Handling

> 💰 **Enterprise-Grade Precision**: Handle financial data, scientific measurements, and precision-critical calculations with accuracy guarantees and regulatory compliance.

## Navigation
- **Previous**: [Progress Tracking](../03_interfaces/progress_tracking.md)
- **Next**: [Testing Frameworks](testing_frameworks.md) → [Context Management](context_management.md)
- **Implementation**: [Level 3: Complex](../05_implementation_levels/level_3_complex.md) → [Level 4: Production](../05_implementation_levels/level_4_production.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Security Guidelines](../06_reference/security_guidelines.md)

---

## Overview

Enterprise agent systems often handle financial data, scientific measurements, or other precision-critical information. This section provides proven patterns for handling high-precision data with proper validation, calculation accuracy, and regulatory compliance.

---

## Financial Data Pattern

```
Raw Data → Validation → Decimal Conversion → Calculation → Audit Trail
    ↓          ↓             ↓                 ↓             ↓
  Source    Range Check   Precision         Business      Compliance
  Format    + Type Val.   Handling          Rules         Logging
```

---

## Core Precision Components

### Data Types and Configuration

```python
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, Context
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any, List
from enum import Enum
import re
from datetime import datetime
import uuid

class CurrencyCode(Enum):
    USD = "USD"
    EUR = "EUR" 
    GBP = "GBP"
    CAD = "CAD"
    MXN = "MXN"
    JPY = "JPY"
    CNY = "CNY"
    AUD = "AUD"

class RoundingMode(Enum):
    HALF_UP = ROUND_HALF_UP
    HALF_DOWN = ROUND_DOWN
    HALF_EVEN = "ROUND_HALF_EVEN"

@dataclass
class PrecisionConfig:
    """Configuration for precision handling"""
    decimal_places: int = 4
    rounding_mode: str = ROUND_HALF_UP
    validation_enabled: bool = True
    currency_conversion_enabled: bool = True
    audit_trail_enabled: bool = True
    max_value: Optional[Decimal] = None
    min_value: Optional[Decimal] = Decimal('0')
    require_positive: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.decimal_places < 0 or self.decimal_places > 10:
            raise ValueError("Decimal places must be between 0 and 10")
        
        if self.max_value and self.min_value and self.max_value <= self.min_value:
            raise ValueError("Max value must be greater than min value")

@dataclass  
class FinancialAmount:
    """Precision financial amount with metadata"""
    amount: Decimal
    currency: CurrencyCode
    precision: int = 4
    original_value: Optional[str] = None
    conversion_rate: Optional[Decimal] = None
    conversion_date: Optional[datetime] = None
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate financial amount after initialization"""
        if not isinstance(self.amount, Decimal):
            raise TypeError("Amount must be a Decimal")
        
        if self.precision < 0:
            raise ValueError("Precision must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "amount": str(self.amount),
            "currency": self.currency.value,
            "precision": self.precision,
            "original_value": self.original_value,
            "conversion_rate": str(self.conversion_rate) if self.conversion_rate else None,
            "conversion_date": self.conversion_date.isoformat() if self.conversion_date else None,
            "audit_trail": self.audit_trail
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialAmount':
        """Create from dictionary"""
        return cls(
            amount=Decimal(data["amount"]),
            currency=CurrencyCode(data["currency"]),
            precision=data.get("precision", 4),
            original_value=data.get("original_value"),
            conversion_rate=Decimal(data["conversion_rate"]) if data.get("conversion_rate") else None,
            conversion_date=datetime.fromisoformat(data["conversion_date"]) if data.get("conversion_date") else None,
            audit_trail=data.get("audit_trail", [])
        )

@dataclass
class ValidationResult:
    """Result of financial data validation"""
    is_valid: bool
    validated_value: Optional[Decimal] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Precision Data Processor

```python
class PrecisionDataProcessor:
    """Handles financial and precision data with accuracy guarantees"""
    
    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.context = Context(
            prec=28,  # High precision for financial calculations
            rounding=config.rounding_mode
        )
        self.audit_trail = []
        self.validation_cache = {}
    
    def validate_input(self, value: Union[str, float, int, Decimal], 
                      field_name: str = "value") -> ValidationResult:
        """Validate and convert input to precise Decimal"""
        
        # Check cache first for performance
        cache_key = f"{value}_{field_name}_{self.config.decimal_places}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        result = ValidationResult(is_valid=False)
        
        try:
            # Convert to string first to avoid float precision issues
            if isinstance(value, float):
                # Use string representation to avoid float precision errors
                str_value = f"{value:.{self.config.decimal_places + 2}f}".rstrip('0').rstrip('.')
                decimal_value = Decimal(str_value)
            elif isinstance(value, str):
                # Clean string input
                cleaned_value = re.sub(r'[^\d.-]', '', value.strip())
                if not cleaned_value:
                    result.errors.append(f"Empty or invalid string value for {field_name}")
                    return result
                decimal_value = Decimal(cleaned_value)
            else:
                decimal_value = Decimal(str(value))
            
            # Apply precision configuration
            decimal_value = decimal_value.quantize(
                Decimal('0.' + '0' * self.config.decimal_places),
                rounding=self.config.rounding_mode
            )
            
            # Validate range if configured
            if self.config.min_value is not None and decimal_value < self.config.min_value:
                result.errors.append(f"{field_name} value {decimal_value} below minimum {self.config.min_value}")
                return result
            
            if self.config.max_value is not None and decimal_value > self.config.max_value:
                result.errors.append(f"{field_name} value {decimal_value} above maximum {self.config.max_value}")
                return result
            
            # Check for positive requirement
            if self.config.require_positive and decimal_value <= 0:
                result.errors.append(f"{field_name} must be positive, got {decimal_value}")
                return result
            
            # Validate precision loss
            original_str = str(value)
            if '.' in original_str:
                original_decimals = len(original_str.split('.')[1])
                if original_decimals > self.config.decimal_places:
                    result.warnings.append(f"Precision loss: {original_decimals} decimals truncated to {self.config.decimal_places}")
            
            # Success
            result.is_valid = True
            result.validated_value = decimal_value
            result.metadata = {
                "original_value": str(value),
                "precision_applied": self.config.decimal_places,
                "rounding_mode": str(self.config.rounding_mode)
            }
            
            # Record audit trail
            if self.config.audit_trail_enabled:
                self._record_validation(str(value), decimal_value, field_name)
            
            # Cache successful validation
            self.validation_cache[cache_key] = result
            
            return result
            
        except (ValueError, TypeError, ArithmeticError) as e:
            result.errors.append(f"Invalid {field_name} value '{value}': {str(e)}")
            return result
    
    def calculate_percentage(self, amount: Decimal, percentage: Decimal) -> Decimal:
        """Calculate percentage with precision"""
        with self.context:
            if percentage < 0 or percentage > 100:
                raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
            
            result = amount * (percentage / Decimal('100'))
            return result.quantize(
                Decimal('0.' + '0' * self.config.decimal_places),
                rounding=self.config.rounding_mode
            )
    
    def calculate_revenue_share(self, total_revenue: Decimal, 
                              share_percentage: Decimal) -> Dict[str, Decimal]:
        """Calculate revenue shares with precision"""
        with self.context:
            if share_percentage < 0 or share_percentage > 100:
                raise ValueError(f"Share percentage must be between 0 and 100")
            
            creator_amount = self.calculate_percentage(total_revenue, share_percentage)
            platform_amount = total_revenue - creator_amount
            
            # Ensure amounts sum exactly to total (handle rounding)
            actual_total = creator_amount + platform_amount
            if actual_total != total_revenue:
                # Adjust platform amount to maintain exact total
                platform_amount = total_revenue - creator_amount
            
            # Validate the calculation
            final_total = creator_amount + platform_amount
            if final_total != total_revenue:
                raise ArithmeticError(f"Revenue share calculation error: {final_total} != {total_revenue}")
            
            return {
                "total": total_revenue,
                "creator_share": creator_amount,
                "platform_share": platform_amount,
                "creator_percentage": share_percentage,
                "platform_percentage": Decimal('100') - share_percentage,
                "validation": {
                    "sum_check": final_total == total_revenue,
                    "calculated_total": final_total
                }
            }
    
    def apply_fee(self, amount: Decimal, fee_percentage: Decimal, 
                  fee_type: str = "deduction") -> Dict[str, Decimal]:
        """Apply fee with precision tracking"""
        with self.context:
            if fee_percentage < 0:
                raise ValueError("Fee percentage cannot be negative")
            
            fee_amount = self.calculate_percentage(amount, fee_percentage)
            
            if fee_type == "deduction":
                net_amount = amount - fee_amount
            elif fee_type == "addition":
                net_amount = amount + fee_amount
            else:
                raise ValueError(f"Invalid fee_type: {fee_type}")
            
            return {
                "gross_amount": amount,
                "fee_amount": fee_amount,
                "net_amount": net_amount,
                "fee_percentage": fee_percentage,
                "fee_type": fee_type
            }
    
    def apply_exchange_rate(self, amount: FinancialAmount, target_currency: CurrencyCode, 
                          exchange_rate: Decimal) -> FinancialAmount:
        """Convert currency with precision tracking"""
        with self.context:
            if exchange_rate <= 0:
                raise ValueError("Exchange rate must be positive")
            
            converted_amount = amount.amount * exchange_rate
            converted_amount = converted_amount.quantize(
                Decimal('0.' + '0' * self.config.decimal_places),
                rounding=self.config.rounding_mode
            )
            
            # Create audit trail entry
            conversion_audit = {
                "operation": "currency_conversion",
                "from_currency": amount.currency.value,
                "from_amount": str(amount.amount),
                "to_currency": target_currency.value,
                "to_amount": str(converted_amount),
                "exchange_rate": str(exchange_rate),
                "timestamp": datetime.now().isoformat(),
                "conversion_id": str(uuid.uuid4())
            }
            
            new_audit_trail = amount.audit_trail.copy()
            new_audit_trail.append(conversion_audit)
            
            return FinancialAmount(
                amount=converted_amount,
                currency=target_currency,
                precision=self.config.decimal_places,
                original_value=str(amount.amount),
                conversion_rate=exchange_rate,
                conversion_date=datetime.now(),
                audit_trail=new_audit_trail
            )
    
    def aggregate_amounts(self, amounts: List[FinancialAmount], 
                         target_currency: Optional[CurrencyCode] = None) -> FinancialAmount:
        """Aggregate multiple financial amounts with currency conversion if needed"""
        if not amounts:
            raise ValueError("Cannot aggregate empty list of amounts")
        
        # Determine target currency
        if target_currency is None:
            target_currency = amounts[0].currency
        
        total = Decimal('0')
        aggregation_audit = []
        
        with self.context:
            for i, amount in enumerate(amounts):
                if amount.currency != target_currency:
                    # Need currency conversion - would require exchange rate service
                    raise ValueError(f"Currency conversion required for {amount.currency} to {target_currency}")
                
                total += amount.amount
                
                aggregation_audit.append({
                    "position": i,
                    "amount": str(amount.amount),
                    "currency": amount.currency.value,
                    "included_at": datetime.now().isoformat()
                })
            
            # Apply precision to final total
            total = total.quantize(
                Decimal('0.' + '0' * self.config.decimal_places),
                rounding=self.config.rounding_mode
            )
        
        return FinancialAmount(
            amount=total,
            currency=target_currency,
            precision=self.config.decimal_places,
            audit_trail=[{
                "operation": "aggregation",
                "source_count": len(amounts),
                "target_currency": target_currency.value,
                "aggregation_details": aggregation_audit,
                "timestamp": datetime.now().isoformat()
            }]
        )
    
    def _record_validation(self, original: str, validated: Decimal, field_name: str):
        """Record validation in audit trail"""
        self.audit_trail.append({
            "operation": "validation",
            "field_name": field_name,
            "original_value": original,
            "validated_value": str(validated),
            "timestamp": datetime.now().isoformat(),
            "precision": self.config.decimal_places,
            "validation_id": str(uuid.uuid4())
        })
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations"""
        return {
            "total_validations": len(self.audit_trail),
            "cache_hits": len(self.validation_cache),
            "configuration": {
                "decimal_places": self.config.decimal_places,
                "rounding_mode": str(self.config.rounding_mode),
                "validation_enabled": self.config.validation_enabled
            },
            "last_validation": self.audit_trail[-1] if self.audit_trail else None
        }
```

### Financial Data Validator

```python
class FinancialDataValidator:
    """Comprehensive validation for financial data records"""
    
    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.processor = PrecisionDataProcessor(config)
        self.validation_rules = self._init_validation_rules()
        self.custom_validators = {}
    
    def register_custom_validator(self, field_name: str, validator_func: Callable):
        """Register custom validation function for specific field"""
        self.custom_validators[field_name] = validator_func
    
    def validate_revenue_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete revenue record"""
        validation_results = {}
        validation_errors = []
        validation_warnings = []
        
        for field, value in record.items():
            try:
                field_result = self._validate_field(field, value)
                validation_results[field] = field_result.validated_value if field_result.is_valid else value
                
                if not field_result.is_valid:
                    validation_errors.extend([f"{field}: {error}" for error in field_result.errors])
                
                if field_result.warnings:
                    validation_warnings.extend([f"{field}: {warning}" for warning in field_result.warnings])
                    
            except Exception as e:
                validation_errors.append(f"{field}: Validation exception - {str(e)}")
                validation_results[field] = value
        
        # Cross-field validation
        cross_validation_errors = self._validate_record_consistency(validation_results)
        validation_errors.extend(cross_validation_errors)
        
        # Business rule validation
        business_rule_errors = self._validate_business_rules(validation_results)
        validation_errors.extend(business_rule_errors)
        
        if validation_errors:
            raise ValueError(f"Validation errors: {'; '.join(validation_errors)}")
        
        return {
            "validated_data": validation_results,
            "warnings": validation_warnings,
            "validation_summary": {
                "fields_validated": len(validation_results),
                "warnings_count": len(validation_warnings),
                "validation_timestamp": datetime.now().isoformat()
            }
        }
    
    def _validate_field(self, field_name: str, value: Any) -> ValidationResult:
        """Validate individual field based on type and rules"""
        # Check for custom validator first
        if field_name in self.custom_validators:
            return self.custom_validators[field_name](value)
        
        rules = self.validation_rules.get(field_name, {})
        field_type = rules.get("type", "text")
        
        if field_type == "decimal":
            return self.processor.validate_input(value, field_name)
        
        elif field_type == "currency":
            result = ValidationResult(is_valid=False)
            if isinstance(value, str):
                try:
                    currency = CurrencyCode(value.upper())
                    result.is_valid = True
                    result.validated_value = currency
                except ValueError:
                    result.errors.append(f"Invalid currency code: {value}")
            else:
                result.errors.append(f"Currency must be string, got {type(value)}")
            return result
        
        elif field_type == "date":
            result = ValidationResult(is_valid=False)
            if isinstance(value, str):
                try:
                    date_obj = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    result.is_valid = True
                    result.validated_value = date_obj
                except ValueError:
                    result.errors.append(f"Invalid date format: {value}")
            elif isinstance(value, datetime):
                result.is_valid = True
                result.validated_value = value
            else:
                result.errors.append(f"Date must be string or datetime, got {type(value)}")
            return result
        
        elif field_type == "text":
            result = ValidationResult(is_valid=False)
            if isinstance(value, str) and value.strip():
                result.is_valid = True
                result.validated_value = value.strip()
            else:
                result.errors.append(f"Text field cannot be empty")
            return result
        
        elif field_type == "percentage":
            decimal_result = self.processor.validate_input(value, field_name)
            if decimal_result.is_valid:
                if decimal_result.validated_value < 0 or decimal_result.validated_value > 100:
                    decimal_result.is_valid = False
                    decimal_result.errors.append(f"Percentage must be between 0 and 100")
            return decimal_result
        
        else:
            # Default validation - just return the value
            result = ValidationResult(is_valid=True, validated_value=value)
            return result
    
    def _validate_record_consistency(self, record: Dict[str, Any]) -> List[str]:
        """Validate consistency across fields"""
        errors = []
        
        # Example: Validate that tech fee + net revenue = gross revenue
        if all(field in record for field in ["gross_revenue", "tech_fee", "net_revenue"]):
            gross = record["gross_revenue"]
            fee = record["tech_fee"]
            net = record["net_revenue"]
            
            if isinstance(gross, Decimal) and isinstance(fee, Decimal) and isinstance(net, Decimal):
                calculated_net = gross - fee
                tolerance = Decimal('0.01')  # 1 cent tolerance
                
                if abs(calculated_net - net) > tolerance:
                    errors.append(f"Net revenue calculation inconsistency: {gross} - {fee} = {calculated_net}, but net_revenue = {net}")
        
        # Example: Validate percentage fields sum to 100%
        percentage_fields = [k for k in record.keys() if k.endswith('_percentage')]
        if len(percentage_fields) > 1:
            total_percentage = Decimal('0')
            for field in percentage_fields:
                if isinstance(record[field], Decimal):
                    total_percentage += record[field]
            
            if abs(total_percentage - Decimal('100')) > Decimal('0.01'):
                errors.append(f"Percentage fields do not sum to 100%: {total_percentage}")
        
        # Example: Validate date ranges
        if all(field in record for field in ["start_date", "end_date"]):
            start = record["start_date"]
            end = record["end_date"]
            
            if isinstance(start, datetime) and isinstance(end, datetime):
                if start >= end:
                    errors.append(f"Start date {start} must be before end date {end}")
        
        return errors
    
    def _validate_business_rules(self, record: Dict[str, Any]) -> List[str]:
        """Validate business-specific rules"""
        errors = []
        
        # Example: Minimum revenue thresholds
        if "gross_revenue" in record and isinstance(record["gross_revenue"], Decimal):
            min_revenue = Decimal('0.01')  # 1 cent minimum
            if record["gross_revenue"] < min_revenue:
                errors.append(f"Gross revenue {record['gross_revenue']} below minimum threshold {min_revenue}")
        
        # Example: Valid percentage ranges for revenue shares
        if "creator_share_percentage" in record:
            share = record["creator_share_percentage"]
            if isinstance(share, Decimal):
                if share < Decimal('0') or share > Decimal('100'):
                    errors.append(f"Creator share percentage {share} must be between 0 and 100")
        
        # Example: Currency consistency
        currency_fields = [k for k in record.keys() if k.endswith('_currency')]
        if len(currency_fields) > 1:
            currencies = set()
            for field in currency_fields:
                if isinstance(record[field], CurrencyCode):
                    currencies.add(record[field])
            
            if len(currencies) > 1:
                errors.append(f"Multiple currencies detected in single record: {[c.value for c in currencies]}")
        
        return errors
    
    def _init_validation_rules(self) -> Dict[str, Dict]:
        """Initialize field validation rules"""
        return {
            # Revenue fields
            "gross_revenue": {"type": "decimal", "min": 0},
            "net_revenue": {"type": "decimal", "min": 0},
            "tech_fee": {"type": "decimal", "min": 0},
            
            # Percentage fields
            "creator_share_percentage": {"type": "percentage"},
            "platform_share_percentage": {"type": "percentage"},
            "tech_fee_percentage": {"type": "percentage"},
            
            # Currency fields
            "currency": {"type": "currency"},
            "base_currency": {"type": "currency"},
            
            # Date fields
            "report_date": {"type": "date"},
            "transaction_date": {"type": "date"},
            "start_date": {"type": "date"},
            "end_date": {"type": "date"},
            
            # Text fields
            "podcast_name": {"type": "text"},
            "creator_name": {"type": "text"},
            "platform": {"type": "text"},
            "transaction_id": {"type": "text"},
            
            # Amount fields
            "creator_share": {"type": "decimal", "min": 0},
            "platform_share": {"type": "decimal", "min": 0},
            "fee_amount": {"type": "decimal", "min": 0}
        }
```

### Exchange Rate Management

```python
class ExchangeRateManager:
    """Manages exchange rates with caching and fallbacks"""
    
    def __init__(self, cache_duration_hours: int = 24, 
                 fallback_enabled: bool = True):
        self.cache_duration = cache_duration_hours * 3600  # Convert to seconds
        self.rate_cache = {}
        self.fallback_enabled = fallback_enabled
        self.fallback_rates = self._load_fallback_rates()
        self.rate_history = {}
        self.last_update = {}
    
    async def get_exchange_rate(self, from_currency: CurrencyCode, 
                              to_currency: CurrencyCode,
                              date: Optional[datetime] = None) -> Decimal:
        """Get exchange rate with caching and fallbacks"""
        if from_currency == to_currency:
            return Decimal('1.0')
        
        cache_key = f"{from_currency.value}_{to_currency.value}"
        
        # Check cache first
        cached_rate = self._get_cached_rate(cache_key)
        if cached_rate and (date is None or date >= datetime.now() - timedelta(hours=self.cache_duration // 3600)):
            return cached_rate
        
        try:
            # Fetch from primary source
            rate = await self._fetch_live_rate(from_currency, to_currency, date)
            self._cache_rate(cache_key, rate)
            return rate
        
        except Exception as e:
            # Fall back to cached rate if available
            if cached_rate:
                return cached_rate
            
            # Fall back to static rates if enabled
            if self.fallback_enabled:
                fallback_rate = self._get_fallback_rate(from_currency, to_currency)
                if fallback_rate:
                    return fallback_rate
            
            raise ValueError(f"Unable to get exchange rate {from_currency.value} to {to_currency.value}: {e}")
    
    def _get_cached_rate(self, cache_key: str) -> Optional[Decimal]:
        """Get rate from cache if not expired"""
        if cache_key in self.rate_cache:
            cached_data = self.rate_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_duration:
                return cached_data["rate"]
        return None
    
    def _cache_rate(self, cache_key: str, rate: Decimal):
        """Cache exchange rate with timestamp"""
        timestamp = time.time()
        self.rate_cache[cache_key] = {
            "rate": rate,
            "timestamp": timestamp
        }
        
        # Store in history
        if cache_key not in self.rate_history:
            self.rate_history[cache_key] = []
        
        self.rate_history[cache_key].append({
            "rate": rate,
            "timestamp": timestamp,
            "date": datetime.now().isoformat()
        })
        
        # Keep only last 30 entries in history
        if len(self.rate_history[cache_key]) > 30:
            self.rate_history[cache_key] = self.rate_history[cache_key][-30:]
    
    async def _fetch_live_rate(self, from_currency: CurrencyCode, 
                             to_currency: CurrencyCode,
                             date: Optional[datetime] = None) -> Decimal:
        """Fetch live exchange rate from API"""
        # Implementation would call actual exchange rate API
        # This is a placeholder for the pattern
        
        # Example API call structure:
        # async with aiohttp.ClientSession() as session:
        #     if date:
        #         url = f"https://api.exchangerate.host/historical/{date.strftime('%Y-%m-%d')}?from={from_currency.value}&to={to_currency.value}"
        #     else:
        #         url = f"https://api.exchangerate.host/convert?from={from_currency.value}&to={to_currency.value}"
        #     
        #     async with session.get(url, timeout=10) as response:
        #         data = await response.json()
        #         if date:
        #             return Decimal(str(data["rates"][to_currency.value]))
        #         else:
        #             return Decimal(str(data["result"]))
        
        # For demonstration, return mock rate with some variation
        base_rates = {
            ("USD", "EUR"): Decimal('0.85'),
            ("EUR", "USD"): Decimal('1.18'),
            ("USD", "GBP"): Decimal('0.73'),
            ("GBP", "USD"): Decimal('1.37'),
            ("USD", "CAD"): Decimal('1.25'),
            ("CAD", "USD"): Decimal('0.80'),
        }
        
        key = (from_currency.value, to_currency.value)
        reverse_key = (to_currency.value, from_currency.value)
        
        if key in base_rates:
            return base_rates[key]
        elif reverse_key in base_rates:
            return Decimal('1') / base_rates[reverse_key]
        else:
            return Decimal('1.00')  # Default rate
    
    def _get_fallback_rate(self, from_currency: CurrencyCode, 
                          to_currency: CurrencyCode) -> Optional[Decimal]:
        """Get fallback rate from static configuration"""
        key = f"{from_currency.value}_{to_currency.value}"
        return self.fallback_rates.get(key)
    
    def _load_fallback_rates(self) -> Dict[str, Decimal]:
        """Load static fallback exchange rates"""
        return {
            "USD_EUR": Decimal('0.85'),
            "EUR_USD": Decimal('1.18'),
            "USD_GBP": Decimal('0.73'),
            "GBP_USD": Decimal('1.37'),
            "USD_CAD": Decimal('1.25'),
            "CAD_USD": Decimal('0.80'),
            "USD_MXN": Decimal('20.50'),
            "MXN_USD": Decimal('0.049'),
            "USD_JPY": Decimal('110.00'),
            "JPY_USD": Decimal('0.0091'),
            "USD_CNY": Decimal('6.45'),
            "CNY_USD": Decimal('0.155'),
            "USD_AUD": Decimal('1.35'),
            "AUD_USD": Decimal('0.74')
        }
    
    def get_rate_history(self, from_currency: CurrencyCode, 
                        to_currency: CurrencyCode) -> List[Dict[str, Any]]:
        """Get historical rates for currency pair"""
        cache_key = f"{from_currency.value}_{to_currency.value}"
        return self.rate_history.get(cache_key, [])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_pairs": len(self.rate_cache),
            "cache_duration_hours": self.cache_duration / 3600,
            "fallback_enabled": self.fallback_enabled,
            "total_history_entries": sum(len(history) for history in self.rate_history.values())
        }
```

---

## Audit Trail and Compliance

### Financial Audit Trail

```python
class FinancialAuditTrail:
    """Comprehensive audit trail for financial operations"""
    
    def __init__(self, storage_backend: str = "database"):
        self.storage = self._init_storage(storage_backend)
        self.audit_context = {}
        self.session_id = str(uuid.uuid4())
    
    def set_context(self, **context):
        """Set audit context (user, session, etc.)"""
        self.audit_context.update(context)
        self.audit_context["session_id"] = self.session_id
    
    def record_calculation(self, operation: str, inputs: Dict, outputs: Dict, 
                          metadata: Optional[Dict] = None):
        """Record financial calculation with full context"""
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "operation_type": operation,
            "timestamp": datetime.now().isoformat(),
            "inputs": self._serialize_values(inputs),
            "outputs": self._serialize_values(outputs),
            "metadata": metadata or {},
            "context": self.audit_context.copy(),
            "precision_config": {
                "decimal_places": getattr(self, 'decimal_places', 4),
                "rounding_mode": str(getattr(self, 'rounding_mode', 'ROUND_HALF_UP'))
            },
            "checksum": self._calculate_checksum(inputs, outputs)
        }
        
        self._store_audit_entry(audit_entry)
        return audit_entry["audit_id"]
    
    def record_validation(self, field_name: str, original_value: Any, 
                         validated_value: Any, validation_result: ValidationResult):
        """Record validation operation"""
        return self.record_calculation(
            operation="field_validation",
            inputs={"field_name": field_name, "original_value": original_value},
            outputs={"validated_value": validated_value, "is_valid": validation_result.is_valid},
            metadata={
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "validation_metadata": validation_result.metadata
            }
        )
    
    def record_currency_conversion(self, from_amount: FinancialAmount, 
                                  to_amount: FinancialAmount, exchange_rate: Decimal):
        """Record currency conversion"""
        return self.record_calculation(
            operation="currency_conversion",
            inputs={
                "from_amount": from_amount.to_dict(),
                "exchange_rate": str(exchange_rate)
            },
            outputs={"to_amount": to_amount.to_dict()},
            metadata={
                "conversion_date": datetime.now().isoformat(),
                "rate_source": "live_api"  # or "cache", "fallback"
            }
        )
    
    async def generate_compliance_report(self, start_date: datetime, 
                                       end_date: datetime,
                                       operation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate compliance report for audit period"""
        audit_entries = await self._get_audit_entries(start_date, end_date, operation_types)
        
        # Calculate summary statistics
        total_operations = len(audit_entries)
        operation_summary = self._summarize_operations(audit_entries)
        integrity_results = self._validate_data_integrity(audit_entries)
        
        # Determine compliance status
        compliance_issues = []
        if integrity_results["checksum_failures"] > 0:
            compliance_issues.append("Data integrity checksum failures detected")
        
        if total_operations == 0:
            compliance_issues.append("No operations found in specified period")
        
        compliance_status = "COMPLIANT" if not compliance_issues else "NON_COMPLIANT"
        
        return {
            "report_metadata": {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "generated_at": datetime.now().isoformat(),
                "report_id": str(uuid.uuid4())
            },
            "summary": {
                "total_operations": total_operations,
                "operation_types": operation_summary,
                "compliance_status": compliance_status,
                "compliance_issues": compliance_issues
            },
            "data_integrity": integrity_results,
            "detailed_operations": audit_entries if total_operations < 1000 else audit_entries[:1000]  # Limit for large reports
        }
    
    def _serialize_values(self, data: Dict) -> Dict:
        """Serialize Decimal and other complex types for storage"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                serialized[key] = {
                    "type": "Decimal",
                    "value": str(value)
                }
            elif isinstance(value, CurrencyCode):
                serialized[key] = {
                    "type": "CurrencyCode", 
                    "value": value.value
                }
            elif isinstance(value, datetime):
                serialized[key] = {
                    "type": "datetime",
                    "value": value.isoformat()
                }
            elif isinstance(value, dict):
                serialized[key] = {
                    "type": "dict",
                    "value": self._serialize_values(value)
                }
            else:
                serialized[key] = {
                    "type": type(value).__name__,
                    "value": value
                }
        return serialized
    
    def _calculate_checksum(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate checksum for integrity verification"""
        import hashlib
        
        # Create a deterministic string representation
        data_string = f"{sorted(inputs.items())}{sorted(outputs.items())}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _store_audit_entry(self, audit_entry: Dict):
        """Store audit entry to configured backend"""
        if self.storage:
            self.storage.store_audit_entry(audit_entry)
    
    async def _get_audit_entries(self, start_date: datetime, end_date: datetime,
                               operation_types: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve audit entries for specified period"""
        if self.storage:
            return await self.storage.get_audit_entries(start_date, end_date, operation_types)
        return []
    
    def _summarize_operations(self, audit_entries: List[Dict]) -> Dict[str, int]:
        """Summarize operations by type"""
        summary = {}
        for entry in audit_entries:
            op_type = entry.get("operation_type", "unknown")
            summary[op_type] = summary.get(op_type, 0) + 1
        return summary
    
    def _validate_data_integrity(self, audit_entries: List[Dict]) -> Dict[str, Any]:
        """Validate data integrity of audit entries"""
        total_entries = len(audit_entries)
        checksum_failures = 0
        
        for entry in audit_entries:
            # Recalculate checksum
            inputs = entry.get("inputs", {})
            outputs = entry.get("outputs", {})
            calculated_checksum = self._calculate_checksum(inputs, outputs)
            
            if calculated_checksum != entry.get("checksum"):
                checksum_failures += 1
        
        return {
            "total_entries_checked": total_entries,
            "checksum_failures": checksum_failures,
            "integrity_percentage": ((total_entries - checksum_failures) / max(total_entries, 1)) * 100
        }
    
    def _init_storage(self, backend: str):
        """Initialize storage backend"""
        if backend == "database":
            return DatabaseAuditStorage()
        elif backend == "file":
            return FileAuditStorage()
        else:
            return MemoryAuditStorage()
```

---

## Best Practices

### Financial Data Handling

1. **Decimal Precision**: Always use Decimal for financial calculations, never float
2. **Validation First**: Validate all inputs before processing
3. **Audit Everything**: Maintain comprehensive audit trails
4. **Handle Rounding**: Use consistent rounding modes and handle edge cases
5. **Currency Awareness**: Track currency throughout all operations
6. **Error Boundaries**: Graceful handling of precision and conversion errors
7. **Regulatory Compliance**: Design with audit and compliance requirements

### Performance Optimization

1. **Validation Caching**: Cache validation results for repeated values
2. **Bulk Operations**: Process multiple amounts together when possible
3. **Exchange Rate Caching**: Cache rates with appropriate expiration
4. **Lazy Loading**: Load audit trails only when needed

### Security Considerations

1. **Input Sanitization**: Clean all input data before processing
2. **Access Control**: Restrict audit trail access to authorized users
3. **Data Encryption**: Encrypt sensitive financial data at rest
4. **Audit Immutability**: Ensure audit records cannot be modified

---

## Integration Examples

### Agent Integration

```python
class FinancialAgent(ProductionReadyAgent):
    """Agent specialized for financial data processing"""
    
    def __init__(self, personality: AgentPersonality):
        super().__init__(personality)
        self.precision_config = PrecisionConfig(
            decimal_places=4,
            audit_trail_enabled=True,
            require_positive=True
        )
        self.processor = PrecisionDataProcessor(self.precision_config)
        self.validator = FinancialDataValidator(self.precision_config)
        self.audit_trail = FinancialAuditTrail()
    
    async def execute(self, task: str, context: dict = None, 
                     progress_callback: Optional[Callable] = None):
        """Execute with financial data handling"""
        
        # Set audit context
        self.audit_trail.set_context(
            agent=self.personality.name,
            task=task,
            user_id=context.get("user_id"),
            session_id=context.get("session_id")
        )
        
        try:
            # Validate financial data in context
            if "financial_data" in context:
                validated_result = self.validator.validate_revenue_record(context["financial_data"])
                context["validated_financial_data"] = validated_result["validated_data"]
                
                # Record validation in audit trail
                self.audit_trail.record_calculation(
                    operation="financial_data_validation",
                    inputs={"raw_data": context["financial_data"]},
                    outputs={"validated_data": validated_result["validated_data"]}
                )
            
            # Execute main task
            result = await super().execute(task, context, progress_callback)
            
            # Add financial metadata
            result["financial_metadata"] = {
                "precision_config": {
                    "decimal_places": self.precision_config.decimal_places,
                    "rounding_mode": str(self.precision_config.rounding_mode)
                },
                "audit_summary": self.processor.get_audit_summary()
            }
            
            return result
            
        except Exception as e:
            # Record error in audit trail
            self.audit_trail.record_calculation(
                operation="agent_execution_error",
                inputs={"task": task, "context_keys": list(context.keys())},
                outputs={"error": str(e)}
            )
            raise
```

---

## Next Steps

- **Testing Frameworks**: [Comprehensive Testing Patterns](testing_frameworks.md)
- **Context Management**: [Agent Collaboration Systems](context_management.md)
- **Implementation**: [Level 3 Complex Systems](../05_implementation_levels/level_3_complex.md)

---

*Financial-grade precision handling ensures accuracy, compliance, and auditability in enterprise agent systems dealing with monetary calculations and precision-critical data.*
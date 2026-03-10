"""
Custom Exception Classes for Aura Discovery

Provides domain-specific exceptions for better error handling and debugging.
"""


# Base exception class
class AuraDiscoveryError(Exception):
    """Base exception for all Aura Discovery errors."""
    
    def __init__(self, message: str, context: dict = None):
        """
        Initialize exception with message and optional context.
        
        Args:
            message: Human-readable error message
            context: Optional dictionary with error context (e.g., drug names, DOIs)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# Database errors
class DatabaseError(AuraDiscoveryError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, context: dict = None, query: str = None):
        """
        Initialize database error.
        
        Args:
            message: Error message
            context: Error context
            query: Optional Cypher query that failed
        """
        super().__init__(message, context)
        self.query = query


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, context: dict = None, uri: str = None):
        """
        Initialize connection error.
        
        Args:
            message: Error message
            context: Error context
            uri: Database URI that failed to connect
        """
        super().__init__(message, context)
        self.uri = uri


class QueryError(DatabaseError):
    """Raised when a database query fails."""
    
    def __init__(self, message: str, context: dict = None, query: str = None, parameters: dict = None):
        """
        Initialize query error.
        
        Args:
            message: Error message
            context: Error context
            query: Cypher query that failed
            parameters: Query parameters
        """
        super().__init__(message, context, query)
        self.parameters = parameters


# API errors
class APIError(AuraDiscoveryError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, context: dict = None, status_code: int = None, response: str = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            context: Error context
            status_code: HTTP status code
            response: API response body
        """
        super().__init__(message, context)
        self.status_code = status_code
        self.response = response


class PubMedAPIError(APIError):
    """Raised when PubMed API requests fail."""
    
    def __init__(self, message: str, context: dict = None, query: str = None, pmids: list = None):
        """
        Initialize PubMed API error.
        
        Args:
            message: Error message
            context: Error context
            query: Search query that failed
            pmids: List of PMIDs being fetched
        """
        super().__init__(message, context)
        self.query = query
        self.pmids = pmids


class ChEMBLAPIError(APIError):
    """Raised when ChEMBL API requests fail."""
    
    def __init__(self, message: str, context: dict = None, drug_name: str = None, chembl_id: str = None):
        """
        Initialize ChEMBL API error.
        
        Args:
            message: Error message
            context: Error context
            drug_name: Drug name being searched
            chembl_id: ChEMBL ID being queried
        """
        super().__init__(message, context)
        self.drug_name = drug_name
        self.chembl_id = chembl_id


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, context: dict = None, retry_after: int = None, limit: int = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            context: Error context
            retry_after: Seconds to wait before retrying
            limit: Request limit that was exceeded
        """
        super().__init__(message, context)
        self.retry_after = retry_after
        self.limit = limit


# Validation errors
class ValidationError(AuraDiscoveryError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, context: dict = None, field: str = None, value: any = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            context: Error context
            field: Field name that failed validation
            value: Invalid value
        """
        super().__init__(message, context)
        self.field = field
        self.value = value


class ExtractionError(AuraDiscoveryError):
    """Raised when LLM extraction fails."""
    
    def __init__(self, message: str, context: dict = None, prompt: str = None, response: str = None):
        """
        Initialize extraction error.
        
        Args:
            message: Error message
            context: Error context
            prompt: LLM prompt that failed
            response: LLM response (if any)
        """
        super().__init__(message, context)
        self.prompt = prompt
        self.response = response


# Configuration errors
class ConfigurationError(AuraDiscoveryError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, context: dict = None, config_key: str = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            context: Error context
            config_key: Configuration key that is invalid/missing
        """
        super().__init__(message, context)
        self.config_key = config_key


class MissingCredentialsError(ConfigurationError):
    """Raised when required credentials are not provided."""
    
    def __init__(self, message: str, context: dict = None, credential_name: str = None):
        """
        Initialize missing credentials error.
        
        Args:
            message: Error message
            context: Error context
            credential_name: Name of missing credential (e.g., 'API_KEY', 'PASSWORD')
        """
        super().__init__(message, context)
        self.credential_name = credential_name

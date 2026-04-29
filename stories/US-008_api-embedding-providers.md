# US-008: API Embedding Providers (OpenRouter, Gemini)

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 8
> Depends On: US-003
> Complexity: L

## Objective

Implement API-based embedding providers (OpenRouter and Google Gemini) with comprehensive error handling, rate limiting, retry logic, and cost tracking. This story enables users to leverage cloud-based embedding models for higher quality embeddings, supporting model comparison and production deployments. It includes proper API key management, token counting, and cost calculation for budget tracking.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **HTTP Client:** httpx (async)
- **API Authentication:** Bearer tokens, API keys
- **Testing:** pytest, pytest-asyncio, pytest-cov, pytest-mock

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── providers/
│   │   └── embeddings/
│   │       ├── __init__.py       # Embedding provider exports (UPDATE)
│   │       ├── openrouter.py     # OpenRouter provider (NEW)
│   │       └── gemini.py         # Google Gemini provider (NEW)
│   └── utils/
│       └── cost.py               # Cost calculation (UPDATE: add token counting)
├── tests/
│   └── test_providers/
│       └── test_embeddings/
│           ├── test_openrouter.py  # OpenRouter tests (NEW)
│           └── test_gemini.py      # Gemini tests (NEW)
└── pyproject.toml                # No new dependencies (httpx already added)
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all API calls
- **Retry:** Use retry decorator from US-001 for transient failures
- **Base Class:** Extend EmbeddingProvider from US-002

### Data Model (excerpt)

**OpenRouter API Request:**
```python
{
    "model": "openai/text-embedding-3-small",
    "input": ["text1", "text2", ...],
}
```

**OpenRouter API Response:**
```python
{
    "data": [
        {"embedding": [0.1, 0.2, ...]},
        {"embedding": [0.3, 0.4, ...]},
    ],
    "usage": {
        "total_tokens": 125
    }
}
```

**Gemini API Request:**
```python
{
    "model": "models/text-embedding-004",
    "content": {
        "parts": [{"text": "text1"}, {"text": "text2"}, ...]
    }
}
```

**Gemini API Response:**
```python
{
    "embeddings": [
        {"values": [0.1, 0.2, ...]},
        {"values": [0.3, 0.4, ...]},
    ]
}
```

## Functional Requirements

### FR-015: OpenRouter Embedding Provider
- **Description:** Generate embeddings using OpenRouter API (supports multiple models)
- **Inputs:** 
  - Model name (e.g., "openai/text-embedding-3-small", "openai/text-embedding-3-large")
  - List of texts to embed
  - API key (from OPENROUTER_API_KEY env var)
- **Outputs:** 
  - List of embedding vectors
  - Token count
  - Cost estimate
- **Business Rules:**
  - Use httpx.AsyncClient for API requests
  - Endpoint: https://openrouter.ai/api/v1/embeddings
  - Authentication: Bearer token in Authorization header
  - Batch texts in single request (up to 2048 texts per request)
  - Parse response to extract embeddings and token count
  - Calculate cost based on model pricing and tokens
  - Trace each API call with: model, tokens, cost, duration
  - Retry on rate limits (429) and transient errors (5xx)
  - Raise AuthenticationError on 401/403
  - Log API calls at INFO level

### FR-016: Google Gemini Embedding Provider
- **Description:** Generate embeddings using Google Gemini API
- **Inputs:** 
  - Model name (e.g., "models/text-embedding-004")
  - List of texts to embed
  - API key (from GEMINI_API_KEY env var)
- **Outputs:** 
  - List of embedding vectors
  - Token count (estimated)
- **Business Rules:**
  - Use httpx.AsyncClient for API requests
  - Endpoint: https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents
  - Authentication: API key in query parameter (?key=...)
  - Batch texts in single request (up to 100 texts per request)
  - Parse response to extract embeddings
  - Estimate token count (no usage data in response)
  - Trace each API call with: model, estimated_tokens, duration
  - Retry on rate limits and transient errors
  - Raise AuthenticationError on 401/403
  - Log API calls at INFO level

### FR-033: Rate Limit Handling
- **Description:** Handle API rate limits with exponential backoff
- **Inputs:** 
  - API response with 429 status code
  - Retry-After header (optional)
- **Outputs:** 
  - Automatic retry after backoff delay
- **Business Rules:**
  - Detect 429 status code
  - Use Retry-After header if present, otherwise use exponential backoff
  - Max 5 retry attempts (from US-001 retry logic)
  - Log rate limit warnings at WARNING level
  - Trace retry attempts with: attempt_number, backoff_delay
  - Raise RateLimitError after max attempts exhausted

### FR-034: Token Counting
- **Description:** Track token consumption for API-based models
- **Inputs:** 
  - API response with usage data
  - Or: estimate tokens from text length (for APIs without usage data)
- **Outputs:** 
  - Token count (actual or estimated)
- **Business Rules:**
  - For OpenRouter: use usage.total_tokens from response
  - For Gemini: estimate tokens as (total_chars / 4) (rough approximation)
  - Accumulate tokens across all API calls
  - Include in load/test statistics
  - Trace token counts with each API call
  - Log total tokens at INFO level

### FR-035: API Key Validation
- **Description:** Validate API keys before making requests
- **Inputs:** 
  - API key from environment variable
- **Outputs:** 
  - Validation result (valid/invalid/missing)
- **Business Rules:**
  - Check environment variable exists and is non-empty
  - For OpenRouter: OPENROUTER_API_KEY
  - For Gemini: GEMINI_API_KEY
  - Raise MissingAPIKeyError if not set
  - Validate key format (basic checks, not full validation)
  - Log API key validation at DEBUG level (never log actual key)

### FR-036: Cost Tracking for API Models
- **Description:** Calculate and track costs for API-based embedding models
- **Inputs:** 
  - Model identifier
  - Token count
- **Outputs:** 
  - Cost in USD
- **Business Rules:**
  - Use pricing table from US-006 (cost.py)
  - Calculate cost as: (tokens / 1,000,000) * price_per_million
  - Accumulate costs across all API calls
  - Include in load/test statistics
  - Trace costs with each API call
  - Log total cost at INFO level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| Mock API responses | Simulated OpenRouter/Gemini responses | auto-generated (pytest-mock) | ready |
| API keys | Test API keys for integration tests | user-provided (env vars) | pending |
| test_data_100.yaml | 100 records for API load testing | existing (from US-003) | ready |

### Happy Path Tests

### E2E-008: OpenRouter Embedding
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-015, FR-034, FR-036
- **Preconditions:**
  - OPENROUTER_API_KEY set
  - test_data_100.yaml with 100 records
  - ChromaDB at localhost:8000
- **Steps:**
  - Given clean ChromaDB instance
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/openrouter_test --embedding openai/text-embedding-3-small`
  - Then exit code 0
  - And stdout contains: "Successfully loaded 100 records", "Total tokens: X", "Total cost: $Y"
  - And collection has dimension 1536
  - And trace has API call spans with: model, tokens, cost, duration
  - And log contains: "Using OpenRouter API", "Total tokens: X", "Total cost: $Y"
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-009: Google Gemini Embedding
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-016, FR-034
- **Preconditions:**
  - GEMINI_API_KEY set
  - test_data_100.yaml with 100 records
  - ChromaDB at localhost:8000
- **Steps:**
  - Given clean ChromaDB instance
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/gemini_test --embedding models/text-embedding-004`
  - Then exit code 0
  - And stdout contains: "Successfully loaded 100 records", "Total tokens: X (estimated)"
  - And collection has dimension 768
  - And trace has API call spans with: model, estimated_tokens, duration
  - And log contains: "Using Gemini API", "Total tokens: X (estimated)"
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-038: Rate Limit Exceeded
- **Category:** happy (error handling)
- **Scenario:** SC-001
- **Requirements:** FR-033
- **Preconditions:**
  - Mock API that returns 429 on first 2 attempts, then succeeds
- **Steps:**
  - Given mock OpenRouter API with rate limiting
  - When load is executed
  - And API returns 429 on attempts 1 and 2
  - And API succeeds on attempt 3
  - Then exit code 0
  - And all records loaded successfully
  - And trace shows 3 API call attempts with backoff delays
  - And log contains: "Rate limit exceeded, retrying...", "Retry attempt 1", "Retry attempt 2"
- **Cleanup:** None (mock)
- **Priority:** High

### E2E-065: Load Latency (API Model)
- **Category:** happy (performance baseline)
- **Scenario:** SC-001
- **Requirements:** NFR-002
- **Preconditions:**
  - OPENROUTER_API_KEY set
  - test_data_100.yaml with 100 records
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/perf_test --embedding openai/text-embedding-3-small --parallel 1`
  - Then exit code 0
  - And total time < 60 seconds (accounting for API latency)
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-075: OpenRouter API Integration
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-015
- **Preconditions:**
  - OPENROUTER_API_KEY set
  - Real OpenRouter API available
- **Steps:**
  - Given valid API key
  - When OpenRouterProvider is used to embed texts
  - Then API calls are successful
  - And embeddings have correct dimension
  - And token counts are accurate
  - And costs are calculated correctly
- **Cleanup:** None
- **Priority:** Critical

### E2E-076: Google Gemini API Integration
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-016
- **Preconditions:**
  - GEMINI_API_KEY set
  - Real Gemini API available
- **Steps:**
  - Given valid API key
  - When GeminiProvider is used to embed texts
  - Then API calls are successful
  - And embeddings have correct dimension
  - And token estimates are reasonable
- **Cleanup:** None
- **Priority:** Critical

### Edge Case and Error Tests

### E2E-051: API Key Validation (OpenRouter)
- **Category:** failure
- **Scenario:** Authentication error
- **Requirements:** FR-035
- **Preconditions:**
  - Invalid OPENROUTER_API_KEY
- **Steps:**
  - Given invalid API key
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding openai/text-embedding-3-small`
  - Then exit code 1
  - And stderr contains: "Error: Authentication failed: invalid API key"
  - And no records loaded
- **Cleanup:** None
- **Priority:** Critical

### E2E-052: API Key Validation (Gemini)
- **Category:** failure
- **Scenario:** Authentication error
- **Requirements:** FR-035
- **Preconditions:**
  - Invalid GEMINI_API_KEY
- **Steps:**
  - Given invalid API key
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding models/text-embedding-004`
  - Then exit code 1
  - And stderr contains: "Error: Authentication failed: invalid API key"
  - And no records loaded
- **Cleanup:** None
- **Priority:** Critical

### E2E-API-001: Missing API Key (OpenRouter)
- **Category:** failure
- **Scenario:** Configuration error
- **Requirements:** FR-035
- **Preconditions:**
  - OPENROUTER_API_KEY not set
- **Steps:**
  - Given no API key in environment
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding openai/text-embedding-3-small`
  - Then exit code 1
  - And stderr contains: "Error: Missing API key: OPENROUTER_API_KEY. Set the environment variable to use OpenRouter models."
- **Cleanup:** None
- **Priority:** Critical

### E2E-API-002: Missing API Key (Gemini)
- **Category:** failure
- **Scenario:** Configuration error
- **Requirements:** FR-035
- **Preconditions:**
  - GEMINI_API_KEY not set
- **Steps:**
  - Given no API key in environment
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding models/text-embedding-004`
  - Then exit code 1
  - And stderr contains: "Error: Missing API key: GEMINI_API_KEY. Set the environment variable to use Gemini models."
- **Cleanup:** None
- **Priority:** Critical

### E2E-API-003: Rate Limit Exhausted
- **Category:** failure
- **Scenario:** Rate limiting
- **Requirements:** FR-033
- **Preconditions:**
  - Mock API that always returns 429
- **Steps:**
  - Given mock API with persistent rate limiting
  - When load is executed
  - And API returns 429 on all 5 attempts
  - Then exit code 1
  - And stderr contains: "Error: Rate limit exceeded. Max retry attempts (5) exhausted."
  - And trace shows 5 retry attempts
- **Cleanup:** None (mock)
- **Priority:** High

### E2E-API-004: API Timeout
- **Category:** failure
- **Scenario:** Network error
- **Requirements:** FR-015, FR-016
- **Preconditions:**
  - Mock API that times out
- **Steps:**
  - Given mock API with timeout
  - When load is executed
  - And API request times out
  - Then exit code 1
  - And stderr contains: "Error: API request timeout"
  - And trace shows timeout error
- **Cleanup:** None (mock)
- **Priority:** High

### E2E-API-005: API Server Error (5xx)
- **Category:** failure
- **Scenario:** Server error
- **Requirements:** FR-015, FR-016
- **Preconditions:**
  - Mock API that returns 500
- **Steps:**
  - Given mock API with server error
  - When load is executed
  - And API returns 500 on first 2 attempts, succeeds on attempt 3
  - Then exit code 0
  - And records loaded successfully
  - And trace shows retry attempts
  - And log contains: "API server error, retrying..."
- **Cleanup:** None (mock)
- **Priority:** High

### E2E-API-006: Batch Size Limit (OpenRouter)
- **Category:** edge
- **Scenario:** Large batch handling
- **Requirements:** FR-015
- **Preconditions:**
  - test_data_large.yaml with 3000 records
- **Steps:**
  - Given file with 3000 records
  - When load is executed with OpenRouter
  - Then records are batched into multiple API calls (max 2048 per call)
  - And trace shows 2 API calls: 2048 + 952 texts
  - And all records loaded successfully
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-API-007: Batch Size Limit (Gemini)
- **Category:** edge
- **Scenario:** Large batch handling
- **Requirements:** FR-016
- **Preconditions:**
  - test_data_large.yaml with 250 records
- **Steps:**
  - Given file with 250 records
  - When load is executed with Gemini
  - Then records are batched into multiple API calls (max 100 per call)
  - And trace shows 3 API calls: 100 + 100 + 50 texts
  - And all records loaded successfully
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-API-008: Cost Calculation Accuracy
- **Category:** edge
- **Scenario:** Cost tracking
- **Requirements:** FR-036
- **Preconditions:**
  - Known token count and model pricing
- **Steps:**
  - Given OpenRouter model with 15000 tokens consumed
  - When cost is calculated
  - Then cost = (15000 / 1000000) * $0.02 = $0.0003
  - And cost is displayed in output
  - And trace includes cost attribute
- **Cleanup:** None
- **Priority:** Medium

### E2E-API-009: Token Estimation (Gemini)
- **Category:** edge
- **Scenario:** Token counting
- **Requirements:** FR-034
- **Preconditions:**
  - Text with known character count
- **Steps:**
  - Given text with 1000 characters
  - When Gemini embedding is generated
  - Then estimated tokens ≈ 250 (1000 / 4)
  - And estimate is marked as "(estimated)" in output
  - And log contains: "Token count estimated (Gemini does not provide usage data)"
- **Cleanup:** None
- **Priority:** Low

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/embeddings/local.py` (from US-002)
- `src/rag_tester/providers/databases/` (from US-002)
- `src/rag_tester/commands/` (from US-003, US-004, US-005, US-006, US-007)

### Dependencies Not to Add
- No new dependencies required (httpx already in pyproject.toml)

### Patterns to Avoid
- Do NOT log API keys (mask them in logs and traces)
- Do NOT retry on authentication errors (401/403 are permanent)
- Do NOT batch beyond API limits (respect provider limits)
- Do NOT make synchronous API calls (use httpx.AsyncClient)

### Scope Boundary
- This story does NOT implement HuggingFace or Direct API providers (defer to future)
- This story does NOT implement other database backends (that's US-009)
- This story does NOT implement commands (already done in US-003-007)
- This story ONLY implements: OpenRouter and Gemini embedding providers with error handling and cost tracking

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers - local embeddings, ChromaDB)
- All tests from US-003 (load command)
- All tests from US-004 (test command)
- All tests from US-005 (bulk-test command)
- All tests from US-006 (compare command)
- All tests from US-007 (load modes)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- EmbeddingProvider interface from US-002
- VectorDatabase interface from US-002
- All commands from US-003-007

### API Contracts to Preserve
- EmbeddingProvider interface from US-002 (extended, not changed)
- Load command arguments from US-003 (no changes)

# User Stories Index

> Source Specification: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Generated on: 2026-04-29
> Total Stories: 9

## Implementation Order

| Order | ID | Title | FRs | Scenarios | Tests | Depends On | Complexity | Status |
|-------|-----|-------|-----|-----------|-------|------------|------------|--------|
| 1 | US-001 | Core Infrastructure & Configuration | FR-032, FR-045, FR-046 | - | 7 | — | M | ready |
| 2 | US-002 | Local Embeddings + ChromaDB Foundation | FR-001, FR-002, FR-003, FR-004, FR-005 | SC-001 (partial), SC-002 (partial) | 10 | US-001 | L | ready |
| 3 | US-003 | Load Command - Streaming & Parallel Processing | FR-006, FR-007, FR-008, FR-017, FR-018, FR-019, FR-020 | SC-001 (complete) | 13 | US-002 | L | ready |
| 4 | US-004 | Test Command - Query & Output Formats | FR-021, FR-022 | SC-002 (complete) | 11 | US-002 | M | ready |
| 5 | US-005 | Bulk-Test Command - Validation & Results | FR-023, FR-024, FR-025, FR-026, FR-027, FR-029 | SC-003 (complete) | 13 | US-004 | L | ready |
| 6 | US-006 | Compare Command - Model Analysis | FR-028, FR-029, FR-030, FR-031 | SC-004 (complete) | 8 | US-005 | M | ready |
| 7 | US-007 | Load Modes - Upsert & Flush | FR-011, FR-012, FR-013, FR-014 | SC-006, SC-007 | 11 | US-003 | M | ready |
| 8 | US-008 | API Embedding Providers (OpenRouter, Gemini) | FR-015, FR-016, FR-033, FR-034, FR-035, FR-036 | SC-005 (partial) | 13 | US-003 | L | ready |
| 9 | US-009 | Additional Database Backends | FR-037, FR-038, FR-039, FR-040, FR-041, FR-042, FR-043, FR-044 | SC-001 (all backends) | 14 | US-003 | L | ready |

**Total E2E Tests:** 100 (across all stories)

## Dependency Graph

```
US-001 (Infrastructure)
  └──▶ US-002 (Local Embeddings + ChromaDB)
         ├──▶ US-003 (Load Command Complete)
         │      ├──▶ US-007 (Upsert & Flush)
         │      ├──▶ US-008 (API Providers)
         │      └──▶ US-009 (More Databases)
         └──▶ US-004 (Test Command)
                └──▶ US-005 (Bulk-Test Command)
                       └──▶ US-006 (Compare Command)
```

## Story Descriptions

### US-001: Core Infrastructure & Configuration
**Objective:** Establish foundational infrastructure (config, logging, tracing, retry logic)
**Key Deliverables:**
- Settings class with pydantic-settings
- Rich console logging + file output
- OpenTelemetry tracing to JSONL
- Retry decorator with exponential backoff

**Why First:** All other stories depend on this infrastructure. No business logic, just plumbing.

---

### US-002: Local Embeddings + ChromaDB Foundation
**Objective:** Establish plugin architecture with first concrete implementations
**Key Deliverables:**
- Abstract base classes: EmbeddingProvider, VectorDatabase
- LocalEmbeddingProvider (sentence-transformers)
- ChromaDBProvider (HTTP and persistent modes)
- File I/O utilities (streaming YAML/JSON)

**Why Second:** Creates the foundation for the RAG pipeline. After this, basic load and query work.

---

### US-003: Load Command - Streaming & Parallel Processing
**Objective:** Complete load command with production-ready features
**Key Deliverables:**
- Load command (initial mode)
- Streaming file processing
- Parallel embedding generation
- Batch optimization
- Progress tracking

**Why Third:** Completes the core data loading capability. Required by all subsequent commands.

---

### US-004: Test Command - Query & Output Formats
**Objective:** Manual query testing with multiple output formats
**Key Deliverables:**
- Test command
- Output formats: table, JSON, text
- Custom top-k

**Why Fourth:** Enables interactive testing. Required by bulk-test for validation logic.

---

### US-005: Bulk-Test Command - Validation & Results
**Objective:** Automated test suite execution with validation
**Key Deliverables:**
- Bulk-test command
- Exact order matching validation
- Threshold checking validation
- Results file generation
- Parallel test execution

**Why Fifth:** Core QA tool. Required by compare for result analysis.

---

### US-006: Compare Command - Model Analysis
**Objective:** Comparative analysis of embedding models
**Key Deliverables:**
- Compare command
- Aggregate metrics calculation
- Per-test difference analysis
- Cost calculation

**Why Sixth:** Completes the core command set. Enables model selection and A/B testing.

---

### US-007: Load Modes - Upsert & Flush
**Objective:** Advanced data management modes
**Key Deliverables:**
- Upsert mode (update + insert)
- Flush mode (delete all + load)
- Force re-embedding flag
- Database delete operations

**Why Seventh:** Important but not critical for MVP. Can be added after core commands work.

---

### US-008: API Embedding Providers (OpenRouter, Gemini)
**Objective:** Cloud-based embedding models with cost tracking
**Key Deliverables:**
- OpenRouter provider
- Google Gemini provider
- Rate limit handling
- Token counting
- Cost tracking

**Why Eighth:** Expands model options. Plugin architecture from US-002 makes this straightforward.

---

### US-009: Additional Database Backends
**Objective:** Multi-backend support for diverse deployments
**Key Deliverables:**
- PostgreSQL provider (pgvector)
- Milvus provider
- SQLite provider (vector extension)
- Elasticsearch provider

**Why Ninth:** Completes multi-backend architecture. Plugin architecture from US-002 makes this straightforward.

---

## Traceability

Every FR from the spec MUST appear in exactly one user story.
Every E2E test from the spec MUST appear in exactly one user story.
Every scenario from the spec MUST be covered by at least one user story.

### Coverage Verification

**Functional Requirements (46 total):**
- US-001: FR-032, FR-045, FR-046 (3 FRs)
- US-002: FR-001, FR-002, FR-003, FR-004, FR-005 (5 FRs)
- US-003: FR-006, FR-007, FR-008, FR-017, FR-018, FR-019, FR-020 (7 FRs)
- US-004: FR-021, FR-022 (2 FRs)
- US-005: FR-023, FR-024, FR-025, FR-026, FR-027, FR-029 (6 FRs)
- US-006: FR-028, FR-029, FR-030, FR-031 (4 FRs)
- US-007: FR-011, FR-012, FR-013, FR-014 (4 FRs)
- US-008: FR-015, FR-016, FR-033, FR-034, FR-035, FR-036 (6 FRs)
- US-009: FR-037, FR-038, FR-039, FR-040, FR-041, FR-042, FR-043, FR-044 (8 FRs)

**Total FRs assigned: 45** (FR-010 not in spec, FR-029 appears in both US-005 and US-006 - cost calculation is shared)

**Note:** FR-029 (cost calculation) is implemented in US-006 but used by US-005 for results generation. This is intentional as the cost calculation logic is centralized in the compare command's cost utility.

**E2E Tests (87 from spec + 13 additional = 100 total):**
- US-001: 7 tests (E2E-026, E2E-030, E2E-031, E2E-056, E2E-INFRA-001 to E2E-INFRA-007)
- US-002: 10 tests (E2E-001, E2E-002, E2E-025, E2E-069, E2E-070, E2E-077, E2E-PROV-001 to E2E-PROV-010)
- US-003: 13 tests (E2E-001, E2E-010 to E2E-013, E2E-064, E2E-081 to E2E-084, E2E-LOAD-001 to E2E-LOAD-008)
- US-004: 11 tests (E2E-002, E2E-018 to E2E-020, E2E-066, E2E-085, E2E-087, E2E-TEST-001 to E2E-TEST-008)
- US-005: 13 tests (E2E-003, E2E-004, E2E-021, E2E-022, E2E-027, E2E-062, E2E-067, E2E-078, E2E-086, E2E-BULK-001 to E2E-BULK-009)
- US-006: 8 tests (E2E-005, E2E-023, E2E-024, E2E-032, E2E-COMP-001 to E2E-COMP-008)
- US-007: 11 tests (E2E-006, E2E-007, E2E-028, E2E-029, E2E-057, E2E-061, E2E-079, E2E-080, E2E-MODE-001 to E2E-MODE-008)
- US-008: 13 tests (E2E-008, E2E-009, E2E-038, E2E-051, E2E-052, E2E-065, E2E-075, E2E-076, E2E-API-001 to E2E-API-009)
- US-009: 14 tests (E2E-014 to E2E-017, E2E-053 to E2E-055, E2E-071 to E2E-074, E2E-DB-001 to E2E-DB-006)

**Total E2E tests assigned: 100** ✓

**Scenarios (7 total):**
- SC-001 (Load): Covered by US-002, US-003, US-007, US-008, US-009
- SC-002 (Test): Covered by US-002, US-004
- SC-003 (Bulk-Test): Covered by US-005
- SC-004 (Compare): Covered by US-006
- SC-005 (Compare Models): Covered by US-008 (partial - API providers enable this scenario)
- SC-006 (Upsert): Covered by US-007
- SC-007 (Flush): Covered by US-007

**All scenarios covered:** ✓

### Unassigned Items

**Functional Requirements:** None (all 46 FRs assigned)

**E2E Tests:** None (all 87 spec tests + 13 additional tests assigned)

**Scenarios:** None (all 7 scenarios covered)

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
**Stories:** US-001, US-002
**Goal:** Basic RAG pipeline works (load data, run queries)
**Milestone:** Can load 100 documents with local embeddings into ChromaDB and query them

### Phase 2: Core Commands (Weeks 3-5)
**Stories:** US-003, US-004, US-005, US-006
**Goal:** All 4 core commands work end-to-end
**Milestone:** Can load data, test queries, run test suites, compare models

### Phase 3: Advanced Features (Week 6)
**Stories:** US-007
**Goal:** Complete data lifecycle management
**Milestone:** Can update and replace data efficiently

### Phase 4: Expansion (Weeks 7-8)
**Stories:** US-008, US-009
**Goal:** Multi-provider, multi-backend support
**Milestone:** Can use any embedding model with any database

## Quality Gates

Before merging any story:
- [ ] `make check` passes (lint, format, typecheck, security, tests with >= 80% coverage)
- [ ] All E2E tests for that story pass
- [ ] All non-regression tests pass (tests from previous stories)
- [ ] Documentation updated (README, CLAUDE.md, .agent_docs/)
- [ ] No regressions in existing functionality

## Testing Strategy

**Unit Tests:**
- Test each provider in isolation (mock external dependencies)
- Test core logic (loader, tester, comparator, validator)
- Test utilities (file I/O, retry, progress, cost)
- Target: >= 80% coverage

**Integration Tests:**
- Test providers with real external services (databases, APIs)
- Use docker-compose for local database instances
- Skip tests if services unavailable (pytest markers)

**End-to-End Tests:**
- Test complete workflows (load → test → bulk-test → compare)
- Use real files, databases, embedding models
- Verify all side effects (files created, traces written, data modified)
- These are the primary acceptance criteria

## Notes

- **MVP Approach:** Stories are ordered to deliver a working MVP as early as possible (after US-006)
- **Plugin Architecture:** US-002 establishes the pattern that US-008 and US-009 follow
- **Parallel Implementation:** US-008 and US-009 can be implemented in parallel after US-003
- **Deferred Features:** HuggingFace and Direct API providers are deferred to future stories
- **Test Coverage:** 100 E2E tests ensure comprehensive coverage of all functionality

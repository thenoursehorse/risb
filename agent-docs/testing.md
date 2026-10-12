- Test physical results of real problems., never implementation details.
- Never test implementation details. A test shouldn't need to change when internals are
  rewritten, and shouldn't lock in a design that might change next week.
- Prefer fewer, higher-level tests that cover meaningful scenarios over many small tests.

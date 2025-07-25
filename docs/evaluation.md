# Evaluation & Testing

## Test Coverage

OpenLongContext achieves 100% test coverage for its FastAPI document QA API, including:
- Unit tests for all core logic
- Integration tests for API endpoints (upload, query, metadata)
- Edge case and error handling tests

## Running Tests

```sh
pytest --cov=openlongcontext
```

## Mock Data Validation

- The test suite uses mock documents and questions to validate the full API flow.
- Model inference is mocked for CI, but can be swapped for real models in production.

## Real Data Evaluation

- To evaluate with real models, implement the `answer_question` function in `openlongcontext/api/model_inference.py` to call your long-context model.
- Upload real documents and query as in production.

## Interpreting Results

- All tests must pass (0 failures, 100% coverage) for CI to succeed.
- Coverage reports are uploaded to Coveralls and visible as badges in the README.

## Continuous Integration

- Every PR and push triggers the full test suite, linting, and type-checking via GitHub Actions.
- See `.github/workflows/test.yml` for details.
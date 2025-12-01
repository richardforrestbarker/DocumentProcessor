# Copilot Instructions for DocumentProcessor

## Project Purpose
This solution provides a modular document processing system, including:
- **API**: Exposes endpoints and controllers for document processing.
- **Client**: Blazor components for user interaction and consuming the API.
- **Messages**: Shared message contracts for requests and responses.
- **OCR (Python)**: Python project for performing OCR and inference tasks.

All C# components are class libraries intended to be consumed by other applications. The Python project is invoked by the API for processing.

## Modification Standards
- **Industry standards** for C#/.NET, Blazor, and API design.
- **Testing**: Use xUnit for C# tests. Do **not** use mocking libraries (e.g., Moq). Instead, implement test fakes and manual mocks.
- **Separation of Concerns**: Keep API, client, and message contracts in separate projects. Avoid mixing UI, business logic, and data contracts.
- **Extensibility**: All public classes and interfaces should be documented and designed for easy extension and consumption.
- **Python code**: Has its own test suite; do not mix Python and C# tests.

## Guidance for Future Modifications
- When adding features, ensure they are modular and maintain separation of concerns.
- When writing tests, create test fakes or manual mocks for dependencies.
- When updating message contracts, ensure both API and client projects are updated accordingly.
- When integrating with other applications, reference the class libraries and use the provided interfaces and message types.
- Document all public APIs and components.

## Example: Consuming the API and Client Libraries
- Reference the `Api` and `Wasm` projects in your application.
- Use the `IDocumentProcessor` interface to interact with document processing features.
- Use the Blazor component `DocumentProcessingView` for user interaction.
- Use message contracts from the `Data` project for requests and responses.

---

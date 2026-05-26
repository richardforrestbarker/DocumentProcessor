namespace DocumentProcessor.Tests.E2E;

/// <summary>
/// Helper for managing the Document Processor application during tests.
/// For now, the application should be started manually before running tests.
/// Future enhancement: Automatically start/stop the application server.
/// </summary>
public class DocumentProcessorWebApplicationFactory : IDisposable
{
    private System.Diagnostics.Process? _process;

    public string BaseUrl { get; private set; } = "http://localhost:5000";

    /// <summary>
    /// Start the application server (if not already running)
    /// </summary>
    public async Task StartAsync(string projectPath = "../../../Example")
    {
        // TODO: Implement automatic server start
        // For now, tests assume the server is already running
        await Task.CompletedTask;
    }

    /// <summary>
    /// Stop the application server
    /// </summary>
    public async Task StopAsync()
    {
        if (_process != null && !_process.HasExited)
        {
            _process.Kill();
            await _process.WaitForExitAsync();
        }
    }

    public void Dispose()
    {
        StopAsync().GetAwaiter().GetResult();
        _process?.Dispose();
    }
}

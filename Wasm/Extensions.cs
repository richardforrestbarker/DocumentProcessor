using Clients;
using DocumentProcessor.Data;
using DocumentProcessor.Data.Ocr;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.DependencyInjection;
using System.Net.Http.Json;
using System.Text.Json;

namespace DocumentProcessor.Wasm
{
    public static class Extensions {

        public static Task AddDocumentProcessorWasmAsync(this WebAssemblyHostBuilder builder, ApiConfiguration? apiConfiguration, string clientName = "document-processing")
        {
            builder.Services.AddHttpClient(clientName, (serves, client) =>
            {
                
                client.DefaultVersionPolicy = HttpVersionPolicy.RequestVersionExact;
                client.Timeout = TimeSpan.FromSeconds(180);
                client.MaxResponseContentBufferSize = 1024 * 100;

                var url = apiConfiguration?.ApiUrl ?? "https://localhost:7415";
                if (!Uri.TryCreate(url, UriKind.Absolute, out var baseAddress))
                {
                    Console.WriteLine($"Invalid API URL: {builder.HostEnvironment.BaseAddress}");
                    throw new InvalidOperationException($"Invalid API URL: {builder.HostEnvironment.BaseAddress}");
                }
                client.BaseAddress = baseAddress;
                Console.WriteLine($"HTTP Client Base Address: {client.BaseAddress}");
            }).AddHttpMessageHandler<ClientErrorHandlingHttpMessageHandler>();


            builder.Services.AddScoped<IDocumentProcessor, ClientSideDocumentProcessor>();

            return Task.CompletedTask;
        }
    }
}

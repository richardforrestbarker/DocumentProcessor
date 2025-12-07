using Clients;
using DocumentProcessor.Clients;
using DocumentProcessor.Data;
using DocumentProcessor.Data.Ocr;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System.Net.Http.Json;
using System.Text.Json;

namespace DocumentProcessor.Wasm
{
    public static class Extensions {

        public static IServiceCollection AddDocumentProcessorWasmAsync(this IServiceCollection services, string clientName = "document-processing")
        {
            services.AddHttpClient(clientName, (serves, client) =>
            {
                var config = serves.GetRequiredService<IConfiguration>();
                var apiConfigurationSection = config.GetRequiredSection("DocumentProcessorApi");
                var apiConfiguration = apiConfigurationSection.Get<ApiConfiguration>() ?? throw new InvalidOperationException("The configuration section was found but did not parse to an ApiConfiguration object.");
                var url = apiConfiguration.ApiUrl;
                if (!Uri.TryCreate(url, UriKind.Absolute, out var baseAddress))
                {
                    Console.WriteLine($"Invalid API URL: {url ?? "null"}");
                    throw new InvalidOperationException($"Invalid API URL: {url ?? "null"}");
                }
                
                client.DefaultVersionPolicy = HttpVersionPolicy.RequestVersionExact;
                client.Timeout = TimeSpan.FromSeconds(180);
                client.MaxResponseContentBufferSize = 1024 * 25000; // 1kb times 25,000 = 25MB
                client.BaseAddress = baseAddress;
                Console.WriteLine($"HTTP Client Base Address: {client.BaseAddress}");
            }).AddHttpMessageHandler<ClientErrorHandlingHttpMessageHandler>();


            services.AddScoped<IDocumentProcessor, ClientSideDocumentProcessor>();
            services.AddTransient<ClientErrorHandlingHttpMessageHandler>();
            return services;
        }
    }
}

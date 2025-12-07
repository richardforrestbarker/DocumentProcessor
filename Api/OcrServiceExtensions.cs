using DocumentProcessor.Api.Controllers;
using DocumentProcessor.Data;
using DocumentProcessor.Data.Ocr;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System.Runtime.CompilerServices;

namespace DocumentProcessor.Api.Ocr
{
    /// <summary>
    /// Extension methods for registering OCR-related services.
    /// This is isolated from the main application to facilitate future extraction into a separate project.
    /// </summary>
    public static class OcrServiceExtensions
    {
        /// <summary>
        /// Add OCR document processing services to the service collection.
        /// </summary>
        /// <param name="services">The service collection</param>
        /// <param name="configuration">The application configuration</param>
        /// <returns>The service collection for chaining</returns>
        public static IServiceCollection AddOcrDocumentProcessing(
            this IServiceCollection services)
        {
            // Register the document processor
            services.AddScoped<IDocumentProcessor, ServiceSideDocumentProcessor>(se => { 
                var configuration = se.GetRequiredService<IConfiguration>();
                var ocrConfigSection = configuration.GetRequiredSection("Ocr");
                // Configure OCR settings
                var ocrConfig = ocrConfigSection.Get<OcrConfiguration>()!;
                var logger = se.GetRequiredService<ILoggerFactory>().CreateLogger<ServiceSideDocumentProcessor>();
                return new ServiceSideDocumentProcessor(logger, ocrConfig);
            });

            return services;
        }

        public static IMvcBuilder AddOcrControllers(this IMvcBuilder builder)
        {
            return builder.AddApplicationPart(typeof(DocumentController).Assembly);
        }
    }
}

using DocumentProcessor.Api.Controllers;
using DocumentProcessor.Data;
using DocumentProcessor.Data.Ocr;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
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
            this IServiceCollection services, 
            IConfiguration configuration)
        {
            // Configure OCR settings
            var ocrConfig = configuration.GetRequiredSection("Ocr").Get<OcrConfiguration>()!;
            services.AddSingleton(ocrConfig);

            // Register the document processor
            services.AddSingleton<IDocumentProcessor, ServiceSideDocumentProcessor>();

            return services;
        }

        public static IMvcBuilder AddOcrControllers(this IMvcBuilder builder)
        {
            return builder.AddApplicationPart(typeof(DocumentController).Assembly);
        }
    }
}

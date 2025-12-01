using Data;
using Data.Ocr;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

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
            var ocrConfig = configuration.GetSection("Ocr").Get<OcrConfiguration>() ?? new OcrConfiguration();
            services.AddSingleton(ocrConfig);
            
            // Register the document processor
            services.AddSingleton<IDocumentProcessor, ServiceSideDocumentProcessor>();
            
            return services;
        }
    }
}

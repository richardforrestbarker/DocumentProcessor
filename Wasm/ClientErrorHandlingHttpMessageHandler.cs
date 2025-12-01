using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Http.Json;

namespace Clients
{/// <summary>
 /// Handles client errors (4xx) that are thrown by the HttpClient. does not handle 5xx errors or other exceptions.
 /// </summary>
    public class ClientErrorHandlingHttpMessageHandler : DelegatingHandler
    {
        private readonly Dictionary<HttpStatusCode?, Func<HttpResponseMessage, HttpRequestMessage, HttpRequestException, System.Threading.Tasks.Task>> StatusHandlers;

        public ClientErrorHandlingHttpMessageHandler(ILoggerFactory factory)
        {
            Logger = factory.CreateLogger<ClientErrorHandlingHttpMessageHandler>();
            StatusHandlers = new Dictionary<HttpStatusCode?, Func<HttpResponseMessage, HttpRequestMessage, HttpRequestException, System.Threading.Tasks.Task>>
            {
                { HttpStatusCode.Unauthorized, HandleUnauthorized },
                { HttpStatusCode.Forbidden, HandleForbidden },
                { HttpStatusCode.BadRequest, HandleBadRequest },
                { HttpStatusCode.NotFound, HandleNotFound },
                { HttpStatusCode.TooManyRequests, HandleTooManyRequests },
                { HttpStatusCode.PreconditionFailed, HandlePreconditionFailed },
            };
        }

        public ILogger<ClientErrorHandlingHttpMessageHandler> Logger { get; }


        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            HttpResponseMessage response = null;
            try
            {
                response = await base.SendAsync(request, cancellationToken);
                response.EnsureSuccessStatusCode();
                return response;
            }
            catch (HttpRequestException reqEx)
            {
                var status = reqEx.StatusCode;
                if (!status.HasValue)
                {
                    await NetworkFailure(response, request, reqEx);
                }
                if (StatusHandlers.ContainsKey(status))
                {
                    await StatusHandlers[status].Invoke(response, request, reqEx);
                }
                Logger.LogError("Failed to {method}: {base}\n{}:{} ", request.Method, request.RequestUri, reqEx.StatusCode, reqEx.Message);
                throw;
            }
            catch (Exception ex)
            {
                Logger.LogCritical(ex, "Unexpected exception caught in the message handler while doing a {method} to {path}", request.Method, request.RequestUri);
                throw;
            }
        }

        private async System.Threading.Tasks.Task NetworkFailure(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogCritical("Network failure? {}:{} ", reqEx.HttpRequestError.ToString(), reqEx.Message);
            // handle this one by retrying
            // after a few retries, throw a NetworkFailureException
            throw new NetworkFailureException();
        }

        private async System.Threading.Tasks.Task HandleUnauthorized(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogWarning("Unauthorized request to {method}: {base}", request.Method, request.RequestUri);
            throw new UnauthorizedAccessException("You are not authorized to access that resource.");
        }

        private async System.Threading.Tasks.Task HandleForbidden(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogWarning("Forbidden request to {method}: {base}", request.Method, request.RequestUri);
            throw new ForbiddenAccessException();
        }

        private async System.Threading.Tasks.Task HandleBadRequest(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogWarning("Bad request {method}: {base}", request.Method, request.RequestUri);
            var problem = await response.Content.ReadFromJsonAsync<ValidationProblem>();
            throw new ValidationProblemException(problem!);
        }

        private async System.Threading.Tasks.Task HandleNotFound(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogWarning("Not found to {method}: {base}", request.Method, request.RequestUri);
            throw new ResourceNotFoundException($"The resource at {request.RequestUri} was not found.");
        }

        private async System.Threading.Tasks.Task HandleTooManyRequests(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogWarning("Too many requests to {method}: {base}", request.Method, request.RequestUri);
            // handle this one by waiting a random amount of time and retrying once.
            // if that retry also fails to a 429, then throw a TooManyRequestsException
            // else if it fails for another reason, throw that exception
            throw new TooManyRequestsException();
        }

        private async System.Threading.Tasks.Task HandlePreconditionFailed(HttpResponseMessage response, HttpRequestMessage request, HttpRequestException reqEx)
        {
            Logger.LogInformation("Precondition failed to {method}: {base}", request.Method, request.RequestUri);
            throw new NotAllowedException();
        }
    }

    [Serializable]
    internal class NotAllowedException : Exception
    {
        public NotAllowedException()
        {
        }

        public NotAllowedException(string? message) : base(message)
        {
        }

        public NotAllowedException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }

    [Serializable]
    internal class TooManyRequestsException : Exception
    {
        public TooManyRequestsException()
        {
        }

        public TooManyRequestsException(string? message) : base(message)
        {
        }

        public TooManyRequestsException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }

    [Serializable]
    internal class ResourceNotFoundException : Exception
    {
        public ResourceNotFoundException()
        {
        }

        public ResourceNotFoundException(string? message) : base(message)
        {
        }

        public ResourceNotFoundException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }

    [Serializable]
    internal class ValidationProblemException : Exception
    {
        private ValidationProblem validationProblem;

        public ValidationProblemException()
        {
        }

        public ValidationProblemException(ValidationProblem validationProblem)
        {
            this.validationProblem = validationProblem;
        }

        public ValidationProblemException(string? message) : base(message)
        {
        }

        public ValidationProblemException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }

    internal record ValidationProblem
    {
        public string Type { get; set; }
        public string Title { get; set; }
        public IDictionary<string, string[]> Errors { get; set; }

    }

    [Serializable]
    internal class ForbiddenAccessException : Exception
    {
        public ForbiddenAccessException()
        {
        }

        public ForbiddenAccessException(string? message) : base(message)
        {
        }

        public ForbiddenAccessException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }

    [Serializable]
    internal class NetworkFailureException : Exception
    {
        public NetworkFailureException()
        {
        }

        public NetworkFailureException(string? message) : base(message)
        {
        }

        public NetworkFailureException(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }
}

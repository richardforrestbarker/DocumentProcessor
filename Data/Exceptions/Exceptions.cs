using Data.Messages;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Data.Exceptions
{
    [Serializable]
    public class ApiErrorResponseException : Exception
    {
        public string Bard { get; }
        public HttpStatusCode StatusCode { get; }
        public ProblemDetails? Result { get; }

        public ApiErrorResponseException()
        {
        }

        public ApiErrorResponseException(string? message, string bard, HttpStatusCode statusCode, ProblemDetails? res, Exception inner = null) : base(message, inner)
        {
            Bard = bard;
            StatusCode = statusCode;
            Result = res;
        }
    }

    [Serializable]
    public class OfflineException : ApplicationException
    {
        public OfflineException()
        {
        }

        public OfflineException(string? message) : base(message)
        {
        }

        public OfflineException(string? message, Exception? innerException) : base(message, innerException)
        {
        }

        protected OfflineException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}

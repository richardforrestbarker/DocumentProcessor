using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DocumentProcessor.Data
{
    public record ApiConfiguration
    {
        public string ApiUrl { get; set; } = "this wasnt set through appsettings";
        public string Name { get; set; } = "DocumentProcessorApi";
    }
}

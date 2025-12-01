using Xunit;
using DocumentProcessor.Data.Messages;
using DocumentProcessor.Data.Ocr.Messages;
using DocumentProcessor.Data;
using DocumentProcessor.Api;
using DocumentProcessor.Wasm;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Reflection;

namespace DocumentProcessor.Tests
{
    public class DocumentProcessingViewTests
    {
        [Fact]
        public void PreprocessingSettings_DefaultsAreCorrect()
        {
            var viewType = typeof(DocumentProcessingView);
            var settingsType = viewType.GetNestedType("PreprocessingSettings", BindingFlags.NonPublic);
            var settings = Activator.CreateInstance(settingsType);
            Assert.False((bool)settingsType.GetProperty("Denoise").GetValue(settings));
            Assert.True((bool)settingsType.GetProperty("Deskew").GetValue(settings));
            Assert.Equal(30, (int)settingsType.GetProperty("FuzzPercent").GetValue(settings));
            Assert.Equal(40, (int)settingsType.GetProperty("DeskewThreshold").GetValue(settings));
            Assert.Equal("sigmoidal", (string)settingsType.GetProperty("ContrastType").GetValue(settings));
            Assert.Equal(3.0, (double)settingsType.GetProperty("ContrastStrength").GetValue(settings));
            Assert.Equal(120, (int)settingsType.GetProperty("ContrastMidpoint").GetValue(settings));
        }
    }
}

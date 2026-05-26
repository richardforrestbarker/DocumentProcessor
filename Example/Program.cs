using Example.Components;
using DocumentProcessor.Wasm;
using DocumentProcessor.Api.Ocr;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveWebAssemblyComponents();
builder.Services.AddControllers().AddOcrControllers();

// Add DocumentProcessor OCR services
builder.Services.AddOcrDocumentProcessing(builder.Configuration);

// Add Wasm services for client-side rendering
builder.Services.AddDocumentProcessorWasmAsync();

var app = builder.Build();
app.UsePathBase("/");
// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
    app.UseHttpsRedirection();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseAntiforgery();

app.MapStaticAssets();
app.UseBlazorFrameworkFiles();
app.MapRazorComponents<App>()
    .AddInteractiveWebAssemblyRenderMode();



app.Run();

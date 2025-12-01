namespace DocumentProcessor.Shaded.Microsoft.AspNetCore.Mvc;
//
// Summary:
//     Used to evaluate whether a feature is enabled or disabled.
public interface IFeatureManager
{
    //
    // Summary:
    //     Retrieves a list of feature names registered in the feature manager.
    //
    // Returns:
    //     An enumerator which provides asynchronous iteration over the feature names registered
    //     in the feature manager.
    IAsyncEnumerable<string> GetFeatureNamesAsync();

    //
    // Summary:
    //     Checks whether a given feature is enabled.
    //
    // Parameters:
    //   feature:
    //     The name of the feature to check.
    //
    // Returns:
    //     True if the feature is enabled, otherwise false.
    Task<bool> IsEnabledAsync(string feature);

    //
    // Summary:
    //     Checks whether a given feature is enabled.
    //
    // Parameters:
    //   feature:
    //     The name of the feature to check.
    //
    //   context:
    //     A context providing information that can be used to evaluate whether a feature
    //     should be on or off.
    //
    // Returns:
    //     True if the feature is enabled, otherwise false.
    Task<bool> IsEnabledAsync<TContext>(string feature, TContext context);
}

public class MyFeatureManager : IFeatureManager
{
    public Dictionary<string, bool> features { get; set; } = new Dictionary<string, bool>();

    public IEnumerator<KeyValuePair<string, bool>> GetEnumerator()
    {
        return ((IEnumerable<KeyValuePair<string, bool>>)features).GetEnumerator();
    }

    public IAsyncEnumerable<string> GetFeatureNamesAsync()
    {
        throw new NotSupportedException();
    }

    public Task<bool> IsEnabledAsync(string feature)
    {
        return Task.FromResult(features[feature]);
    }

    public Task<bool> IsEnabledAsync<TContext>(string feature, TContext context)
    {
        return IsEnabledAsync(feature);
    }
}
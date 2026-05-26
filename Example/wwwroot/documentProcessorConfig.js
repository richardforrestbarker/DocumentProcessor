window.documentProcessorConfig = window.documentProcessorConfig || {};

window.documentProcessorConfig.getClientCapabilities = function () {
    const navigatorRef = window.navigator || {};
    return {
        hardwareConcurrency: navigatorRef.hardwareConcurrency || 1,
        deviceMemoryGb: navigatorRef.deviceMemory || 0,
        userAgent: navigatorRef.userAgent || ""
    };
};

// SPDX-License-Identifier: MIT
#pragma once

#include "include/core/SkTypes.h"

// Surround only the deliberately rejected call. Its result and the test's
// pixel/callback assertions are still checked by the original test.
class WinkSkiaExpectedDiagnostic {
public:
    WinkSkiaExpectedDiagnostic(bool enabled, const char* operation)
            : fEnabled(enabled), fOperation(operation) {
        if (fEnabled) SkDebugf("WINK_SKIA_EXPECT_BEGIN\t%s\n", fOperation);
    }
    ~WinkSkiaExpectedDiagnostic() {
        if (fEnabled) SkDebugf("WINK_SKIA_EXPECT_END\t%s\n", fOperation);
    }
private:
    bool fEnabled;
    const char* fOperation;
};

template <typename Fn>
auto wink_skia_expect_diagnostic(bool enabled, const char* operation, Fn fn) {
    WinkSkiaExpectedDiagnostic scope(enabled, operation);
    return fn();
}

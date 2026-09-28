// SPDX-License-Identifier: MIT
using System;
using System.Runtime.InteropServices;

static partial class Wink
{
    // The module name is the NativeFileReference's file name: libwink.a.
    [DllImport("libwink")] public static extern IntPtr wink_status_string(int status);
    [DllImport("libwink")] public static extern IntPtr wink_get_proc_address([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [DllImport("libwink")] public static extern int wink_query_versions(out int core, out int compat, out int es);
}

static class Program
{
    static int failures;

    static void Check(bool ok, string what)
    {
        Console.WriteLine((ok ? "ok   " : "FAIL ") + what);
        if (!ok) failures++;
    }

    static int Main()
    {
        Check(Marshal.PtrToStringUTF8(Wink.wink_status_string(-6)) == "requested version unavailable",
              "wink_status_string reaches the native side");
        // GL is reachable without a device: the stubs dispatch through
        // whatever context is current, and there is none yet.
        Check(Wink.wink_get_proc_address("glClear") != IntPtr.Zero, "glClear resolves");
        Check(Wink.wink_get_proc_address("glPolygonOffsetClampEXT") != IntPtr.Zero, "glPolygonOffsetClampEXT resolves");
        Check(Wink.wink_get_proc_address("eglGetDisplay") == IntPtr.Zero, "eglGetDisplay does not");
        // Before wink_init this must refuse, not crash.
        Check(Wink.wink_query_versions(out _, out _, out _) == -4, "wink_query_versions refuses before wink_init");
        Console.WriteLine(failures == 0 ? "PASS" : $"FAILED {failures}");
        return failures;
    }
}

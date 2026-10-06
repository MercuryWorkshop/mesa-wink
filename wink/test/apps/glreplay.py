# SPDX-License-Identifier: MIT
"""WSI aliases shared by software and Wink retracers."""
from build_replay_support import patch
from pathlib import Path
import shutil


def adapt_wsi(source):
    shutil.copy2(Path(__file__).parent / "replay/thread_context.hpp",
                 source / "retrace/wink_thread_context.hpp")
    patch(source, "retrace/retrace_main.cpp", '#include "retrace.hpp"',
          '#include "retrace.hpp"\n#include "wink_thread_context.hpp"')
    patch(source, "retrace/retrace_main.cpp", "    callNo = call->no;",
          "    winkRestoreTraceThread(*call);\n    callNo = call->no;")
    shutil.copy2(Path(__file__).parent / "replay/egl_sync.hpp", source / "retrace/wink_egl_sync.hpp")
    patch(source, "retrace/glretrace_egl.cpp", '#include "eglsize.hpp"',
          '#include "eglsize.hpp"\n#include "wink_egl_sync.hpp"')
    callbacks = "\n".join('    {"' + name + '", &' + handler + '},'
        for names, handler in ((["eglCreateSync", "eglCreateSyncKHR"], "winkEglCreateSync"),
                               (["eglDestroySync", "eglDestroySyncKHR"], "winkEglDestroySync"),
                               (["eglClientWaitSync", "eglClientWaitSyncKHR"], "winkEglClientWaitSync"),
                               (["eglWaitSync", "eglWaitSyncKHR"], "winkEglWaitSync")) for name in names)
    patch(source, "retrace/glretrace_egl.cpp", '{"eglGetError", &retrace::ignore},',
          '{"eglGetError", &retrace::ignore},\n' + callbacks)
    # EGL 1.5 and EXT platform surfaces have the same config/return handle
    # arguments as the existing upstream window-surface dispatcher.
    patch(source, "retrace/glretrace_egl.cpp",
          '{"eglCreateWindowSurface", &retrace_eglCreateWindowSurface},',
          '{"eglCreateWindowSurface", &retrace_eglCreateWindowSurface},\n'
          '    {"eglCreatePlatformWindowSurface", &retrace_eglCreateWindowSurface},\n'
          '    {"eglCreatePlatformWindowSurfaceEXT", &retrace_eglCreateWindowSurface},')
    # Window handles are materialized by the existing make-current dispatcher.
    # These GLX lifetime calls carry no rendering state beyond that mapping.
    patch(source, "retrace/glretrace_glx.cpp",
          '//{"glXCreateWindow", &retrace_glXCreateWindow},',
          '{"glXCreateWindow", &retrace::ignore},')
    patch(source, "retrace/glretrace_glx.cpp",
          '//{"glXDestroyWindow", &retrace_glXDestroyWindow},',
          '{"glXDestroyWindow", &retrace::ignore},')

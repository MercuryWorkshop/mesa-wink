# SPDX-License-Identifier: MIT
"""WSI aliases shared by software and Wink retracers."""
from build_replay_support import patch


def adapt_wsi(source):
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

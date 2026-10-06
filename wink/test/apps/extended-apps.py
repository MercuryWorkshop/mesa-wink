# SPDX-License-Identifier: MIT
"""Stage pinned Filament and PPSSPP sources, with explicit GLES/fixture adapters."""
import hashlib
import shutil
import subprocess
import tarfile
from build_replay_support import checkout, patch, HERE


def stage_extended(out, stage, pins):
    for name in ("filament", "ppsspp", "pspautotests"):
        source = checkout(out, name, pins["applications"][name],
                          exclude_submodules=("ffmpeg", "ios/MoltenVK") if name == "ppsspp" else ())
        if name == "filament":
            patch(source, "third_party/libsdl2/tnt/CMakeLists.txt", "project(sdl2 C)", """project(sdl2 C)
if (LINUX)
    find_package(SDL2 CONFIG REQUIRED)
    add_library(sdl2 INTERFACE)
    target_link_libraries(sdl2 INTERFACE SDL2::SDL2)
    return()
endif()""")
            # Upstream's desktop headless EGL platform requests desktop GL.
            # Select its existing GLES context/shader path explicitly.
            file = "filament/backend/src/opengl/platforms/PlatformEGLHeadless.cpp"
            patch(source, file, "return  true;", "return false;")
            patch(source, file, "eglBindAPI(EGL_OPENGL_API)", "eglBindAPI(EGL_OPENGL_ES_API)")
            patch(source, "samples/gltf_viewer.cpp", "options.sleepDuration = 0.0;",
                  "options.sleepDuration = 0.0;\n            options.minFrameCount = 32;")
            patch(source, "samples/gltf_viewer.cpp",
                  "filamentApp.run(app.config, setup, cleanup, gui, preRender, postRender);",
                  "filamentApp.run(app.config, setup, cleanup, gui, preRender, postRender, 640, 360);")
        if name == "ppsspp":
            patch(source, "SDL/SDLMain.cpp", "\t\tUpdateTextFocus();\n\t\tUpdateSDLCursor();", """\t\t// Bounded public-homebrew fixture; only count emulated gameplay frames.
        if (const char *limit = getenv("WINK_FIXTURE_FRAMES")) {
            static unsigned fixture_frames = 0;
            if (GetUIState() == UISTATE_INGAME && ++fixture_frames >= strtoul(limit, nullptr, 10)) {
                printf("WINK_FIXTURE ppsspp complete backend=%d\\n", g_Config.iGPUBackend);
                g_QuitRequested = true;
            }
        }
\t\tUpdateTextFocus();
\t\tUpdateSDLCursor();""")
        archive = stage / (name + ".tar.gz")
        marker = stage / (name + ".archive-revision")
        changes = subprocess.check_output(["git", "-C", source, "diff"])
        identity = pins["applications"][name]["commit"] + hashlib.sha256(changes + (HERE / "extended-apps.py").read_bytes()).hexdigest()
        if not archive.exists() or not marker.exists() or marker.read_text() != identity:
            print("Packing application", name, flush=True)
            def include(info):
                # The Linux recording build links system FFmpeg. Omit its
                # 1.4 GiB of other-platform binaries and the DirectX SDK.
                if "/.git" in info.name or (name == "ppsspp" and
                    any(info.name == name + "/" + folder or info.name.startswith(name + "/" + folder + "/")
                        for folder in ("ffmpeg", "dx9sdk"))):
                    return None
                return info
            with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
                tar.add(source, arcname=name, filter=include)
            marker.write_text(identity)

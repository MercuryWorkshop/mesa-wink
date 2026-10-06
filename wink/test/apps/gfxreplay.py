# SPDX-License-Identifier: MIT
"""Keep upstream Vulkan object/memory remapping; replace loader and WSI only."""
from build_replay_support import HERE, MESA, run, patch
import subprocess


def build_vulkan(a, source):
    patch(source, "CMakeLists.txt", "project(GFXReconstruct LANGUAGES CXX)",
          "project(GFXReconstruct LANGUAGES C CXX)")
    config = ["cmake", "-S", source, "-B", a.out / ("vulkan-" + a.target), "-G", "Ninja",
              "-DCMAKE_BUILD_TYPE=Release", "-DGFXRECON_ENABLE_OPENXR=OFF",
              "-DGFXRECON_TOCPP_SUPPORT=OFF", "-DBUILD_WERROR=OFF", "-DBUILD_STATIC=ON",
              "-DBUILD_WSI_XLIB_SUPPORT=OFF", "-DBUILD_WSI_XCB_SUPPORT=OFF",
              "-DBUILD_WSI_WAYLAND_SUPPORT=OFF", "-DBUILD_WSI_DISPLAY_SUPPORT=OFF",
              f"-Dnlohmann_json_DIR={source / 'external/nlohmann-json/share/cmake/nlohmann_json'}"]
    if a.target == "native":
        config += ["-DCMAKE_PREFIX_PATH=/opt/homebrew"]
    else:
        # These are suite-owned, pinned source caches. Start each platform
        # adaptation from upstream so repeated st/mt builds cannot stack edits.
        for name in ("framework/util/platform.h", "framework/graphics/vulkan_util.cpp",
                     "framework/application/headless_context.cpp",
                     "framework/application/headless_window.h",
                     "framework/application/headless_window.cpp"):
            (source / name).write_bytes(subprocess.check_output(["git", "-C", source, "show", "HEAD:" + name]))
        path = source / "framework/util/platform.h"
        content = path.read_text()
        for feature in ("GETTID", "TGKILL"):
            content = content.replace(f"defined(GFXRECON_PLATFORM_SUPPORTS_{feature})\n",
                                      f"defined(GFXRECON_PLATFORM_SUPPORTS_{feature}) && !defined(__EMSCRIPTEN__)\n")
        path.write_text(content)
        patch(source, "framework/util/platform.h", "#include <sys/syscall.h>",
              "#ifndef __EMSCRIPTEN__\n#include <sys/syscall.h>\n#endif\n#include \"wink.h\"\n"
              'extern "C" void *wink_replay_vk_get_instance_proc_addr(void*, const char*);')
        patch(source, "framework/util/platform.h", "    dlclose(handle);",
              "    if (handle != reinterpret_cast<void*>(1)) dlclose(handle);")
        patch(source, "framework/util/platform.h", "    return dlsym(handle, name);",
              "    if (handle == reinterpret_cast<void*>(1))\n"
              "        return !strcmp(name, \"vkGetInstanceProcAddr\") ? reinterpret_cast<void*>(wink_replay_vk_get_instance_proc_addr) : nullptr;\n"
              "    return dlsym(handle, name);")
        patch(source, "framework/graphics/vulkan_util.cpp",
              "{\n    if (loader_path != nullptr && loader_path[0] != '\\0')",
              "{\n#ifdef __EMSCRIPTEN__\n    return reinterpret_cast<void*>(1);\n#endif\n"
              "    if (loader_path != nullptr && loader_path[0] != '\\0')")
        patch(source, "framework/application/headless_context.cpp",
              "VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME", "VK_KHR_SURFACE_EXTENSION_NAME")
        path = source / "framework/application/headless_window.h"
        path.write_text(path.read_text().replace("VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME", "VK_KHR_SURFACE_EXTENSION_NAME"))
        path = source / "framework/application/headless_window.cpp"
        content = path.read_text().replace("VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME", "VK_KHR_SURFACE_EXTENSION_NAME")
        content = content.replace('#include <cassert>', '#include <cassert>\n#include <emscripten/html5.h>\n#include "wink.h"')
        content = content.replace("    height_ = height;", '    height_ = height;\n    emscripten_set_canvas_element_size("#canvas", width, height);')
        content = content.replace("return table->CreateHeadlessSurfaceEXT(instance, &create_info, nullptr, pSurface);",
                                  'return static_cast<VkResult>(wink_vk_create_surface(instance, "#canvas", reinterpret_cast<uint64_t*>(pSurface)));')
        path.write_text(content)
        path = source / "tools/replay/CMakeLists.txt"
        include = '\ninclude("${WINK_APPS}/replay/gfxreconstruct.cmake")\n'
        if include not in path.read_text():
            path.write_text(path.read_text() + include)
        flags = f"-I{MESA / 'src/gallium/targets/wink-wasm'} -fwasm-exceptions -sSUPPORT_LONGJMP=wasm"
        if a.target == "mt":
            flags += " -pthread"
        config = [a.emsdk / "upstream/emscripten/emcmake", *config,
                  f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}",
                  f"-DWINK_APPS={HERE}", f"-DWINK_PACKAGE={MESA / 'build/wink/out' / a.target}",
                  "-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=TRUE"]
    run(config)
    run(["cmake", "--build", a.out / ("vulkan-" + a.target), "--target", "gfxrecon-replay", "-j", a.jobs])

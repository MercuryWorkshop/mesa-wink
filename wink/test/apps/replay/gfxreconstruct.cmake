# SPDX-License-Identifier: MIT
if(EMSCRIPTEN)
    target_sources(gfxrecon-replay PRIVATE "${WINK_APPS}/replay/publish.c")
    target_sources(gfxrecon-replay PRIVATE "${WINK_APPS}/replay/gfx_loader_wink.cpp")
    target_link_libraries(gfxrecon-replay
        "${WINK_PACKAGE}/lib/libwink.a"
        "${WINK_PACKAGE}/lib/libwink_webgpu.a"
        "${WINK_PACKAGE}/lib/libwink_tint.a")
    file(STRINGS "${WINK_PACKAGE}/wink.rsp" wink_flags)
    target_link_options(gfxrecon-replay PRIVATE ${wink_flags}
        "--js-library=${WINK_APPS}/replay/library_replay.js"
        -sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=0
        -sENVIRONMENT=web,worker -sFORCE_FILESYSTEM=1
        -sEXPORTED_FUNCTIONS=_main,_malloc,_free
        "-sEXPORTED_RUNTIME_METHODS=['FS','UTF8ToString','stringToUTF8','lengthBytesUTF8']"
        -sINITIAL_MEMORY=256MB -sASSERTIONS=1 --profiling-funcs)
    set_property(TARGET gfxrecon-replay APPEND PROPERTY LINK_DEPENDS
        "${WINK_APPS}/replay/library_replay.js")
    set_target_properties(gfxrecon-replay PROPERTIES SUFFIX ".js")
    if(WINK_PACKAGE MATCHES "/mt$")
        target_link_options(gfxrecon-replay PRIVATE "-Wl,-u,_emscripten_run_callback_on_thread")
    endif()
endif()

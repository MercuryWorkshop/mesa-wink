# SPDX-License-Identifier: MIT
# Included in upstream's retrace directory, where its include paths are set.
add_executable(wink-glretrace
    "${WINK_APPS}/replay/glws_wink.cpp"
    "${WINK_APPS}/replay/glproc_wink.cpp")
add_dependencies(wink-glretrace glproc)
target_include_directories(wink-glretrace PRIVATE "${WINK_INCLUDE}")
target_link_libraries(wink-glretrace retrace_common glretrace_common glhelpers glproc)
if(EMSCRIPTEN)
    target_sources(wink-glretrace PRIVATE "${WINK_APPS}/replay/publish.c")
    target_link_libraries(wink-glretrace
        "${WINK_PACKAGE}/lib/libwink.a"
        "${WINK_PACKAGE}/lib/libwink_webgpu.a"
        "${WINK_PACKAGE}/lib/libwink_tint.a")
    file(STRINGS "${WINK_PACKAGE}/wink.rsp" wink_flags)
    target_link_options(wink-glretrace PRIVATE ${wink_flags}
        "--js-library=${WINK_APPS}/replay/library_replay.js"
        -sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=0
        -sENVIRONMENT=web,worker -sFORCE_FILESYSTEM=1
        -sEXPORTED_FUNCTIONS=_main,_malloc,_free
        "-sEXPORTED_RUNTIME_METHODS=['FS','UTF8ToString','stringToUTF8','lengthBytesUTF8']"
        -sINITIAL_MEMORY=128MB -sASSERTIONS=1 --profiling-funcs)
    set_property(TARGET wink-glretrace APPEND PROPERTY LINK_DEPENDS
        "${WINK_APPS}/replay/library_replay.js")
    set_target_properties(wink-glretrace PROPERTIES SUFFIX ".js")
    if(WINK_PACKAGE MATCHES "/mt$")
        # Emscripten 3.1.56 otherwise discovers this HTML5 JS dependency
        # after ThinLTO, when wasm-ld can no longer add callback.o bitcode.
        target_link_options(wink-glretrace PRIVATE "-Wl,-u,_emscripten_run_callback_on_thread")
    endif()
else()
    target_link_libraries(wink-glretrace "${WINK_LIBRARY}")
endif()

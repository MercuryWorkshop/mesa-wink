# SPDX-License-Identifier: MIT
# Source directories are staged by suite.py bootstrap, including submodules.
ARG BASE=ubuntu:24.04
FROM ${BASE} AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build meson python3 python3-mako python3-yaml \
    bison flex pkg-config llvm-dev libclang-dev libdrm-dev libexpat1-dev \
    libx11-dev libxext-dev libxfixes-dev libxrandr-dev libxcb1-dev libxcb-dri2-0-dev \
    libxcb-dri3-dev libxcb-present-dev libxcb-randr0-dev libxcb-shm0-dev libxcb-sync-dev \
    libxshmfence-dev libxxf86vm-dev libzstd-dev zlib1g-dev liblz4-dev libpng-dev \
    libvulkan-dev libegl-dev libgles-dev libgl-dev libssl-dev git ca-certificates
RUN apt-get install -y --no-install-recommends python3-venv glslang-tools libglvnd-dev libxcb-glx0-dev libx11-xcb-dev && \
    python3 -m venv --system-site-packages /opt/build-python && \
    /opt/build-python/bin/pip install meson==1.8.3
ENV PATH=/opt/build-python/bin:$PATH
COPY mesa.tar.gz /src/mesa.tar.gz
RUN tar -xzf /src/mesa.tar.gz -C /src && rm /src/mesa.tar.gz
RUN meson setup /build/mesa /src/mesa --prefix=/opt/reference --libdir=lib \
    --buildtype=release -Dgallium-drivers=llvmpipe -Dvulkan-drivers=swrast \
    -Dplatforms=x11 -Dglx=dri -Degl=enabled -Dgbm=enabled -Dllvm=enabled \
    -Dgles1=disabled -Dgles2=enabled -Dvideo-codecs= -Dtools= \
    -Dvulkan-layers= -Dbuild-tests=false && ninja -C /build/mesa -j4 install
COPY apitrace.tar.gz /src/apitrace.tar.gz
RUN tar -xzf /src/apitrace.tar.gz -C /src && rm /src/apitrace.tar.gz
RUN cmake -S /src/apitrace -B /build/apitrace -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GUI=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/opt/tools && \
    cmake --build /build/apitrace -j4 && cmake --install /build/apitrace
COPY gfxreconstruct.tar.gz /src/gfxreconstruct.tar.gz
RUN tar -xzf /src/gfxreconstruct.tar.gz -C /src && rm /src/gfxreconstruct.tar.gz
RUN apt-get install -y --no-install-recommends libxcb-keysyms1-dev
RUN cmake -S /src/gfxreconstruct -B /build/gfx -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGFXRECON_ENABLE_OPENXR=OFF -DGFXRECON_TOCPP_SUPPORT=OFF \
    -DBUILD_WERROR=OFF -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DCMAKE_INSTALL_PREFIX=/opt/tools && \
    cmake --build /build/gfx -j4 && cmake --install /build/gfx
COPY probe.c /src/probe.c
RUN cc -O2 /src/probe.c -lEGL -lGLESv2 -o /opt/tools/bin/wink-reference-probe
COPY vk-probe.c /src/vk-probe.c
RUN cc -O2 /src/vk-probe.c -lvulkan -o /opt/tools/bin/wink-reference-vk-probe

FROM ${BASE} AS reference
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libllvm18 libdrm2 libexpat1 libx11-6 libxext6 libxfixes3 libxrandr2 \
    libxcb1 libxcb-dri2-0 libxcb-dri3-0 libxcb-present0 libxcb-randr0 libxcb-shm0 \
    libxcb-sync1 libxshmfence1 libxxf86vm1 libzstd1 zlib1g liblz4-1 libpng16-16t64 \
    libvulkan1 vulkan-tools mesa-utils mesa-utils-extra xvfb xauth \
    python3 python3-pil ca-certificates fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/reference /opt/reference
COPY --from=builder /opt/tools /opt/tools
COPY reference.py /opt/tools/bin/reference.py
COPY vulkan_lifetimes.py /opt/tools/bin/vulkan_lifetimes.py
ENV PATH=/opt/tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/reference/lib:/opt/tools/lib \
    VK_LAYER_PATH=/opt/tools/share/vulkan/explicit_layer.d \
    LIBGL_DRIVERS_PATH=/opt/reference/lib/dri \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=4 \
    VK_DRIVER_FILES=/opt/reference/share/vulkan/icd.d/lvp_icd.aarch64.json
WORKDIR /work
ENTRYPOINT ["python3", "/opt/tools/bin/reference.py"]

FROM reference AS applications
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ qt6-base-dev qt6-base-private-dev qt6-declarative-dev qml6-module-qtquick \
    qml6-module-qtquick-window qml6-module-qtqml-workerscript \
    openjdk-21-jre-headless unzip libasound2t64 libpulse0 libxcursor1 libxi6 \
    libxinerama1 libxkbcommon0 libflite1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/app-inputs /src/app-inputs
RUN cmake -S /src/app-inputs/qtquick -B /build/qt -G Ninja && cmake --build /build/qt && \
    mkdir -p /opt/apps && cp /build/qt/wink-qt-fixture /opt/apps/
RUN unzip /src/app-inputs/godot.zip -d /opt/apps && \
    mv /opt/apps/Godot* /opt/apps/godot && chmod +x /opt/apps/godot && \
    mkdir /opt/suite && cp /src/app-inputs/*.py /src/app-inputs/*.json /opt/suite/

FROM applications AS application-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang libc++-dev libc++abi-dev libegl1-mesa-dev libgles2-mesa-dev libsdl2-dev \
    libvulkan-dev libgl-dev libglew-dev libpng-dev libsnappy-dev libzstd-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
    libavdevice-dev libavfilter-dev libzip-dev \
    zlib1g-dev pkg-config python3-dev && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/extended-inputs /src/
RUN CC=clang CXX=clang++ cmake -S /src/filament -B /build/filament -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DFILAMENT_BUILD_FILAMAT=ON -DFILAMENT_BUILD_GLTFIO=ON \
    -DFILAMENT_SUPPORTS_VULKAN=ON -DFILAMENT_BUILD_JAVA=OFF -DFILAMENT_SUPPORTS_EGL_ON_LINUX=ON \
    -DFILAMENT_LINUX_IS_MOBILE=ON -DIS_HOST_PLATFORM=TRUE -DWINK_REFERENCE_GLES=ON && \
    cmake --build /build/filament --target gltf_viewer -j4 && \
    cp /build/filament/samples/gltf_viewer /opt/apps/gltf_viewer-gles && \
    mkdir -p /opt/apps/filament-assets /opt/apps/assets && \
    cp -r /build/filament/samples/assets/ibl /opt/apps/filament-assets/ && \
    cp -r /build/filament/samples/assets/fonts /opt/apps/assets/
RUN CC=clang CXX=clang++ cmake -S /src/filament -B /build/filament-vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DFILAMENT_BUILD_FILAMAT=ON -DFILAMENT_BUILD_GLTFIO=ON \
    -DFILAMENT_SUPPORTS_VULKAN=ON -DFILAMENT_BUILD_JAVA=OFF && \
    cmake --build /build/filament-vulkan --target gltf_viewer -j4 && \
    cp /build/filament-vulkan/samples/gltf_viewer /opt/apps/gltf_viewer-vulkan
RUN cmake -S /src/ppsspp -B /build/ppsspp -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DUSING_GLES2=ON -DUSING_EGL=OFF -DUSE_SYSTEM_FFMPEG=ON -DUNITTEST=OFF && \
    cmake --build /build/ppsspp --target PPSSPPSDL -j4 && \
    cp /build/ppsspp/PPSSPPSDL /opt/apps/ && cp -r /src/ppsspp/assets /opt/apps/
RUN cp -r /src/pspautotests/demos /opt/apps/psp-demos

FROM applications AS extended
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc++1 libc++abi1 libsdl2-2.0-0 libavcodec60 libavformat60 libavutil58 \
    libswresample4 libswscale7 libavdevice60 libavfilter9 libzip4t64 libsnappy1v5 libglew2.2 libegl1 libgles2 libgl1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=application-builder /opt/apps /opt/apps

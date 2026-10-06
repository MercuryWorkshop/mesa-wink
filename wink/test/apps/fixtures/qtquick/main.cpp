// SPDX-License-Identifier: MIT
#include <vulkan/vulkan.h>
#include <QGuiApplication>
#include <QQuickView>
#include <QQuickGraphicsDevice>
#include <QVulkanInstance>
#include <QSurfaceFormat>
#include <QTimer>
#include <cstdio>
#include <vector>
#include <QOpenGLContext>
#include <QSGRendererInterface>
// Pinned Qt 6.4 RHI adapter, applied before rendering starts. Access is
// deliberately confined to this recording fixture and recorded in its lock.
#define private public
#include <private/qrhi_p.h>
#include <private/qrhigles2_p_p.h>
#undef private

// Qt 6.4's internally created device enables every feature reported by the
// CPU reference. Import a device with the workload's explicit feature profile
// so replay does not request unused sparse or storage-image capabilities.
static VkDevice makeDevice(QVulkanInstance &instance, VkPhysicalDevice &physical,
                           uint32_t &family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance.vkInstance(), &count, nullptr) != VK_SUCCESS || !count)
        return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance.vkInstance(), &count, devices.data());
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        uint32_t queues = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> families(queues);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, families.data());
        for (uint32_t i = 0; i < queues; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkPhysicalDeviceFeatures available;
            vkGetPhysicalDeviceFeatures(candidate, &available);
            if (!available.occlusionQueryPrecise) return VK_NULL_HANDLE;
            const float priority = 1;
            const VkDeviceQueueCreateInfo queue = {
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, i, 1, &priority
            };
            const char *extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
            VkPhysicalDeviceFeatures features = {};
            features.occlusionQueryPrecise = VK_TRUE;
            const VkDeviceCreateInfo info = {
                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue,
                0, nullptr, 1, &extension, &features
            };
            VkDevice device = VK_NULL_HANDLE;
            if (vkCreateDevice(candidate, &info, nullptr, &device) == VK_SUCCESS) {
                physical = candidate; family = i;
                std::puts("WINK_FIXTURE qtquick Vulkan profile=precise-occlusion swapchain");
                return device;
            }
        }
    }
    return VK_NULL_HANDLE;
}
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    if (qEnvironmentVariable("QSG_RHI_BACKEND") == "opengl") {
        QSurfaceFormat format;
        format.setRenderableType(QSurfaceFormat::OpenGLES);
        format.setVersion(3, 0);
        QSurfaceFormat::setDefaultFormat(format);
    }
    QVulkanInstance instance;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = 0;
    if (qEnvironmentVariable("QSG_RHI_BACKEND") == "vulkan") {
        instance.setApiVersion(QVersionNumber(1, 1));
        if (!instance.create() || !(device = makeDevice(instance, physical, family))) return 1;
    }
    int result;
    {
    QQuickView view;
    if (qEnvironmentVariable("QSG_RHI_BACKEND") == "opengl") {
        QObject::connect(&view, &QQuickWindow::sceneGraphInitialized, &view, [&view] {
            auto *rhi = static_cast<QRhi *>(view.rendererInterface()->getResource(
                    &view, QSGRendererInterface::RhiResource));
            if (!rhi || rhi->backend() != QRhi::OpenGLES2) qFatal("Expected GLES RHI");
            auto *gles = static_cast<QRhiGles2 *>(rhi->d);
            if (!gles->ctx->isOpenGLES()) qFatal("Expected an ES context");
            // Qt probes this desktop entry point in ES too. GL_FILL is
            // already the ES rasterization mode; disable the illegal call
            // in the application's capability state before the first draw.
            gles->glPolygonMode = nullptr;
            std::puts("WINK_FIXTURE qtquick GLES procedure profile: glPolygonMode unavailable");
        }, Qt::DirectConnection);
    }
    if (device) {
        view.setVulkanInstance(&instance);
        view.setGraphicsDevice(QQuickGraphicsDevice::fromDeviceObjects(physical, device, family));
    }
    view.resize(640, 360);
    view.setSource(QUrl::fromLocalFile("/fixtures/qtquick/main.qml"));
    if (view.status() == QQuickView::Error) return 1;
    int frames = 0;
    QObject::connect(&view, &QQuickWindow::frameSwapped, &app, [&] {
        if (++frames == 64) {
            std::puts("WINK_FIXTURE qtquick complete");
            app.quit();
        } else view.update();
    });
    view.show();
    std::puts("WINK_FIXTURE qtquick initialized");
    result = app.exec();
    }
    if (device) {
        vkDeviceWaitIdle(device);
        vkDestroyDevice(device, nullptr);
    }
    return result;
}

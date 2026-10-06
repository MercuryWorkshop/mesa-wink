/* SPDX-License-Identifier: MIT
 * A portable headless-swapchain capture, with image-to-host readback. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#define VK(call) do { VkResult r=(call); if(r!=VK_SUCCESS){fprintf(stderr,"error: %s = %d\n",#call,r);return 1;} } while(0)
int main(void) {
    VkInstance instance;
    const char *ie[]={"VK_KHR_surface","VK_EXT_headless_surface"};
    VK(vkCreateInstance(&(VkInstanceCreateInfo){.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo=&(VkApplicationInfo){.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1},
        .enabledExtensionCount=2,.ppEnabledExtensionNames=ie},NULL,&instance));
    VkSurfaceKHR surface;
    PFN_vkCreateHeadlessSurfaceEXT create=(PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance,"vkCreateHeadlessSurfaceEXT");
    if(!create) return 2;
    VK(create(instance,&(VkHeadlessSurfaceCreateInfoEXT){.sType=VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT},NULL,&surface));
    uint32_t count=1; VkPhysicalDevice physical;
    VK(vkEnumeratePhysicalDevices(instance,&count,&physical));
    VkPhysicalDeviceProperties properties; vkGetPhysicalDeviceProperties(physical,&properties);
    printf("renderer: %s\n",properties.deviceName);
    float priority=1; const char *de[]={"VK_KHR_swapchain"}; VkDevice device;
    VK(vkCreateDevice(physical,&(VkDeviceCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&(VkDeviceQueueCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=0,.queueCount=1,.pQueuePriorities=&priority},.enabledExtensionCount=1,.ppEnabledExtensionNames=de},NULL,&device));
    VkQueue queue; vkGetDeviceQueue(device,0,0,&queue);
    VkSurfaceCapabilitiesKHR caps; VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical,surface,&caps));
    uint32_t nf=0; VK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical,surface,&nf,NULL));
    VkSurfaceFormatKHR *formats=calloc(nf,sizeof(*formats)); VK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical,surface,&nf,formats));
    VkSurfaceFormatKHR format=formats[0];
    for(uint32_t i=0;i<nf;i++) if(formats[i].format==VK_FORMAT_B8G8R8A8_UNORM){format=formats[i];break;}
    if(format.format!=VK_FORMAT_B8G8R8A8_UNORM && format.format!=VK_FORMAT_R8G8B8A8_UNORM) return 2;
    VkSwapchainKHR swapchain; uint32_t ni=caps.minImageCount>4?caps.minImageCount:4;
    if(caps.maxImageCount && ni>caps.maxImageCount) ni=caps.maxImageCount;
    VK(vkCreateSwapchainKHR(device,&(VkSwapchainCreateInfoKHR){.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface=surface,.minImageCount=ni,.imageFormat=format.format,.imageColorSpace=format.colorSpace,
        .imageExtent={640,360},.imageArrayLayers=1,.imageUsage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,.preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,.presentMode=VK_PRESENT_MODE_FIFO_KHR},NULL,&swapchain));
    VK(vkGetSwapchainImagesKHR(device,swapchain,&ni,NULL)); VkImage *images=calloc(ni,sizeof(*images));
    VK(vkGetSwapchainImagesKHR(device,swapchain,&ni,images));
    VkBuffer buffer; VK(vkCreateBuffer(device,&(VkBufferCreateInfo){.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=640*360*4,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT},NULL,&buffer));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,buffer,&req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical,&mp);
    uint32_t mt=0; while(mt<mp.memoryTypeCount && (!(req.memoryTypeBits&(1u<<mt)) ||
        (mp.memoryTypes[mt].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))!=
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) mt++;
    if(mt==mp.memoryTypeCount) return 2;
    VkDeviceMemory memory; VK(vkAllocateMemory(device,&(VkMemoryAllocateInfo){.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size,.memoryTypeIndex=mt},NULL,&memory)); VK(vkBindBufferMemory(device,buffer,memory,0));
    unsigned char *pixels; VK(vkMapMemory(device,memory,0,VK_WHOLE_SIZE,0,(void**)&pixels));
    VkCommandPool pool; VK(vkCreateCommandPool(device,&(VkCommandPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},NULL,&pool));
    VkCommandBuffer cmd; VK(vkAllocateCommandBuffers(device,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cmd));
    VkSemaphore acquired; VK(vkCreateSemaphore(device,&(VkSemaphoreCreateInfo){.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO},NULL,&acquired));
    for(int frame=1;frame<=16;frame++) {
        uint32_t index; VK(vkAcquireNextImageKHR(device,swapchain,UINT64_MAX,acquired,VK_NULL_HANDLE,&index));
        VK(vkResetCommandBuffer(cmd,0)); VK(vkBeginCommandBuffer(cmd,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
        VkImageMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.image=images[index],
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&barrier);
        VkClearColorValue color={.float32={frame/16.f,.25f,.75f,1}};
        vkCmdClearColorImage(cmd,images[index],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&color,1,&barrier.subresourceRange);
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&barrier);
        vkCmdCopyImageToBuffer(cmd,images[index],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&(VkBufferImageCopy){
            .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={640,360,1}});
        VkBufferMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask=VK_ACCESS_HOST_READ_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.buffer=buffer,.size=VK_WHOLE_SIZE};
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&host,0,NULL);
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT; barrier.dstAccessMask=0;
        barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; barrier.newLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,NULL,0,NULL,1,&barrier);
        VK(vkEndCommandBuffer(cmd)); VkPipelineStageFlags stage=VK_PIPELINE_STAGE_TRANSFER_BIT;
        VK(vkQueueSubmit(queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.waitSemaphoreCount=1,
            .pWaitSemaphores=&acquired,.pWaitDstStageMask=&stage,.commandBufferCount=1,.pCommandBuffers=&cmd},VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(queue));
        if(frame==1||frame==4||frame==8||frame==16) {
            char name[64]; snprintf(name,sizeof(name),"live-%d.ppm",frame); FILE *out=fopen(name,"wb"); if(!out)return 3;
            fprintf(out,"P6\n640 360\n255\n");
            int bgra=format.format==VK_FORMAT_B8G8R8A8_UNORM;
            for(int p=0;p<640*360;p++){unsigned char rgb[]={pixels[p*4+(bgra?2:0)],pixels[p*4+1],pixels[p*4+(bgra?0:2)]};fwrite(rgb,1,3,out);}
            fclose(out);
        }
        VK(vkQueuePresentKHR(queue,&(VkPresentInfoKHR){.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount=1,.pSwapchains=&swapchain,.pImageIndices=&index})); VK(vkQueueWaitIdle(queue));
    }
    vkUnmapMemory(device,memory); vkDestroySemaphore(device,acquired,NULL); vkDestroyCommandPool(device,pool,NULL);
    vkDestroyBuffer(device,buffer,NULL); vkFreeMemory(device,memory,NULL); vkDestroySwapchainKHR(device,swapchain,NULL);
    vkDestroyDevice(device,NULL); vkDestroySurfaceKHR(instance,surface,NULL); vkDestroyInstance(instance,NULL);
    free(images); free(formats); puts("WINK_FIXTURE Vulkan probe complete"); return 0;
}

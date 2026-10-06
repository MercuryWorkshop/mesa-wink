#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain framebuffer attachment owners in the pinned Filament cache."""
from pathlib import Path
import sys


def patch(root):
    vulkan = Path(root) / "filament/backend/src/vulkan"
    edits = {
        "VulkanFboCache.h": [
            ('#include "VulkanContext.h"', '#include "VulkanContext.h"\n#include "vulkan/memory/ResourcePointer.h"'),
            ('    struct FboVal {\n        VkFramebuffer handle;\n        uint32_t timestamp;\n    };',
             '    struct FboVal {\n        VkFramebuffer handle;\n        uint32_t timestamp;\n'
             '        // Cached image views must outlive every framebuffer use.\n'
             '        fvkmemory::resource_ptr<fvkmemory::Resource> owner;\n    };'),
            ('VkFramebuffer getFramebuffer(FboKey const& config) noexcept;',
             'VkFramebuffer getFramebuffer(FboKey const& config,\n'
             '            fvkmemory::resource_ptr<fvkmemory::Resource> const& owner) noexcept;'),
        ],
        "VulkanFboCache.cpp": [
            ('VkFramebuffer VulkanFboCache::getFramebuffer(FboKey const& config) noexcept {',
             'VkFramebuffer VulkanFboCache::getFramebuffer(FboKey const& config,\n'
             '        fvkmemory::resource_ptr<fvkmemory::Resource> const& owner) noexcept {'),
            ('mFramebufferCache[config] = {framebuffer, mCurrentTime};',
             'mFramebufferCache[config] = {framebuffer, mCurrentTime, owner};'),
            ('            iter.value().handle = VK_NULL_HANDLE;\n        }\n    }\n    for (auto iter = mRenderPassCache.begin();',
             '            iter.value().handle = VK_NULL_HANDLE;\n'
             '            iter.value().owner = {};\n        }\n    }\n    for (auto iter = mRenderPassCache.begin();'),
        ],
        "VulkanDriver.cpp": [
            ('mFramebufferCache.getFramebuffer(fbkey);', 'mFramebufferCache.getFramebuffer(fbkey, rt);'),
        ],
    }
    for name, replacements in edits.items():
        path = vulkan / name
        text = path.read_text()
        for before, after in replacements:
            if after in text:
                continue
            if text.count(before) != 1:
                raise RuntimeError("Pinned Filament source changed: " + name)
            text = text.replace(before, after)
        path.write_text(text)


if __name__ == "__main__":
    patch(sys.argv[1])

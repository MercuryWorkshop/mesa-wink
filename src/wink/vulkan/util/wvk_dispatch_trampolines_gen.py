# coding=utf-8
# Copyright 2020 Intel Corporation
# Copyright 2025 LunarG, Inc
# SPDX-License-Identifier: MIT
#
# From kosmickrisp's kk_dispatch_trampolines_gen.py.
#
# wink records every vkCmd* into the command buffer's vk_cmd_queue and
# replays it at submit through the driver's real entrypoints, because a
# WebGPU command buffer can be submitted only once. So there are two device
# dispatch tables: vk_device::dispatch_table holds the driver's entrypoints
# -- what replay and the runtime's own helpers (render passes, secondary
# command buffers) call -- and wvk_device::exposed_dispatch_table the ones
# an application reaches, with every vkCmd* recording. The loader's
# trampolines must land on the latter, which Mesa's generic ones do not.

import argparse
import os
import sys

from mako.template import Template

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'vulkan', 'util'))
from vk_entrypoints import get_entrypoints_from_xml  # noqa: E402

COPYRIGHT = """\
/*
 * Copyright 2020 Intel Corporation
 * Copyright 2025 LunarG, Inc
 * SPDX-License-Identifier: MIT
 */
"""

TEMPLATE_H = Template(COPYRIGHT + """\
/* This file generated from ${filename}, don't edit directly. */

#ifndef WVK_DISPATCH_TRAMPOLINES_H
#define WVK_DISPATCH_TRAMPOLINES_H

#include "vk_dispatch_table.h"

#ifdef __cplusplus
extern "C" {
#endif

extern struct vk_physical_device_dispatch_table wvk_physical_device_trampolines;
extern struct vk_device_dispatch_table wvk_device_trampolines;

#ifdef __cplusplus
}
#endif

#endif /* WVK_DISPATCH_TRAMPOLINES_H */
""")

TEMPLATE_C = Template(COPYRIGHT + """\
/* This file generated from ${filename}, don't edit directly. */

#include "wvk_device.h"
#include "wvk_dispatch_trampolines.h"
#include "vk_object.h"
#include "vk_physical_device.h"

% for e in entrypoints:
  % if not e.is_physical_device_entrypoint() or e.alias:
    <% continue %>
  % endif
  % if e.guard is not None:
#ifdef ${e.guard}
  % endif
static VKAPI_ATTR ${e.return_type} VKAPI_CALL
${e.prefixed_name('wvk_tramp')}(${e.decl_params()})
{
    <% assert e.params[0].type == 'VkPhysicalDevice' %>
    VK_FROM_HANDLE(vk_physical_device, vk_physical_device, ${e.params[0].name});
  % if e.return_type == 'void':
    vk_physical_device->dispatch_table.${e.name}(${e.call_params()});
  % else:
    return vk_physical_device->dispatch_table.${e.name}(${e.call_params()});
  % endif
}
  % if e.guard is not None:
#endif
  % endif
% endfor

struct vk_physical_device_dispatch_table wvk_physical_device_trampolines = {
% for e in entrypoints:
  % if not e.is_physical_device_entrypoint() or e.alias:
    <% continue %>
  % endif
  % if e.guard is not None:
#ifdef ${e.guard}
  % endif
    .${e.name} = ${e.prefixed_name('wvk_tramp')},
  % if e.guard is not None:
#endif
  % endif
% endfor
};

% for e in entrypoints:
  % if not e.is_device_entrypoint() or e.alias:
    <% continue %>
  % endif
  % if e.guard is not None:
#ifdef ${e.guard}
  % endif
static VKAPI_ATTR ${e.return_type} VKAPI_CALL
${e.prefixed_name('wvk_tramp')}(${e.decl_params()})
{
  % if e.params[0].type == 'VkDevice':
    VK_FROM_HANDLE(wvk_device, wvk_device, ${e.params[0].name});
    % if e.return_type == 'void':
    wvk_device->exposed_dispatch_table.${e.name}(${e.call_params()});
    % else:
    return wvk_device->exposed_dispatch_table.${e.name}(${e.call_params()});
    % endif
  % elif e.params[0].type in ('VkCommandBuffer', 'VkQueue'):
    struct vk_object_base *vk_object = (struct vk_object_base *)${e.params[0].name};
    struct wvk_device *wvk_device = container_of(vk_object->device, struct wvk_device, vk);
    % if e.return_type == 'void':
    wvk_device->exposed_dispatch_table.${e.name}(${e.call_params()});
    % else:
    return wvk_device->exposed_dispatch_table.${e.name}(${e.call_params()});
    % endif
  % else:
    assert(!"Unhandled device child trampoline case: ${e.params[0].type}");
  % endif
}
  % if e.guard is not None:
#endif
  % endif
% endfor

struct vk_device_dispatch_table wvk_device_trampolines = {
% for e in entrypoints:
  % if not e.is_device_entrypoint() or e.alias:
    <% continue %>
  % endif
  % if e.guard is not None:
#ifdef ${e.guard}
  % endif
    .${e.name} = ${e.prefixed_name('wvk_tramp')},
  % if e.guard is not None:
#endif
  % endif
% endfor
};
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-c', help='Output C file.')
    parser.add_argument('--out-h', help='Output H file.')
    parser.add_argument('--beta', required=True, help='Enable beta extensions.')
    parser.add_argument('--xml', help='Vulkan API XML file.', required=True,
                        action='append', dest='xml_files')
    args = parser.parse_args()

    entrypoints = get_entrypoints_from_xml(args.xml_files, args.beta)
    filename = os.path.basename(__file__)
    if args.out_h:
        with open(args.out_h, 'w', encoding='utf-8') as f:
            f.write(TEMPLATE_H.render(entrypoints=entrypoints, filename=filename))
    if args.out_c:
        with open(args.out_c, 'w', encoding='utf-8') as f:
            f.write(TEMPLATE_C.render(entrypoints=entrypoints, filename=filename))


if __name__ == '__main__':
    main()

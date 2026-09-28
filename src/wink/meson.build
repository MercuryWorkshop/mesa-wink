# SPDX-License-Identifier: MIT

# Both drivers need the shader compiler, and gallium is configured after
# this directory.
if get_option('gex-emdawn-dir') != ''
  subdir('compiler')
endif
if with_wink_vk
  subdir('vulkan')
endif

// SPDX-License-Identifier: MIT
#include "wink.h"

void *_getPublicProcAddress(const char *name) { return wink_get_proc_address(name); }
void *_getPrivateProcAddress(const char *name) { return wink_get_proc_address(name); }

/* SPDX-License-Identifier: MIT
 *
 * The native test's main, as an export the page enters through
 * WebAssembly.promising: every wait of the driver suspends under it. */
#include <emscripten.h>

int wvk_test_main(void);

EMSCRIPTEN_KEEPALIVE int
wvk_browser_main(void)
{
   return wvk_test_main();
}

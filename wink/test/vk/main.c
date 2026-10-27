/* SPDX-License-Identifier: MIT
 *
 * The native test's main, as an export the page enters through
 * WebAssembly.promising: every wait of the driver suspends under it. */
#include <emscripten.h>
#include "wink.h"
#ifdef __EMSCRIPTEN_PTHREADS__
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <emscripten/threading.h>
#endif

int wvk_test_main(void);

EMSCRIPTEN_KEEPALIVE int
wvk_browser_main(void)
{
   return wvk_test_main();
}

#ifdef __EMSCRIPTEN_PTHREADS__
static void *worker(void *unused)
{
   wink_init_attribs a;
   wink_init_attribs_default(&a);
   a.flags = WINK_INIT_GL_ON_MAIN_THREAD;
   int rc = wink_init(&a);
   fprintf(stderr, "WINK_VK_WORKER\t%d\n", !emscripten_is_main_browser_thread());
   if (rc == WINK_OK)
      setenv("WINK_VK_WORKER_TEST", "1", 1);
   if (rc == WINK_OK)
      rc = wvk_test_main();
   MAIN_THREAD_ASYNC_EM_ASM({ Module['wvkTestDone']($0); }, rc);
   return NULL;
}

EMSCRIPTEN_KEEPALIVE int wvk_browser_start(void)
{
   pthread_t thread;
   int rc = pthread_create(&thread, NULL, worker, NULL);
   if (!rc)
      pthread_detach(thread);
   return rc;
}
#endif

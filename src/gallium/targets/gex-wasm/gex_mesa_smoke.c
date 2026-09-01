/* End-to-end wasm bootstrap for the gex Mesa frontend.
 *
 * This deliberately uses a user framebuffer so no window-system frontend is
 * involved.  Public GL dispatch still goes through st/mesa and softpipe.
 */
#include <GL/gl.h>

#include "frontend/api.h"
#include "frontend/sw_winsys.h"
#include "mesa/main/menums.h"
#include "mesa/state_tracker/st_context.h"
#include "pipe/p_state.h"
#include "softpipe/sp_public.h"
#include "sw/null/null_sw_winsys.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int
get_frontend_param(struct pipe_frontend_screen *fscreen,
                   enum st_manager_param param)
{
   (void)fscreen;
   (void)param;
   return 0;
}

static int
run_smoke(void)
{
   struct sw_winsys *winsys = null_sw_create();
   if (!winsys) {
      fprintf(stderr, "FAIL winsys\n");
      return 1;
   }

   struct pipe_frontend_screen fscreen = {0};
   fscreen.screen = softpipe_create_screen(winsys);
   fscreen.get_param = get_frontend_param;
   if (!fscreen.screen) {
      fprintf(stderr, "FAIL screen\n");
      winsys->destroy(winsys);
      return 2;
   }

   struct st_context_attribs attribs;
   memset(&attribs, 0, sizeof(attribs));
   attribs.profile = API_OPENGL_CORE;
   attribs.major = 3;
   attribs.minor = 3;
   attribs.visual.color_format = PIPE_FORMAT_NONE;
   attribs.visual.depth_stencil_format = PIPE_FORMAT_NONE;
   attribs.visual.accum_format = PIPE_FORMAT_NONE;

   enum st_context_error error = ST_CONTEXT_SUCCESS;
   struct st_context *st = st_api_create_context(&fscreen, &attribs, &error, NULL);
   if (!st || !st_api_make_current(st, NULL, NULL)) {
      fprintf(stderr, "FAIL context %d\n", error);
      if (st)
         st_destroy_context(st);
      st_screen_destroy(&fscreen);
      fscreen.screen->destroy(fscreen.screen);
      winsys->destroy(winsys);
      return 3;
   }

   const char *version = (const char *)glGetString(GL_VERSION);
   if (!version) {
      fprintf(stderr, "FAIL version\n");
      return 4;
   }

   GLuint texture = 0;
   GLuint framebuffer = 0;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D, texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FAIL framebuffer\n");
      return 5;
   }

   glViewport(0, 0, 8, 8);
   glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   uint8_t pixel[4] = {0};
   glReadPixels(3, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   const uint8_t expected[4] = {64, 128, 191, 255};
   if (memcmp(pixel, expected, sizeof(pixel)) != 0) {
      fprintf(stderr, "FAIL pixel %u %u %u %u\n",
              pixel[0], pixel[1], pixel[2], pixel[3]);
      return 6;
   }
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "FAIL gl-error\n");
      return 7;
   }

   printf("PASS gex-mesa-wasm version=%s pixel=%u,%u,%u,%u\n",
          version, pixel[0], pixel[1], pixel[2], pixel[3]);

   glDeleteFramebuffers(1, &framebuffer);
   glDeleteTextures(1, &texture);
   st_api_make_current(NULL, NULL, NULL);
   st_destroy_context(st);
   st_screen_destroy(&fscreen);
   fscreen.screen->destroy(fscreen.screen);
   winsys->destroy(winsys);
   return 0;
}

int
main(void)
{
   return run_smoke();
}

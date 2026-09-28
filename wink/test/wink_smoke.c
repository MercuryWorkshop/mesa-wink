/* SPDX-License-Identifier: MIT */
/*
 * End-to-end check of a packaged wink, linked the way a consumer links it:
 * plain emcc 3.1.56, the three archives and the --js-library files from
 * manifest.json. For each API (GL ES, core, compatibility) at the highest
 * version there is, it clears the default framebuffer and reads it back --
 * the read is a JSPI suspend, so this also proves the promising/Suspending
 * pairing -- draws a triangle with that API's shading language, swaps,
 * resizes and reads again.
 *
 * Run by wink/test/run-browser.mjs; the result lands in
 * globalThis.__winkResult.
 */
#include <GL/gl.h>
#include <GL/glext.h>
#include <emscripten/emscripten.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wink.h"

#define W 64
#define H 48

static int failures;

static void
note(const char *fmt, ...)
{
   char line[512];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(line, sizeof(line), fmt, ap);
   va_end(ap);
   printf("%s\n", line);
}

#define CHECK(cond, ...) do {                                          \
      if (!(cond)) { ++failures; note("FAIL " __VA_ARGS__); }          \
   } while (0)

/* GL through wink_get_proc_address only: wink exports no gl* symbol. The
 * pointers are p_gl* because GL/gl.h already declares the gl* prototypes. */
#define GL_FUNCS(X)                                                              \
   X(const GLubyte *, GetString, (GLenum))                                       \
   X(GLenum, GetError, (void))                                                   \
   X(void, GetIntegerv, (GLenum, GLint *))                                       \
   X(void, Viewport, (GLint, GLint, GLsizei, GLsizei))                           \
   X(void, ClearColor, (GLfloat, GLfloat, GLfloat, GLfloat))                     \
   X(void, Clear, (GLbitfield))                                                  \
   X(void, ReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *)) \
   X(GLuint, CreateShader, (GLenum))                                             \
   X(void, ShaderSource, (GLuint, GLsizei, const GLchar *const *, const GLint *)) \
   X(void, CompileShader, (GLuint))                                              \
   X(void, GetShaderiv, (GLuint, GLenum, GLint *))                               \
   X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *))             \
   X(GLuint, CreateProgram, (void))                                              \
   X(void, AttachShader, (GLuint, GLuint))                                       \
   X(void, LinkProgram, (GLuint))                                                \
   X(void, GetProgramiv, (GLuint, GLenum, GLint *))                              \
   X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *))            \
   X(void, UseProgram, (GLuint))                                                 \
   X(void, DeleteProgram, (GLuint))                                              \
   X(void, DeleteShader, (GLuint))                                               \
   X(void, GenVertexArrays, (GLsizei, GLuint *))                                 \
   X(void, BindVertexArray, (GLuint))                                            \
   X(void, DeleteVertexArrays, (GLsizei, const GLuint *))                        \
   X(void, DrawArrays, (GLenum, GLint, GLsizei))                                 \
   X(void, GenTextures, (GLsizei, GLuint *))                                     \
   X(void, BindTexture, (GLenum, GLuint))                                        \
   X(void, TexParameteri, (GLenum, GLenum, GLint))                               \
   X(void, CompressedTexImage2D, (GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *)) \
   X(void, DeleteTextures, (GLsizei, const GLuint *))

#define DECLARE(ret, name, args) static ret (*p_gl##name) args;
GL_FUNCS(DECLARE)
#undef DECLARE

static int
load_gl(void)
{
   int missing = 0;
#define LOAD(ret, name, args)                                           \
   p_gl##name = (ret (*) args)wink_get_proc_address("gl" #name);         \
   if (!p_gl##name) { note("FAIL no gl" #name); ++missing; }
   GL_FUNCS(LOAD)
#undef LOAD
   return missing;
}

static GLuint
compile(GLenum type, const char *src)
{
   GLuint s = p_glCreateShader(type);
   p_glShaderSource(s, 1, &src, NULL);
   p_glCompileShader(s);
   GLint ok = 0;
   p_glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024] = "";
      p_glGetShaderInfoLog(s, sizeof(log), NULL, log);
      note("FAIL shader: %s", log);
      ++failures;
   }
   return s;
}

static void
read_center(uint8_t px[4], int w, int h)
{
   memset(px, 0, 4);
   p_glReadPixels(w / 2, h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

static int
near(const uint8_t px[4], int r, int g, int b)
{
   return abs(px[0] - r) <= 2 && abs(px[1] - g) <= 2 && abs(px[2] - b) <= 2;
}

static GLuint
link_program(const char *vs_src, const char *fs_src, const char *label)
{
   GLuint v = compile(GL_VERTEX_SHADER, vs_src), f = compile(GL_FRAGMENT_SHADER, fs_src);
   GLuint prog = p_glCreateProgram();
   p_glAttachShader(prog, v);
   p_glAttachShader(prog, f);
   p_glLinkProgram(prog);
   p_glDeleteShader(v);
   p_glDeleteShader(f);
   GLint linked = 0;
   p_glGetProgramiv(prog, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = "";
      p_glGetProgramInfoLog(prog, sizeof(log), NULL, log);
      note("FAIL %s: link: %s", label, log);
      ++failures;
   }
   return prog;
}

/* One compressed block, uploaded and sampled by a full-screen draw. The
 * expected colour is what the block decodes to, so this checks the format
 * reached the GPU (or st/mesa's CPU decode) intact, not merely that the
 * upload was accepted. */
static void
sample_block(const char *label, GLenum internal_format, GLsizei width, GLsizei height,
             const void *data, GLsizei size, int r, int g, int b)
{
   static const char *vs =
      "#version 300 es\n"
      "void main() {\n"
      "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
      "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
      "}\n";
   static const char *fs =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform highp sampler2D tex;\n"
      "out vec4 color;\n"
      "void main() { color = texture(tex, vec2(0.5)); }\n";
   GLuint tex = 0;
   p_glGenTextures(1, &tex);
   p_glBindTexture(GL_TEXTURE_2D, tex);
   p_glCompressedTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, size, data);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s: upload error 0x%x", label, err);

   GLuint prog = link_program(vs, fs, label);
   GLuint vao = 0;
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glUseProgram(prog);
   p_glViewport(0, 0, W, H);
   p_glClearColor(0, 0, 0, 1);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArrays(GL_TRIANGLES, 0, 3);
   uint8_t px[4];
   read_center(px, W, H);
   CHECK(near(px, r, g, b), "%s: sampled %u,%u,%u,%u, want %d,%d,%d", label,
         px[0], px[1], px[2], px[3], r, g, b);
   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glDeleteTextures(1, &tex);
}

static void
compressed(const char *label, wink_surface *surface)
{
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = WINK_API_OPENGL_ES;
   ca.major = 3;
   ca.minor = 2;
   int status = 0;
   wink_context *ctx = wink_context_create(&ca, &status);
   CHECK(ctx, "%s: ES 3.2 context: %s", label, wink_status_string(status));
   if (!ctx)
      return;
   wink_make_current(ctx, surface, surface);

   /* ASTC 4x4 void-extent block: one constant colour, as UNORM16 RGBA. */
   static const uint8_t astc[16] = {
      0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0x80, 0x80, 0x00, 0x00, 0xff, 0xff,
   };
   char name[64];
   snprintf(name, sizeof(name), "%s astc", label);
   sample_block(name, GL_COMPRESSED_RGBA_ASTC_4x4_KHR, 4, 4, astc, sizeof(astc), 255, 128, 0);

   /* A block size that is not a power of two, on a texture that is not a
    * multiple of it: 7x5 in 6x6 blocks is 2x1 blocks, stored as 12x6. A
    * void-extent block reads the same whatever the block size. */
   uint8_t astc6[32];
   memcpy(astc6, astc, 16);
   memcpy(astc6 + 16, astc, 16);
   snprintf(name, sizeof(name), "%s astc 6x6 on 7x5", label);
   sample_block(name, GL_COMPRESSED_RGBA_ASTC_6x6_KHR, 7, 5, astc6, sizeof(astc6), 255, 128, 0);

   /* ETC2 RGB8 in individual mode: both halves 4-bit (F,8,0), table 0, every
    * pixel index 0 -- which adds 2 to each channel. */
   static const uint8_t etc2[8] = { 0xff, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
   snprintf(name, sizeof(name), "%s etc2", label);
   sample_block(name, GL_COMPRESSED_RGB8_ETC2, 4, 4, etc2, sizeof(etc2), 255, 138, 2);

   wink_make_current(NULL, NULL, NULL);
   wink_context_destroy(ctx);
}

static void
exercise(const char *label, int api, wink_surface *surface)
{
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = api;
   int status = 0;
   wink_context *ctx = wink_context_create(&ca, &status);
   CHECK(ctx, "%s: context: %s", label, wink_status_string(status));
   if (!ctx)
      return;
   int got_api, major, minor;
   wink_context_get_version(ctx, &got_api, &major, &minor);
   status = wink_make_current(ctx, surface, surface);
   CHECK(status == WINK_OK, "%s: make current: %s", label, wink_status_string(status));

   const char *version = (const char *)p_glGetString(GL_VERSION);
   const char *renderer = (const char *)p_glGetString(GL_RENDERER);
   note("%s: %d.%d -- %s -- %s", label, major, minor, version ? version : "?", renderer ? renderer : "?");

   /* Clear and read back: the read suspends until the GPU is done. */
   p_glViewport(0, 0, W, H);
   p_glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   p_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   uint8_t px[4];
   read_center(px, W, H);
   CHECK(near(px, 64, 128, 191), "%s: clear read %u,%u,%u,%u", label, px[0], px[1], px[2], px[3]);

   /* A full-screen triangle in this API's shading language. */
   const int es = api == WINK_API_OPENGL_ES;
   const char *header = es ? (major * 10 + minor >= 32 ? "#version 320 es\nprecision highp float;\n"
                                                        : "#version 300 es\nprecision highp float;\n")
                           : "#version 330 core\n";
   char vs[512], fs[512];
   snprintf(vs, sizeof(vs), "%s"
            "void main() {\n"
            "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
            "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
            "}\n", header);
   snprintf(fs, sizeof(fs), "%s"
            "out vec4 color;\n"
            "void main() { color = vec4(1.0, 0.5, 0.0, 1.0); }\n", header);
   GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
   GLuint prog = p_glCreateProgram();
   p_glAttachShader(prog, v);
   p_glAttachShader(prog, f);
   p_glLinkProgram(prog);
   GLint linked = 0;
   p_glGetProgramiv(prog, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = "";
      p_glGetProgramInfoLog(prog, sizeof(log), NULL, log);
      note("FAIL %s: link: %s", label, log);
      ++failures;
   }
   GLuint vao = 0;
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glUseProgram(prog);
   p_glDrawArrays(GL_TRIANGLES, 0, 3);
   read_center(px, W, H);
   CHECK(near(px, 255, 128, 0), "%s: draw read %u,%u,%u,%u", label, px[0], px[1], px[2], px[3]);

   status = wink_swap_buffers(surface);
   CHECK(status == WINK_OK, "%s: swap: %s", label, wink_status_string(status));

   /* Resize: the default framebuffer is rebuilt at the new size. */
   wink_surface_resize(surface, W / 2, H / 2);
   GLint vp[4] = {0};
   p_glViewport(0, 0, W / 2, H / 2);
   p_glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
   p_glClear(GL_COLOR_BUFFER_BIT);
   read_center(px, W / 2, H / 2);
   CHECK(near(px, 0, 255, 0), "%s: resized read %u,%u,%u,%u", label, px[0], px[1], px[2], px[3]);
   p_glGetIntegerv(GL_VIEWPORT, vp);
   wink_swap_buffers(surface);
   wink_surface_resize(surface, W, H);

   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s: GL error 0x%x", label, err);

   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glDeleteShader(v);
   p_glDeleteShader(f);
   wink_make_current(NULL, NULL, NULL);
   wink_context_destroy(ctx);
}

#ifndef __EMSCRIPTEN_PTHREADS__
/* Linked with -Wl,--wrap=pthread_create in the st build: Mesa must never
 * try to start a thread there, and a stub that fails quietly would hide it. */
int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int
__wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg)
{
   (void)t; (void)a; (void)fn; (void)arg;
   ++failures;
   note("FAIL a thread was started in the single-threaded build");
   return 11; /* EAGAIN */
}
#endif

/* library_wink_test.js: linking this in is what publishes the raw export. */
void wink_test_publish(void);

static int
run_pass(const char *label)
{
   int status = wink_init(NULL);
   if (status != WINK_OK) {
      note("FAIL %s: init: %s", label, wink_status_string(status));
      return 1;
   }
   note("-- %s", label);
   CHECK(load_gl() == 0, "GL lookup");
   CHECK(wink_get_proc_address("glPolygonOffsetClampEXT"), "no glPolygonOffsetClampEXT");
   CHECK(wink_get_proc_address("glBlendBarrierKHR"), "no glBlendBarrierKHR");
   CHECK(!wink_get_proc_address("eglGetDisplay"), "eglGetDisplay resolved");

   int core = 0, compat = 0, es = 0;
   wink_query_versions(&core, &compat, &es);
   note("versions: core %d, compat %d, es %d", core, compat, es);
   CHECK(core >= 33 && es >= 32, "versions too low");

   wink_surface_attribs sa;
   wink_surface_attribs_default(&sa);
   sa.canvas = "#canvas";
   sa.width = W;
   sa.height = H;
   sa.manage_canvas_size = 1;
   wink_surface *surface = wink_surface_create(&sa, &status);
   CHECK(surface, "surface: %s", wink_status_string(status));
   if (!surface)
      return 1;

   exercise("es", WINK_API_OPENGL_ES, surface);
   exercise("core", WINK_API_OPENGL_CORE, surface);
   exercise("compat", WINK_API_OPENGL_COMPAT, surface);
   compressed(label, surface);

   /* A version that does not exist is refused, not quietly lowered. */
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = WINK_API_OPENGL_CORE;
   ca.major = 9;
   wink_context *none = wink_context_create(&ca, &status);
   CHECK(!none && status == WINK_ERROR_BAD_VERSION, "core 9.0 was not refused");

   wink_surface_destroy(surface);
   wink_shutdown();
   return 0;
}

EMSCRIPTEN_KEEPALIVE int
wink_smoke_main(void)
{
   wink_test_publish();
   run_pass("native compression");
   /* And again from a fresh device with ASTC and ETC2 hidden from the
    * driver: st/mesa must still expose GL ES 3.2, decoding on the CPU. */
   wink_setenv("GEX_WGPU_NO_ASTC", "1");
   wink_setenv("GEX_WGPU_NO_ETC2", "1");
   run_pass("cpu-decoded compression");
   note(failures ? "FAILED %d" : "PASS", failures);
   return failures;
}

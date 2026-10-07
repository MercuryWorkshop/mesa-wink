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
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/stack.h>
#include <emscripten/threading.h>
#endif
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wink.h"

#define W 64
#define H 48

static int failures;
/* wink_init's flags for this run: 0, or WINK_INIT_GL_ON_MAIN_THREAD. */
static uint32_t init_flags;

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
   X(const GLubyte *, GetStringi, (GLenum, GLuint))                               \
   X(GLenum, GetError, (void))                                                   \
   X(void, GetIntegerv, (GLenum, GLint *))                                       \
   X(void, Viewport, (GLint, GLint, GLsizei, GLsizei))                           \
   X(void, ClearColor, (GLfloat, GLfloat, GLfloat, GLfloat))                     \
   X(void, Clear, (GLbitfield))                                                  \
   X(void, ClampColor, (GLenum, GLenum))                                         \
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
   X(void, GenBuffers, (GLsizei, GLuint *))                                     \
   X(void, BindBuffer, (GLenum, GLuint))                                        \
   X(void, BufferData, (GLenum, GLsizeiptr, const void *, GLenum))               \
   X(void, BufferStorage, (GLenum, GLsizeiptr, const void *, GLbitfield))        \
   X(void *, MapBufferRange, (GLenum, GLintptr, GLsizeiptr, GLbitfield))        \
   X(GLboolean, UnmapBuffer, (GLenum))                                         \
   X(void, Finish, (void))                                                     \
   X(void, DeleteBuffers, (GLsizei, const GLuint *))                            \
   X(void, GetBufferSubData, (GLenum, GLintptr, GLsizeiptr, void *))             \
   X(void, CopyBufferSubData, (GLenum, GLenum, GLintptr, GLintptr, GLsizeiptr))  \
   X(void, BindBufferBase, (GLenum, GLuint, GLuint))                            \
   X(void, DispatchCompute, (GLuint, GLuint, GLuint))                          \
   X(void, MemoryBarrier, (GLbitfield))                                        \
   X(void, EnableVertexAttribArray, (GLuint))                                   \
   X(void, VertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void *)) \
   X(void, VertexAttribIPointer, (GLuint, GLint, GLenum, GLsizei, const void *)) \
   X(void, DrawArrays, (GLenum, GLint, GLsizei))                                 \
   X(void, DrawArraysInstanced, (GLenum, GLint, GLsizei, GLsizei))               \
   X(void, DrawElementsInstanced, (GLenum, GLsizei, GLenum, const void *, GLsizei)) \
   X(void, Enable, (GLenum))                                                   \
   X(void, Disable, (GLenum))                                                  \
   X(void, GenTextures, (GLsizei, GLuint *))                                     \
   X(void, BindTexture, (GLenum, GLuint))                                        \
   X(void, TexParameteri, (GLenum, GLenum, GLint))                               \
   X(void, TexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *)) \
   X(void, GenFramebuffers, (GLsizei, GLuint *))                                \
   X(void, BindFramebuffer, (GLenum, GLuint))                                   \
   X(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))       \
   X(GLenum, CheckFramebufferStatus, (GLenum))                                 \
   X(void, DeleteFramebuffers, (GLsizei, const GLuint *))                       \
   X(void, DrawBuffers, (GLsizei, const GLenum *))                              \
   X(void, ReadBuffer, (GLenum))                                                \
   X(void, TexStorage2DMultisample, (GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLboolean)) \
   X(void, BlitFramebuffer, (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)) \
   X(void, CompressedTexImage2D, (GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *)) \
   X(void, DeleteTextures, (GLsizei, const GLuint *))                             \
   X(void, BufferSubData, (GLenum, GLintptr, GLsizeiptr, const void *))          \
   X(void, VertexAttribDivisor, (GLuint, GLuint))                                \
   X(void, MultiDrawArrays, (GLenum, const GLint *, const GLsizei *, GLsizei))   \
   X(void, MultiDrawArraysIndirect, (GLenum, const void *, GLsizei, GLsizei))  \
   X(void, EnableClientState, (GLenum))                                          \
   X(void, DisableClientState, (GLenum))                                         \
   X(void, VertexPointer, (GLint, GLenum, GLsizei, const void *))                \
   X(void, ColorPointer, (GLint, GLenum, GLsizei, const void *))

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

/* The pixel at the middle of each third of the screen, left to right. */
static void
check_thirds(const char *label, const int want[3][3])
{
   for (int t = 0; t < 3; ++t) {
      uint8_t px[4] = {0};
      p_glReadPixels(W * (2 * t + 1) / 6, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
      CHECK(near(px, want[t][0], want[t][1], want[t][2]),
            "%s: third %d read %u,%u,%u, want %d,%d,%d", label, t, px[0], px[1], px[2],
            want[t][0], want[t][1], want[t][2]);
   }
}

/* Sodium's terrain path: GL_QUADS through glMultiDrawArraysIndirect, each
 * command taking its tint from an instance-stepped attribute through its
 * baseInstance, with an empty command among them; then glMultiDrawArrays of
 * quads. gex_wgpu draws both against its quad index buffer and records every
 * command after the first as a repeat of it (GEX_WGPU_REPEAT), so a wrong base
 * vertex or base instance shows as a wrong colour in some third. */
static void
multidraw_quads(const char *label, wink_surface *surface)
{
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = WINK_API_OPENGL_COMPAT;
   int status = 0;
   wink_context *ctx = wink_context_create(&ca, &status);
   CHECK(ctx, "%s: compat context: %s", label, wink_status_string(status));
   if (!ctx)
      return;
   int api, major, minor;
   wink_context_get_version(ctx, &api, &major, &minor);
   wink_make_current(ctx, surface, surface);
   char name[96];
   if (major * 10 + minor < 43) {
      note("%s: multidraw quads skipped (compat %d.%d)", label, major, minor);
      wink_make_current(NULL, NULL, NULL);
      wink_context_destroy(ctx);
      return;
   }
   static const char *vs =
      "#version 330 core\n"
      "layout(location = 0) in vec2 pos;\n"
      "layout(location = 1) in vec3 tint;\n"
      "out vec3 v_tint;\n"
      "void main() { v_tint = tint; gl_Position = vec4(pos, 0.0, 1.0); }\n";
   static const char *fs =
      "#version 330 core\n"
      "in vec3 v_tint;\n"
      "out vec4 color;\n"
      "void main() { color = vec4(v_tint, 1.0); }\n";
   snprintf(name, sizeof(name), "%s multidraw quads", label);
   GLuint prog = link_program(vs, fs, name);
   /* Quad q covers third q, as four vertices from 4q. */
   float pos[3 * 8];
   for (int q = 0; q < 3; ++q) {
      const float x0 = -1.0f + 2.0f * q / 3.0f, x1 = -1.0f + 2.0f * (q + 1) / 3.0f;
      const float v[8] = { x0, -1.0f, x1, -1.0f, x1, 1.0f, x0, 1.0f };
      memcpy(pos + q * 8, v, sizeof(v));
   }
   static const float tints[4 * 3] = { 1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 0 };
   /* count, instanceCount, first, baseInstance */
   static const GLuint cmds[4 * 4] = {
      4, 1, 0, 2,   /* left: blue */
      0, 1, 0, 3,   /* empty */
      4, 1, 4, 0,   /* middle: red */
      4, 1, 8, 1,   /* right: green */
   };
   GLuint vao = 0, bufs[3] = {0};
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glGenBuffers(3, bufs);
   p_glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
   p_glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_STATIC_DRAW);
   p_glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   p_glEnableVertexAttribArray(0);
   p_glBindBuffer(GL_ARRAY_BUFFER, bufs[1]);
   p_glBufferData(GL_ARRAY_BUFFER, sizeof(tints), tints, GL_STATIC_DRAW);
   p_glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
   p_glEnableVertexAttribArray(1);
   p_glVertexAttribDivisor(1, 1);
   p_glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufs[2]);
   p_glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmds), cmds, GL_STATIC_DRAW);
   p_glUseProgram(prog);
   p_glViewport(0, 0, W, H);

   p_glClearColor(0, 0, 0, 1);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glMultiDrawArraysIndirect(GL_QUADS, NULL, 4, 0);
   static const int indirect_want[3][3] = { {0, 0, 255}, {255, 0, 0}, {0, 255, 0} };
   snprintf(name, sizeof(name), "%s quads indirect", label);
   check_thirds(name, indirect_want);

   /* Every sub-draw takes instance 0's tint; the right and left thirds are
    * drawn, in that order, and the middle one's sub-draw is empty. */
   p_glClear(GL_COLOR_BUFFER_BIT);
   static const GLint firsts[3] = { 8, 4, 0 };
   static const GLsizei counts[3] = { 4, 0, 4 };
   p_glMultiDrawArrays(GL_QUADS, firsts, counts, 3);
   static const int multi_want[3][3] = { {255, 0, 0}, {0, 0, 0}, {255, 0, 0} };
   snprintf(name, sizeof(name), "%s quads multidraw", label);
   check_thirds(name, multi_want);

   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s multidraw quads: GL error 0x%x", label, err);
   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glDeleteBuffers(3, bufs);
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   wink_make_current(NULL, NULL, NULL);
   wink_context_destroy(ctx);
}

/* One quad over third `t`, as four interleaved (x, y, r, g, b) vertices. */
static void
client_quad(float *v, int t, float r, float g, float b)
{
   const float x0 = -1.0f + 2.0f * t / 3.0f, x1 = -1.0f + 2.0f * (t + 1) / 3.0f;
   const float xy[8] = { x0, -1.0f, x1, -1.0f, x1, 1.0f, x0, 1.0f };
   for (int i = 0; i < 4; ++i) {
      v[i * 5 + 0] = xy[i * 2];
      v[i * 5 + 1] = xy[i * 2 + 1];
      v[i * 5 + 2] = r;
      v[i * 5 + 3] = g;
      v[i * 5 + 4] = b;
   }
}

/* Fixed-function draws from client arrays, the way Minecraft 1.16 draws its
 * entities, text and GUI. With GL on the main thread the driver reads these
 * arrays later, on another thread, so each draw either waits for it or takes
 * a copy of what it reads (glthread's draw_arrays_remote_copy) -- and the
 * application rewrites the array the moment the call returns, as here. A copy
 * taken late, or a wait that did not happen, draws the overwritten quad. */
static void
client_arrays(const char *label, wink_surface *surface)
{
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = WINK_API_OPENGL_COMPAT;
   int status = 0;
   wink_context *ctx = wink_context_create(&ca, &status);
   CHECK(ctx, "%s: compat context: %s", label, wink_status_string(status));
   if (!ctx)
      return;
   wink_make_current(ctx, surface, surface);
   char name[96];
   static float v[8 * 5];
   p_glUseProgram(0);
   p_glBindVertexArray(0);
   p_glBindBuffer(GL_ARRAY_BUFFER, 0);
   p_glViewport(0, 0, W, H);
   p_glClearColor(0, 0, 0, 1);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glEnableClientState(GL_VERTEX_ARRAY);
   p_glEnableClientState(GL_COLOR_ARRAY);
   p_glVertexPointer(2, GL_FLOAT, 5 * sizeof(float), v);
   p_glColorPointer(3, GL_FLOAT, 5 * sizeof(float), v + 2);

   client_quad(v, 0, 1, 0, 0);
   p_glDrawArrays(GL_QUADS, 0, 4);
   client_quad(v, 2, 0, 1, 0);
   p_glDrawArrays(GL_QUADS, 0, 4);
   client_quad(v, 1, 0, 0, 1);   /* never drawn */
   static const int want[3][3] = { {255, 0, 0}, {0, 0, 0}, {0, 255, 0} };
   snprintf(name, sizeof(name), "%s client arrays", label);
   check_thirds(name, want);

   /* A non-zero first: the draw reads vertices 4..7 only. */
   p_glClear(GL_COLOR_BUFFER_BIT);
   client_quad(v, 0, 1, 1, 0);
   client_quad(v + 4 * 5, 1, 0, 1, 1);
   p_glDrawArrays(GL_QUADS, 4, 4);
   client_quad(v + 4 * 5, 2, 1, 0, 1);   /* never drawn */
   static const int want_first[3][3] = { {0, 0, 0}, {0, 255, 255}, {0, 0, 0} };
   snprintf(name, sizeof(name), "%s client arrays from 4", label);
   check_thirds(name, want_first);

   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s client arrays: GL error 0x%x", label, err);
   p_glDisableClientState(GL_COLOR_ARRAY);
   p_glDisableClientState(GL_VERTEX_ARRAY);
   wink_make_current(NULL, NULL, NULL);
   wink_context_destroy(ctx);
}

static void
vertex_input_types(void)
{
   static const char *vs =
      "#version 330 core\n"
      "layout(location=3) in vec2 value;\n"
      "out vec2 data;\n"
      "void main() {\n"
      "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
      "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
      "  data = value;\n"
      "}\n";
   static const char *fs =
      "#version 330 core\n"
      "in vec2 data; out vec4 color;\n"
      "void main() { color = vec4(data, 0.0, 1.0); }\n";
   GLuint prog = link_program(vs, fs, "vertex input types");
   GLuint vao, buffer;
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glGenBuffers(1, &buffer);
   p_glBindBuffer(GL_ARRAY_BUFFER, buffer);
   p_glEnableVertexAttribArray(3);
   p_glUseProgram(prog);

   /* Iris uses glVertexAttribIPointer for a float input. GL leaves that
    * value undefined; this tests our raw-bit policy and, especially, that
    * the mismatch does not invalidate WebGPU or poison subsequent draws.
    * Switch the same shader back to ordinary floats to exercise its cache. */
   const GLenum types[] = {GL_FLOAT, GL_INT, GL_UNSIGNED_INT, GL_SHORT,
                          GL_UNSIGNED_SHORT, GL_FLOAT};
   const uint32_t float_bits[6] = {0x3f800000, 0x3f000000, 0x3f800000,
                                  0x3f000000, 0x3f800000, 0x3f000000};
   const uint16_t zeros[6] = {0};
   for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
      const bool short_type = types[i] == GL_SHORT || types[i] == GL_UNSIGNED_SHORT;
      p_glBufferData(GL_ARRAY_BUFFER, short_type ? sizeof(zeros) : sizeof(float_bits),
                     short_type ? (const void *)zeros : (const void *)float_bits,
                     GL_STATIC_DRAW);
      if (types[i] == GL_FLOAT)
         p_glVertexAttribPointer(3, 2, types[i], GL_FALSE, 8, NULL);
      else
         p_glVertexAttribIPointer(3, 2, types[i], short_type ? 4 : 8, NULL);
      p_glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
      p_glClear(GL_COLOR_BUFFER_BIT);
      p_glDrawArrays(GL_TRIANGLES, 0, 3);
      uint8_t px[4];
      read_center(px, W, H);
      CHECK(near(px, short_type ? 0 : 255, short_type ? 0 : 128, 0),
            "vertex type 0x%x: read %u,%u,%u,%u", types[i], px[0], px[1], px[2], px[3]);
      CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(),
            "vertex type 0x%x: context failed", types[i]);
   }
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glDeleteBuffers(1, &buffer);
}

static void
snorm_attachments(void)
{
   const GLenum internal[] = {GL_R8_SNORM, GL_RG8_SNORM, GL_RGB8_SNORM, GL_RGBA8_SNORM};
   const GLenum format[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
   GLuint texture, framebuffer;
   p_glGenTextures(1, &texture);
   p_glBindTexture(GL_TEXTURE_2D, texture);
   p_glGenFramebuffers(1, &framebuffer);
   p_glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   p_glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
   for (unsigned i = 0; i < 4; ++i) {
      p_glTexImage2D(GL_TEXTURE_2D, 0, internal[i], 4, 4, 0, format[i], GL_BYTE, NULL);
      p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
      GLenum status = p_glCheckFramebufferStatus(GL_FRAMEBUFFER);
      CHECK(status == GL_FRAMEBUFFER_COMPLETE, "snorm 0x%x: framebuffer 0x%x", internal[i], status);
      if (status != GL_FRAMEBUFFER_COMPLETE)
         continue;
      p_glClearColor(-1.0f, -0.5f, 0.25f, 0.75f);
      p_glClear(GL_COLOR_BUFFER_BIT);
      float px[4] = {0};
      p_glReadPixels(2, 2, 1, 1, GL_RGBA, GL_FLOAT, px);
      const float expected[4] = {-1.0f, i >= 1 ? -0.5f : 0.0f,
                                 i >= 2 ? 0.25f : 0.0f, i >= 3 ? 0.75f : 1.0f};
      for (unsigned c = 0; c < 4; ++c)
         CHECK(fabsf(px[c] - expected[c]) < 0.01f,
               "snorm 0x%x: component %u read %f expected %f", internal[i], c, px[c], expected[c]);
   }
   CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(), "snorm attachments: context failed");
   p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
   p_glClampColor(GL_CLAMP_READ_COLOR, GL_FIXED_ONLY);
   p_glDeleteFramebuffers(1, &framebuffer);
   p_glDeleteTextures(1, &texture);
}

static void
norm16_filtering(void)
{
   static const char *vs =
      "#version 330 core\n"
      "void main() {\n"
      "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
      "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
      "}\n";
   static const char *fs =
      "#version 330 core\n"
      "uniform sampler2D tex; out vec4 color;\n"
      "void main() { color = texture(tex, vec2(0.5)); }\n";
   GLuint prog = link_program(vs, fs, "norm16 filtering");
   GLuint vao, texture;
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glGenTextures(1, &texture);
   p_glBindTexture(GL_TEXTURE_2D, texture);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   const uint16_t unorm[8] = {0, 65535, 32768, 65535, 65535, 0, 32768, 65535};
   const int16_t snorm[8] = {-32767, 32767, 0, 32767, 32767, -32767, 0, 32767};
   p_glUseProgram(prog);
   for (unsigned i = 0; i < 2; ++i) {
      p_glTexImage2D(GL_TEXTURE_2D, 0, i ? GL_RGBA16_SNORM : GL_RGBA16,
                    2, 1, 0, GL_RGBA, i ? GL_SHORT : GL_UNSIGNED_SHORT,
                    i ? (const void *)snorm : (const void *)unorm);
      p_glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
      p_glClear(GL_COLOR_BUFFER_BIT);
      p_glDrawArrays(GL_TRIANGLES, 0, 3);
      uint8_t px[4];
      read_center(px, W, H);
      CHECK(near(px, i ? 0 : 128, i ? 0 : 128, i ? 0 : 128),
            "%snorm16 linear filtering: read %u,%u,%u,%u", i ? "s" : "u", px[0], px[1], px[2], px[3]);
   }
   CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(), "norm16 filtering: context failed");
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glDeleteTextures(1, &texture);
}

/* A comparison-enabled depth texture is still read as raw depth by
 * texelFetch: this operation has no sampler. */
static void
raw_shadow_depth(void)
{
   const char *vs = "#version 330 core\n"
      "void main() { vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
      "gl_Position=vec4(p*2.0-1.0,0,1); }\n";
   const char *fs = "#version 330 core\n"
      "uniform sampler2D depth; out vec4 color;"
      "void main() { color=vec4(texelFetch(depth,ivec2(0),0).r,0,0,1); }\n";
   GLuint tex, vao;
   const float depth = 0.25f;
   p_glGenTextures(1, &tex);
   p_glBindTexture(GL_TEXTURE_2D, tex);
   p_glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 1, 1, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
   GLuint prog = link_program(vs, fs, "raw shadow depth");
   p_glGenVertexArrays(1, &vao);
   p_glBindVertexArray(vao);
   p_glUseProgram(prog);
   p_glClearColor(0, 0, 0, 1);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArrays(GL_TRIANGLES, 0, 3);
   uint8_t px[4];
   read_center(px, W, H);
   CHECK(near(px, 64, 0, 0), "raw shadow depth: read %u,%u,%u,%u", px[0], px[1], px[2], px[3]);
   CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(), "raw shadow depth: context failed");
   p_glUseProgram(0);
   p_glBindVertexArray(0);
   p_glDeleteProgram(prog);
   p_glDeleteVertexArrays(1, &vao);
   p_glDeleteTextures(1, &tex);
}

static void
buffer_read_sources(void)
{
   const uint32_t original[4] = {3, 11, 29, 47};
   uint32_t data[4] = {0};
   GLuint buffers[3];
   p_glGenBuffers(3, buffers);
   p_glBindBuffer(GL_COPY_READ_BUFFER, buffers[0]);
   p_glBufferData(GL_COPY_READ_BUFFER, sizeof(original), original, GL_STATIC_DRAW);
   p_glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(data), data);
   CHECK(memcmp(data, original, sizeof(data)) == 0, "CPU-authored buffer read");

   p_glBindBuffer(GL_COPY_WRITE_BUFFER, buffers[1]);
   p_glBufferData(GL_COPY_WRITE_BUFFER, sizeof(data), NULL, GL_STATIC_DRAW);
   p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(data));
   p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(data), data);
   CHECK(memcmp(data, original, sizeof(data)) == 0, "GPU-copied buffer read");
   p_glBindBuffer(GL_COPY_READ_BUFFER, buffers[1]);
   p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 12, 4);
   p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(data), data);
   CHECK(data[0] == 3 && data[1] == 11 && data[2] == 29 && data[3] == 3,
         "CPU-authored self-copy preserved other bytes");
   p_glBindBuffer(GL_COPY_READ_BUFFER, buffers[0]);
   p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(data));
   p_glBindBuffer(GL_COPY_WRITE_BUFFER, buffers[2]);
   p_glBufferData(GL_COPY_WRITE_BUFFER, sizeof(data), original, GL_STATIC_DRAW);
   p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 12, 4);
   p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(data), data);
   CHECK(data[0] == 3 && data[1] == 11 && data[2] == 29 && data[3] == 3,
         "partial copy preserved an unread destination");
   p_glBindBuffer(GL_COPY_WRITE_BUFFER, buffers[1]);

   static const char *cs =
      "#version 430 core\n"
      "layout(local_size_x=1) in;\n"
      "layout(std430, binding=0) buffer Data { uint values[]; };\n"
      "void main() { values[gl_GlobalInvocationID.x] += 17u; }\n";
   GLuint shader = compile(GL_COMPUTE_SHADER, cs), prog = p_glCreateProgram();
   p_glAttachShader(prog, shader);
   p_glLinkProgram(prog);
   GLint linked = 0;
   p_glGetProgramiv(prog, GL_LINK_STATUS, &linked);
   CHECK(linked, "buffer read compute program linked");
   p_glUseProgram(prog);
   p_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[1]);
   p_glDispatchCompute(4, 1, 1);
   p_glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(data), data);
   for (unsigned i = 0; i < 4; ++i)
      CHECK(data[i] == original[i] + 17, "GPU-written buffer read %u: %u", i, data[i]);

   GLuint persistent;
   p_glGenBuffers(1, &persistent);
   p_glBindBuffer(GL_COPY_READ_BUFFER, persistent);
   const GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                            GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
   p_glBufferStorage(GL_COPY_READ_BUFFER, sizeof(data), original, flags);
   uint32_t *mapped = p_glMapBufferRange(GL_COPY_READ_BUFFER, 0, sizeof(data), flags);
   CHECK(mapped && memcmp(mapped, original, sizeof(data)) == 0, "CPU-authored persistent read");
   if (mapped) {
      p_glBindBuffer(GL_COPY_READ_BUFFER, buffers[1]);
      p_glBindBuffer(GL_COPY_WRITE_BUFFER, persistent);
      p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(data));
      p_glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
      p_glFinish();
      for (unsigned i = 0; i < 4; ++i)
         CHECK(mapped[i] == original[i] + 17, "GPU-copied persistent read %u: %u", i, mapped[i]);
      mapped[0] = 123;
      p_glFinish();
      p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(data), data);
      CHECK(data[0] == 123 && data[3] == original[3] + 17,
            "persistent CPU write preserved GPU data: %u, %u", data[0], data[3]);
      p_glUnmapBuffer(GL_COPY_WRITE_BUFFER);
   }
   p_glDeleteBuffers(1, &persistent);
   CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(), "buffer reads: context failed");
   p_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glDeleteShader(shader);
   p_glDeleteBuffers(3, buffers);
}

/* Repeated triangles in a single draw must read earlier primitives' writes.
 * Three increments distinguish coherent fetch from a once-per-draw snapshot. */
static void
framebuffer_fetch(void)
{
   GLint n = 0;
   bool supported = false;
   p_glGetIntegerv(GL_NUM_EXTENSIONS, &n);
   for (GLint i = 0; i < n; ++i)
      supported |= strcmp((const char *)p_glGetStringi(GL_EXTENSIONS, i),
                          "GL_EXT_shader_framebuffer_fetch") == 0;
   if (!supported) {
      note("coherent framebuffer fetch: unavailable");
      return;
   }
   const char *vs = "#version 300 es\n"
      "void main(){ int i=gl_VertexID%3; vec2 p=vec2(i==1?3.0:-1.0,i==2?3.0:-1.0);"
      "gl_Position=vec4(p,0,1); }\n";
   const char *fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 color;"
      "void main(){ color += vec4(0.125,0.0,0.0,0.0); }\n";
   GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
   GLuint prog = p_glCreateProgram();
   p_glAttachShader(prog, v); p_glAttachShader(prog, f); p_glLinkProgram(prog);
   GLint linked = 0; p_glGetProgramiv(prog, GL_LINK_STATUS, &linked);
   CHECK(linked, "framebuffer fetch: link failed");
   p_glUseProgram(prog);
   p_glClearColor(0, 0, 0, 1); p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArrays(GL_TRIANGLES, 0, 9);
   uint8_t px[4]; read_center(px, W, H);
   CHECK(near(px, 96, 0, 0), "coherent framebuffer fetch: %u,%u,%u,%u", px[0],px[1],px[2],px[3]);
   CHECK(p_glGetError() == GL_NO_ERROR && !wink_device_lost(), "framebuffer fetch: context failed");
   p_glDeleteProgram(prog); p_glDeleteShader(f);
   const char *mrt = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 c0;"
      "layout(location=1) inout vec4 c1;"
      "void main(){ c0+=vec4(0.125,0,0,0); c1+=vec4(0,0.25,0,0); }\n";
   f = compile(GL_FRAGMENT_SHADER, mrt);
   prog = p_glCreateProgram(); p_glAttachShader(prog,v); p_glAttachShader(prog,f); p_glLinkProgram(prog);
   p_glGetProgramiv(prog,GL_LINK_STATUS,&linked); CHECK(linked,"MRT fetch: link failed");
   p_glUseProgram(prog);
   const char *array_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 c[2];"
      "void main(){ int i=int(gl_FragCoord.x)&1; c[i]+=i==0?vec4(0.125,0,0,0):vec4(0,0.25,0,0); }";
   GLuint array_prog=link_program(vs,array_fs,"array fetch");
   const char *sample_vs = "#version 320 es\n"
      "void main(){ int i=gl_VertexID%3; gl_Position=vec4(i==1?3.0:-1.0,i==2?3.0:-1.0,0,1); }";
   const char *sample_fs = "#version 320 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 c;"
      "void main(){ c.r+=float(gl_SampleID+1)*0.125; }";
   GLuint sample_prog=link_program(sample_vs,sample_fs,"per-sample fetch");
   const char *readonly_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 source;"
      "layout(location=1) out vec4 destination; void main(){ destination=source+vec4(0.125,0,0,0); }";
   GLuint readonly_prog=link_program(vs,readonly_fs,"readonly fetch");
   const GLenum buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
   GLuint fb[2], tex[3]; p_glGenFramebuffers(2,fb); p_glGenTextures(3,tex);
   p_glBindFramebuffer(GL_FRAMEBUFFER,fb[1]);
   p_glBindTexture(GL_TEXTURE_2D,tex[2]);
   p_glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
   p_glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex[2],0);
   for (unsigned samples = 1; samples <= 4; samples += 3) {
      const GLenum target = samples == 1 ? GL_TEXTURE_2D : GL_TEXTURE_2D_MULTISAMPLE;
      p_glBindFramebuffer(GL_FRAMEBUFFER,fb[0]);
      for (unsigned i=0;i<2;++i) {
         p_glBindTexture(target,tex[i]);
         if (samples == 1)
            p_glTexImage2D(target,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
         else
            p_glTexStorage2DMultisample(target,4,GL_RGBA8,W,H,GL_TRUE);
         p_glFramebufferTexture2D(GL_FRAMEBUFFER,buffers[i],target,tex[i],0);
      }
      p_glDrawBuffers(2,buffers);
      CHECK(p_glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"MRT fetch framebuffer");
      p_glClear(GL_COLOR_BUFFER_BIT); p_glDrawArrays(GL_TRIANGLES,0,9);
      for (unsigned i=0;i<2;++i) {
         p_glReadBuffer(buffers[i]);
         if (samples == 4) {
            p_glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fb[1]);
            p_glBlitFramebuffer(0,0,W,H,0,0,W,H,GL_COLOR_BUFFER_BIT,GL_NEAREST);
            p_glBindFramebuffer(GL_READ_FRAMEBUFFER,fb[1]);
         }
         read_center(px,W,H);
         CHECK(near(px,i==0?96:0,i==1?192:0,0),"MRT fetch %ux RT%u: %u,%u,%u,%u",
               samples,i,px[0],px[1],px[2],px[3]);
         p_glBindFramebuffer(GL_FRAMEBUFFER,fb[0]);
      }
      if (samples==1) {
         p_glUseProgram(readonly_prog); p_glClearColor(0.25,0,0,1); p_glClear(GL_COLOR_BUFFER_BIT);
         p_glDrawArrays(GL_TRIANGLES,0,9);
         for (unsigned i=0;i<2;++i) {
            p_glReadBuffer(buffers[i]); read_center(px,W,H);
            CHECK(near(px,i==0?64:96,0,0),"readonly fetch RT%u: %u,%u,%u,%u",
                  i,px[0],px[1],px[2],px[3]);
         }
         p_glClearColor(0,0,0,1);
         p_glUseProgram(array_prog); p_glClear(GL_COLOR_BUFFER_BIT);
         p_glDrawArrays(GL_TRIANGLES,0,9);
         for (unsigned i=0;i<2;++i) {
            p_glReadBuffer(buffers[i]);
            p_glReadPixels(W/2+i,H/2,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
            CHECK(near(px,i==0?96:0,i==1?192:0,0),"array fetch RT%u: %u,%u,%u,%u",
                  i,px[0],px[1],px[2],px[3]);
         }
         p_glUseProgram(prog);
      } else {
         p_glUseProgram(sample_prog); p_glClear(GL_COLOR_BUFFER_BIT);
         p_glDrawArrays(GL_TRIANGLES,0,9);
         p_glReadBuffer(GL_COLOR_ATTACHMENT0); p_glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fb[1]);
         p_glBlitFramebuffer(0,0,W,H,0,0,W,H,GL_COLOR_BUFFER_BIT,GL_NEAREST);
         p_glBindFramebuffer(GL_READ_FRAMEBUFFER,fb[1]); read_center(px,W,H);
         CHECK(near(px,200,0,0),"per-sample fetch: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
         p_glBindFramebuffer(GL_FRAMEBUFFER,fb[0]); p_glUseProgram(prog);
      }
      p_glBindTexture(target,0);
      // A texture's target is immutable; make new objects for the MSAA pass.
      if (samples == 1) { p_glDeleteTextures(2,tex); p_glGenTextures(2,tex); }
   }
   p_glBindFramebuffer(GL_FRAMEBUFFER,0);
   p_glDeleteFramebuffers(2,fb); p_glDeleteTextures(3,tex);
   CHECK(p_glGetError()==GL_NO_ERROR && !wink_device_lost(),"MRT fetch context failed");
   p_glUseProgram(0); p_glDeleteProgram(prog); p_glDeleteShader(v); p_glDeleteShader(f);
   p_glDeleteProgram(array_prog);
   p_glDeleteProgram(sample_prog);
   p_glDeleteProgram(readonly_prog);

   const char *inst_vs = "#version 300 es\n"
      "flat out int instance; void main(){ int i=gl_VertexID%3;"
      "gl_Position=vec4(i==1?3.0:-1.0,i==2?3.0:-1.0,0,1); instance=gl_InstanceID; }";
   const char *inst_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; flat in int instance; layout(location=0) inout vec4 c;"
      "void main(){ c.r=c.r*0.5+float(instance+1)*0.125; }";
   prog = link_program(inst_vs, inst_fs, "instanced fetch");
   p_glUseProgram(prog); p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArraysInstanced(GL_TRIANGLES,3,3,3); read_center(px,W,H);
   CHECK(near(px,136,0,0),"instanced fetch: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   GLuint indices; const uint16_t elements[] = {3,4,5};
   p_glGenBuffers(1,&indices); p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indices);
   p_glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(elements),elements,GL_STATIC_DRAW);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawElementsInstanced(GL_TRIANGLES,3,GL_UNSIGNED_SHORT,NULL,3); read_center(px,W,H);
   CHECK(near(px,136,0,0),"indexed instanced fetch: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0); p_glDeleteBuffers(1,&indices);
   p_glDeleteProgram(prog);
   const char *strip_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 c;"
      "void main(){ c+=gl_FrontFacing?vec4(0.125,0,0,0):vec4(0,0,0.125,0); }";
   prog = link_program(vs,strip_fs,"strip fetch"); p_glUseProgram(prog);
   p_glClear(GL_COLOR_BUFFER_BIT); p_glDrawArrays(GL_TRIANGLE_STRIP,0,6); read_center(px,W,H);
   CHECK(near(px,64,0,64),"strip fetch winding: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   p_glClear(GL_COLOR_BUFFER_BIT); p_glEnable(GL_CULL_FACE);
   p_glDrawArrays(GL_TRIANGLE_STRIP,0,6); p_glDisable(GL_CULL_FACE); read_center(px,W,H);
   CHECK(near(px,64,0,0),"strip fetch culling: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   p_glDeleteProgram(prog);

   uint8_t gradient[W*H*4];
   for (unsigned y=0;y<H;++y) for(unsigned x=0;x<W;++x) {
      unsigned at=(y*W+x)*4;
      gradient[at]=(y%4)*32; gradient[at+1]=(x%4)*32;
      gradient[at+2]=0; gradient[at+3]=255;
   }
   p_glGenTextures(1,tex); p_glGenFramebuffers(1,fb);
   p_glBindTexture(GL_TEXTURE_2D,tex[0]);
   p_glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,gradient);
   p_glBindFramebuffer(GL_FRAMEBUFFER,fb[0]);
   p_glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex[0],0);
   prog=link_program(vs,fs,"coordinate fetch"); p_glUseProgram(prog);
   p_glDrawArrays(GL_TRIANGLES,0,9);
   p_glReadPixels(9,10,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
   CHECK(near(px,160,32,0),"fetch coordinates: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   p_glDeleteProgram(prog);
   const char *uint_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; precision highp int; layout(location=0) inout uvec4 c;"
      "void main(){ c.r+=1u; }";
   uint32_t initial[W*H];
   for (unsigned i=0;i<W*H;++i) initial[i]=1;
   p_glTexImage2D(GL_TEXTURE_2D,0,GL_R32UI,W,H,0,GL_RED_INTEGER,GL_UNSIGNED_INT,initial);
   prog=link_program(vs,uint_fs,"integer fetch"); p_glUseProgram(prog);
   p_glDrawArrays(GL_TRIANGLES,0,9);
   uint32_t integer=0; p_glReadPixels(9,10,1,1,GL_RED_INTEGER,GL_UNSIGNED_INT,&integer);
   CHECK(integer==4,"integer fetch: %u, want 4",integer);
   p_glDeleteProgram(prog);
   uint8_t red[W*H]={0};
   p_glTexImage2D(GL_TEXTURE_2D,0,GL_R8,W,H,0,GL_RED,GL_UNSIGNED_BYTE,red);
   const char *red_fs = "#version 300 es\n"
      "#extension GL_EXT_shader_framebuffer_fetch : require\n"
      "precision highp float; layout(location=0) inout vec4 c;"
      "void main(){ c.r=(c.g==0.0&&c.b==0.0&&c.a==1.0)?c.r+0.125:0.0; }";
   prog=link_program(vs,red_fs,"single channel fetch"); p_glUseProgram(prog);
   p_glDrawArrays(GL_TRIANGLES,0,9); read_center(px,W,H);
   CHECK(near(px,96,0,0),"single channel fetch: %u,%u,%u,%u",px[0],px[1],px[2],px[3]);
   p_glBindFramebuffer(GL_FRAMEBUFFER,0); p_glDeleteFramebuffers(1,fb); p_glDeleteTextures(1,tex);
   p_glUseProgram(0); p_glDeleteProgram(prog);
   CHECK(p_glGetError()==GL_NO_ERROR && !wink_device_lost(),"ordered fetch context failed");
}

/* Buffer uploads bigger than one glthread command (8 KiB): with GL on the main
 * thread they are sent along as a copy, and the application rewrites its memory
 * the moment each call returns, as here. A 64 KiB vertex buffer filled with
 * glBufferData (left and right quads), drawn; then a 20 KiB glBufferSubData
 * that also places a middle quad at vertex 1000, drawn again. */
static void
big_uploads(const char *label, wink_surface *surface)
{
   wink_context_attribs ca;
   wink_context_attribs_default(&ca);
   ca.api = WINK_API_OPENGL_COMPAT;
   int status = 0;
   wink_context *ctx = wink_context_create(&ca, &status);
   CHECK(ctx, "%s: compat context: %s", label, wink_status_string(status));
   if (!ctx)
      return;
   wink_make_current(ctx, surface, surface);
   char name[96];
   enum { N = 64 * 1024 / (5 * sizeof(float)) };
   const size_t bytes = N * 5 * sizeof(float);
   float *v = calloc(N * 5, sizeof(float));
   GLuint buf = 0;
   p_glUseProgram(0);
   p_glBindVertexArray(0);
   p_glGenBuffers(1, &buf);
   p_glBindBuffer(GL_ARRAY_BUFFER, buf);
   p_glViewport(0, 0, W, H);
   p_glClearColor(0, 0, 0, 1);
   p_glEnableClientState(GL_VERTEX_ARRAY);
   p_glEnableClientState(GL_COLOR_ARRAY);
   p_glVertexPointer(2, GL_FLOAT, 5 * sizeof(float), (const void *)0);
   p_glColorPointer(3, GL_FLOAT, 5 * sizeof(float),
                    (const void *)(2 * sizeof(float)));

   client_quad(v, 0, 1, 0, 0);          /* vertices 0..3: left, red */
   client_quad(v + 4 * 5, 2, 0, 1, 0);  /* 4..7: right, green */
   p_glBufferData(GL_ARRAY_BUFFER, bytes, v, GL_STREAM_DRAW);
   memset(v, 0xff, bytes);              /* rewritten after the call */
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArrays(GL_QUADS, 0, 8);
   static const int want_data[3][3] = { {255, 0, 0}, {0, 0, 0}, {0, 255, 0} };
   snprintf(name, sizeof(name), "%s big BufferData", label);
   check_thirds(name, want_data);

   memset(v, 0, bytes);
   client_quad(v, 0, 1, 0, 0);
   client_quad(v + 4 * 5, 2, 0, 1, 0);
   client_quad(v + 1000 * 5, 1, 1, 1, 0);   /* 1000..1003: middle, yellow */
   p_glBufferSubData(GL_ARRAY_BUFFER, 0, 1004 * 5 * sizeof(float), v);
   memset(v, 0xff, bytes);                  /* rewritten after the call */
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glDrawArrays(GL_QUADS, 0, 8);
   p_glDrawArrays(GL_QUADS, 1000, 4);
   static const int want_sub[3][3] = { {255, 0, 0}, {255, 255, 0}, {0, 255, 0} };
   snprintf(name, sizeof(name), "%s big BufferSubData", label);
   check_thirds(name, want_sub);

   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s big uploads: GL error 0x%x", label, err);
   p_glDisableClientState(GL_COLOR_ARRAY);
   p_glDisableClientState(GL_VERTEX_ARRAY);
   p_glBindBuffer(GL_ARRAY_BUFFER, 0);
   p_glDeleteBuffers(1, &buf);
   free(v);
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

   /* One paced swap each way: a turn of the event loop, then a frame. */
   wink_swap_interval(surface, 0);
   status = wink_swap_buffers(surface);
   CHECK(status == WINK_OK, "%s: swap: %s", label, wink_status_string(status));
   wink_swap_interval(surface, 1);

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
   wink_swap_interval(surface, (init_flags & WINK_INIT_GL_ON_MAIN_THREAD) ? 1 : WINK_SWAP_UNPACED);

   /* And back up, past the old size: a pixel there exists only if the
    * default framebuffer really was rebuilt. */
   wink_surface_resize(surface, W, H);
   p_glViewport(0, 0, W, H);
   p_glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
   p_glClear(GL_COLOR_BUFFER_BIT);
   p_glReadPixels(W - 2, H - 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   CHECK(near(px, 0, 0, 255), "%s: regrown read %u,%u,%u,%u", label, px[0], px[1], px[2], px[3]);

   GLenum err = p_glGetError();
   CHECK(err == GL_NO_ERROR, "%s: GL error 0x%x", label, err);

   if (api == WINK_API_OPENGL_ES)
      framebuffer_fetch();

   if (api == WINK_API_OPENGL_CORE) {
      vertex_input_types();
      snorm_attachments();
      norm16_filtering();
      raw_shadow_depth();
      buffer_read_sources();
   }

   p_glBindVertexArray(0);
   p_glDeleteVertexArrays(1, &vao);
   p_glUseProgram(0);
   p_glDeleteProgram(prog);
   p_glDeleteShader(v);
   p_glDeleteShader(f);
   wink_make_current(NULL, NULL, NULL);
   wink_context_destroy(ctx);
}

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
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
   wink_init_attribs ia;
   wink_init_attribs_default(&ia);
   ia.flags = init_flags;
   int status = wink_init(&ia);
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
#ifndef __EMSCRIPTEN__
   sa.offscreen = 1;
#endif
   wink_surface *surface = wink_surface_create(&sa, &status);
   CHECK(surface, "surface: %s", wink_status_string(status));
   if (!surface)
      return 1;

   /* Pacing: unpaced by default in the ordinary mode; a paced swap and a
    * bare yield both suspend on this thread and come back. */
   if (!(init_flags & WINK_INIT_GL_ON_MAIN_THREAD)) {
      wink_yield_frames(0);
      wink_yield_frames(1);
   }
   CHECK(wink_swap_interval(surface, -2) == WINK_ERROR_BAD_ATTRIBUTE, "swap interval -2 accepted");

   exercise("es", WINK_API_OPENGL_ES, surface);
   exercise("core", WINK_API_OPENGL_CORE, surface);
   exercise("compat", WINK_API_OPENGL_COMPAT, surface);
   compressed(label, surface);
   multidraw_quads(label, surface);
   client_arrays(label, surface);
   big_uploads(label, surface);

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

static int
run_all(void)
{
   run_pass("native compression");
   /* And again from a fresh device with ASTC and ETC2 hidden from the
    * driver: st/mesa must still expose GL ES 3.2, decoding on the CPU. */
   wink_setenv("GEX_WGPU_NO_ASTC", "1");
   wink_setenv("GEX_WGPU_NO_ETC2", "1");
   run_pass("cpu-decoded compression");
   wink_setenv("GEX_WGPU_NO_ASTC", NULL);
   wink_setenv("GEX_WGPU_NO_ETC2", NULL);
   note(failures ? "FAILED %d" : "PASS", failures);
   return failures;
}

#ifdef __EMSCRIPTEN__
/* How deep this thread's C stack went, which is what the manifest's
 * STACK_SIZE and a GL thread's stack are sized by: what is below the
 * stack pointer marked first, and what was written over found after. */
#define STACK_MARK 0xa5
static void
mark_stack(void)
{
   const uintptr_t end = emscripten_stack_get_end(), here = emscripten_stack_get_current();
   if (here > end + 8192)
      memset((void *)end, STACK_MARK, here - end - 4096);
}

static void
note_stack(void)
{
   const uintptr_t end = emscripten_stack_get_end(), base = emscripten_stack_get_base();
   const uint8_t *p = (const uint8_t *)end;
   while ((uintptr_t)p < base && *p == STACK_MARK)
      ++p;
   note("stack: %u KiB deep of %u KiB", (unsigned)((base - (uintptr_t)p) >> 10),
        (unsigned)((base - end) >> 10));
}

EMSCRIPTEN_KEEPALIVE int
wink_smoke_main(void)
{
   wink_test_publish();
   mark_stack();
   const int failed = run_all();
   note_stack();
   return failed;
}
#else
int
main(void)
{
   return run_all();
}
#endif

#ifdef __EMSCRIPTEN_PTHREADS__
/* The same suite from a worker, with GL executed on the main thread: this
 * returns at once, so the main thread stays free to be the driver thread,
 * and the result arrives through Module.winkSmokeDone. */
static void *
smoke_thread(void *arg)
{
   (void)arg;
   init_flags = WINK_INIT_GL_ON_MAIN_THREAD;
   int rc = run_all();
   MAIN_THREAD_ASYNC_EM_ASM({ Module['winkSmokeDone']($0); }, rc);
   return NULL;
}

EMSCRIPTEN_KEEPALIVE int
wink_smoke_start_worker(void)
{
   pthread_t thread;
   if (pthread_create(&thread, NULL, smoke_thread, NULL) != 0)
      return 1;
   pthread_detach(thread);
   return 0;
}

/* library_wink_test.js: on this thread's event loop, call the named export
 * through WebAssembly.promising, as the .NET runtime enters its exports. */
void wink_test_run_promising(const char *export_name);

/* The same suite again, on a worker that owns the canvas: wink in its
 * ordinary mode, with the device, the OffscreenCanvas and JSPI all on the
 * worker. The body is entered through promising from the worker's event
 * loop -- pthread entry points are not -- and reports to the main thread. */
EMSCRIPTEN_KEEPALIVE int
wink_smoke_offscreen_body(void)
{
   init_flags = 0;
   CHECK(!emscripten_is_main_browser_thread(), "offscreen body on the main thread");
   int rc = run_all();
   MAIN_THREAD_ASYNC_EM_ASM({ Module['winkSmokeDone']($0); }, rc);
   emscripten_runtime_keepalive_pop();
   return rc;
}

static void *
offscreen_thread(void *arg)
{
   (void)arg;
   /* Return from the entry point, but keep the worker for the body. */
   emscripten_runtime_keepalive_push();
   wink_test_run_promising("wink_smoke_offscreen_body");
   return NULL;
}

EMSCRIPTEN_KEEPALIVE int
wink_smoke_start_offscreen(void)
{
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   /* "#canvas" is Module.canvas; it arrives as an OffscreenCanvas. */
   emscripten_pthread_attr_settransferredcanvases(&attr, "#canvas");
   pthread_t thread;
   int rc = pthread_create(&thread, &attr, offscreen_thread, NULL);
   pthread_attr_destroy(&attr);
   if (rc != 0)
      return 1;
   pthread_detach(thread);
   return 0;
}
#endif

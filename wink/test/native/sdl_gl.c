/* SPDX-License-Identifier: MIT */
/*
 * libwink as SDL's GL on a desktop (the wink fork of SDL, Cocoa, with
 * SDL_VIDEO_COCOA_WINK naming libwink.dylib): a core context, a clear and a
 * draw read back, a resize and another read, and the gl* symbols a loader
 * finds by dlsym being SDL_GL_GetProcAddress's -- what LWJGL checks.
 */
#include <SDL3/SDL.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef float GLfloat;
typedef char GLchar;
typedef unsigned char GLubyte;

#define GL_FUNCS(X)                                                                   \
   X(const GLubyte *, GetString, (GLenum))                                            \
   X(GLenum, GetError, (void))                                                        \
   X(void, Viewport, (GLint, GLint, GLsizei, GLsizei))                                \
   X(void, ClearColor, (GLfloat, GLfloat, GLfloat, GLfloat))                          \
   X(void, Clear, (GLbitfield))                                                       \
   X(void, ReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *))      \
   X(GLuint, CreateShader, (GLenum))                                                  \
   X(void, ShaderSource, (GLuint, GLsizei, const GLchar *const *, const GLint *))     \
   X(void, CompileShader, (GLuint))                                                   \
   X(GLuint, CreateProgram, (void))                                                   \
   X(void, AttachShader, (GLuint, GLuint))                                            \
   X(void, LinkProgram, (GLuint))                                                     \
   X(void, GetProgramiv, (GLuint, GLenum, GLint *))                                   \
   X(void, UseProgram, (GLuint))                                                      \
   X(void, GenVertexArrays, (GLsizei, GLuint *))                                      \
   X(void, BindVertexArray, (GLuint))                                                 \
   X(void, DrawArrays, (GLenum, GLint, GLsizei))

#define X(ret, name, params) static ret (*gl##name) params;
GL_FUNCS(X)
#undef X

static int failures;

static void
check(int ok, const char *what)
{
   printf("%s %s\n", ok ? "ok  " : "FAIL", what);
   failures += !ok;
}

/* The triangle's red at the centre, the clear's blue in a corner. */
static void
check_frame(const char *when, int w, int h)
{
   unsigned char centre[4], corner[4];
   glReadPixels(w / 2, h / 2, 1, 1, 0x1908 /* RGBA */, 0x1401 /* UNSIGNED_BYTE */, centre);
   glReadPixels(1, 1, 1, 1, 0x1908, 0x1401, corner);
   char what[128];
   snprintf(what, sizeof(what), "%s: centre %u,%u,%u corner %u,%u,%u", when,
            centre[0], centre[1], centre[2], corner[0], corner[1], corner[2]);
   check(centre[0] > 250 && centre[1] < 5 && centre[2] < 5 &&
         corner[0] == 51 && corner[1] == 153 && corner[2] == 229, what);
}

static void
draw(int w, int h)
{
   glViewport(0, 0, w, h);
   glClearColor(0.2f, 0.6f, 0.9f, 1.0f);
   glClear(0x4000 | 0x100);
   glDrawArrays(0x0004 /* TRIANGLES */, 0, 3);
}

int
main(void)
{
   const char *wink = getenv("SDL_VIDEO_COCOA_WINK");
   if (!wink || !*wink) {
      fprintf(stderr, "SDL_VIDEO_COCOA_WINK names no libwink\n");
      return 2;
   }
   if (!SDL_Init(SDL_INIT_VIDEO)) {
      fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
      return 1;
   }
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
   SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
   SDL_Window *window = SDL_CreateWindow("wink sdl_gl", 320, 240, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
   SDL_GLContext context = window ? SDL_GL_CreateContext(window) : NULL;
   if (!context) {
      fprintf(stderr, "context: %s\n", SDL_GetError());
      return 1;
   }

#define X(ret, name, params) gl##name = (ret (*) params)SDL_GL_GetProcAddress("gl" #name);
   GL_FUNCS(X)
#undef X
   printf("GL_VERSION %s\nGL_RENDERER %s\n", glGetString(0x1F02), glGetString(0x1F01));

   void *library = dlopen(wink, RTLD_NOW | RTLD_NOLOAD);
   check(library && dlsym(library, "glGetError") == (void *)glGetError &&
         dlsym(library, "glActiveTextureARB") == (void *)SDL_GL_GetProcAddress("glActiveTexture"),
         "dlsym's gl* are SDL_GL_GetProcAddress's");

   static const char *vs =
      "#version 330 core\n"
      "void main() {\n"
      "   vec2 p[3] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.0, 0.5));\n"
      "   gl_Position = vec4(p[gl_VertexID], 0.0, 1.0);\n"
      "}\n";
   static const char *fs =
      "#version 330 core\n"
      "out vec4 colour;\n"
      "void main() { colour = vec4(1.0, 0.0, 0.0, 1.0); }\n";
   GLuint program = glCreateProgram();
   const char *sources[2] = { vs, fs };
   for (int i = 0; i < 2; ++i) {
      GLuint shader = glCreateShader(i ? 0x8B30 /* FRAGMENT */ : 0x8B31 /* VERTEX */);
      glShaderSource(shader, 1, &sources[i], NULL);
      glCompileShader(shader);
      glAttachShader(program, shader);
   }
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, 0x8B82 /* LINK_STATUS */, &linked);
   check(linked, "program linked");
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glUseProgram(program);

   int w, h;
   SDL_GetWindowSizeInPixels(window, &w, &h);
   for (int i = 0; i < 30; ++i) {
      SDL_Event e;
      while (SDL_PollEvent(&e)) {
      }
      draw(w, h);
      if (i == 29)
         check_frame("first size", w, h);
      SDL_GL_SwapWindow(window);
   }

   SDL_SetWindowSize(window, 400, 300);
   SDL_SyncWindow(window);
   for (int i = 0; i < 30; ++i) {
      SDL_Event e;
      while (SDL_PollEvent(&e)) {
      }
      SDL_GetWindowSizeInPixels(window, &w, &h);
      draw(w, h);
      if (i == 29) {
         char what[64];
         snprintf(what, sizeof(what), "resized to %dx%d", w, h);
         check(w > 320, what);
         check_frame("after the resize", w, h);
      }
      SDL_GL_SwapWindow(window);
   }
   check(glGetError() == 0, "no GL error");

   SDL_GL_DestroyContext(context);
   SDL_DestroyWindow(window);
   SDL_Quit();
   printf(failures ? "FAIL\n" : "PASS\n");
   return failures != 0;
}

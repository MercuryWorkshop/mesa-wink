/*
 * Copyright © 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "main/glthread_marshal.h"
#include "dispatch.h"

/* Driver thread (or the application thread with glthread idle): the fog state
 * glthread lost track of, from the context itself. */
static void
glthread_fetch_fog(struct gl_context *ctx, void *unused)
{
   struct glthread_fog *fog = &ctx->GLThread.Fog;
   fog->enabled = ctx->Fog.Enabled;
   fog->mode = ctx->Fog.Mode;
   fog->start = ctx->Fog.Start;
   fog->end = ctx->Fog.End;
   fog->density = ctx->Fog.Density;
   memcpy(fog->color, ctx->Fog.ColorUnclamped, sizeof(fog->color));
   fog->known = true;
}

/* Whether this context's fog state can answer a query here. Fog is gone from
 * core and ES 2+, where a query is an error to report; elsewhere a state lost
 * to a display list is fetched once, in one wait, and kept. */
static bool
glthread_fog_ready(struct gl_context *ctx)
{
   if (ctx->GLThread.inside_begin_end || ctx->API == API_OPENGL_CORE ||
       ctx->API == API_OPENGLES2)
      return false;
   if (ctx->GLThread.Fog.known)
      return true;
#if MESA_GLTHREAD_REMOTE
   if (GLTHREAD_IS_REMOTE(ctx)) {
      ctx->GLThread.sync_why = "fog state, refetched";
      _mesa_glthread_call(ctx, glthread_fetch_fog, NULL);
      return ctx->GLThread.Fog.known;
   }
#endif
   _mesa_glthread_finish_before(ctx, "fog state, refetched");
   glthread_fetch_fog(ctx, NULL);
   return true;
}

static bool
glthread_is_fog_pname(GLenum pname)
{
   switch (pname) {
   case GL_FOG:
   case GL_FOG_MODE:
   case GL_FOG_START:
   case GL_FOG_END:
   case GL_FOG_DENSITY:
   case GL_FOG_COLOR:
      return true;
   default:
      return false;
   }
}

/* The fog query as floats; GetBooleanv and GetIntegerv convert. */
static void
glthread_fog_values(struct gl_context *ctx, GLenum pname, GLfloat *out,
                    unsigned *count)
{
   const struct glthread_fog *fog = &ctx->GLThread.Fog;
   *count = 1;
   switch (pname) {
   case GL_FOG:       out[0] = fog->enabled ? 1.0f : 0.0f; break;
   case GL_FOG_MODE:  out[0] = (GLfloat)fog->mode; break;
   case GL_FOG_START: out[0] = fog->start; break;
   case GL_FOG_END:   out[0] = fog->end; break;
   case GL_FOG_DENSITY: out[0] = fog->density; break;
   case GL_FOG_COLOR:
      for (unsigned i = 0; i < 4; ++i)
         out[i] = CLAMP(fog->color[i], 0.0f, 1.0f);
      *count = 4;
      break;
   }
}

uint32_t
_mesa_unmarshal_GetFloatv(struct gl_context *ctx,
                          const struct marshal_cmd_GetFloatv *restrict cmd)
{
   UNREACHABLE("never executed");
   return 0;
}

void GLAPIENTRY
_mesa_marshal_GetFloatv(GLenum pname, GLfloat *p)
{
   GET_CURRENT_CONTEXT(ctx);

   if (glthread_is_fog_pname(pname) && glthread_fog_ready(ctx)) {
      unsigned n;
      glthread_fog_values(ctx, pname, p, &n);
      return;
   }
#if MESA_GLTHREAD_REMOTE
   if (GLTHREAD_IS_REMOTE(ctx)) {
      ctx->GLThread.sync_why = _mesa_glthread_remote_pname_name("GetFloatv", pname);
      _mesa_glthread_call_GetFloatv(ctx, pname, p);
      return;
   }
#endif
   _mesa_glthread_finish_before(ctx, "GetFloatv");
   CALL_GetFloatv(ctx->Dispatch.Current, (pname, p));
}

uint32_t
_mesa_unmarshal_GetBooleanv(struct gl_context *ctx,
                            const struct marshal_cmd_GetBooleanv *restrict cmd)
{
   UNREACHABLE("never executed");
   return 0;
}

void GLAPIENTRY
_mesa_marshal_GetBooleanv(GLenum pname, GLboolean *p)
{
   GET_CURRENT_CONTEXT(ctx);

   if (glthread_is_fog_pname(pname) && glthread_fog_ready(ctx)) {
      GLfloat v[4];
      unsigned n;
      glthread_fog_values(ctx, pname, v, &n);
      for (unsigned i = 0; i < n; ++i)
         p[i] = v[i] != 0.0f ? GL_TRUE : GL_FALSE;
      return;
   }
#if MESA_GLTHREAD_REMOTE
   if (GLTHREAD_IS_REMOTE(ctx)) {
      ctx->GLThread.sync_why = _mesa_glthread_remote_pname_name("GetBooleanv", pname);
      _mesa_glthread_call_GetBooleanv(ctx, pname, p);
      return;
   }
#endif
   _mesa_glthread_finish_before(ctx, "GetBooleanv");
   CALL_GetBooleanv(ctx->Dispatch.Current, (pname, p));
}

uint32_t
_mesa_unmarshal_GetIntegerv(struct gl_context *ctx,
                            const struct marshal_cmd_GetIntegerv *restrict cmd)
{
   UNREACHABLE("never executed");
   return 0;
}

void GLAPIENTRY
_mesa_marshal_GetIntegerv(GLenum pname, GLint *p)
{
   GET_CURRENT_CONTEXT(ctx);

   /* This will generate GL_INVALID_OPERATION, as it should. */
   if (ctx->GLThread.inside_begin_end)
      goto sync;

   /* TODO: Use get_hash_params.py to return values for items containing:
    * - CONST(
    * - CONTEXT_[A-Z]*(Const
    */

   switch (pname) {
   case GL_ACTIVE_TEXTURE:
      *p = GL_TEXTURE0 + ctx->GLThread.ActiveTexture;
      return;
   case GL_ARRAY_BUFFER_BINDING:
      *p = ctx->GLThread.CurrentArrayBufferName;
      return;
   case GL_ATTRIB_STACK_DEPTH:
      *p = ctx->GLThread.AttribStackDepth;
      return;
   case GL_CLIENT_ACTIVE_TEXTURE:
      *p = GL_TEXTURE0 + ctx->GLThread.ClientActiveTexture;
      return;
   case GL_CLIENT_ATTRIB_STACK_DEPTH:
      *p = ctx->GLThread.ClientAttribStackTop;
      return;
   case GL_CURRENT_PROGRAM:
      *p = ctx->GLThread.CurrentProgram;
      return;
   case GL_DRAW_INDIRECT_BUFFER_BINDING:
      *p = ctx->GLThread.CurrentDrawIndirectBufferName;
      return;
   case GL_DRAW_FRAMEBUFFER_BINDING:
      *p = ctx->GLThread.CurrentDrawFramebuffer;
      return;
   case GL_READ_FRAMEBUFFER_BINDING:
      *p = ctx->GLThread.CurrentReadFramebuffer;
      return;
   case GL_PIXEL_PACK_BUFFER_BINDING:
      *p = ctx->GLThread.CurrentPixelPackBufferName;
      return;
   case GL_PIXEL_UNPACK_BUFFER_BINDING:
      *p = ctx->GLThread.CurrentPixelUnpackBufferName;
      return;
   case GL_QUERY_BUFFER_BINDING:
      *p = ctx->GLThread.CurrentQueryBufferName;
      return;

   case GL_MATRIX_MODE:
      *p = ctx->GLThread.MatrixMode;
      return;
   case GL_CURRENT_MATRIX_STACK_DEPTH_ARB:
      *p = ctx->GLThread.MatrixStackDepth[ctx->GLThread.MatrixIndex] + 1;
      return;
   case GL_MODELVIEW_STACK_DEPTH:
      *p = ctx->GLThread.MatrixStackDepth[M_MODELVIEW] + 1;
      return;
   case GL_PROJECTION_STACK_DEPTH:
      *p = ctx->GLThread.MatrixStackDepth[M_PROJECTION] + 1;
      return;
   case GL_TEXTURE_STACK_DEPTH:
      *p = ctx->GLThread.MatrixStackDepth[M_TEXTURE0 + ctx->GLThread.ActiveTexture] + 1;
      return;

   case GL_VERTEX_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_POS)) != 0;
      return;
   case GL_NORMAL_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_NORMAL)) != 0;
      return;
   case GL_COLOR_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_COLOR0)) != 0;
      return;
   case GL_SECONDARY_COLOR_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_COLOR1)) != 0;
      return;
   case GL_FOG_COORD_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_FOG)) != 0;
      return;
   case GL_INDEX_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_COLOR_INDEX)) != 0;
      return;
   case GL_EDGE_FLAG_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_EDGEFLAG)) != 0;
      return;
   case GL_TEXTURE_COORD_ARRAY:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled &
            (1 << (VERT_ATTRIB_TEX0 + ctx->GLThread.ClientActiveTexture))) != 0;
      return;
   case GL_POINT_SIZE_ARRAY_OES:
      *p = (ctx->GLThread.CurrentVAO->UserEnabled & (1 << VERT_ATTRIB_POINT_SIZE)) != 0;
      return;
   case GL_VERTEX_ARRAY_BINDING:
      *p = ctx->GLThread.CurrentVAO->Name;
      return;
   case GL_FOG:
   case GL_FOG_MODE:
      if (glthread_fog_ready(ctx)) {
         *p = pname == GL_FOG ? ctx->GLThread.Fog.enabled
                              : ctx->GLThread.Fog.mode;
         return;
      }
      break;
   }

sync:
#if MESA_GLTHREAD_REMOTE
   /* The census names this wait by pname: which queries glthread cannot answer
    * itself is the whole question about them. */
   if (GLTHREAD_IS_REMOTE(ctx)) {
      ctx->GLThread.sync_why = _mesa_glthread_remote_pname_name("GetIntegerv", pname);
      _mesa_glthread_call_GetIntegerv(ctx, pname, p);
   } else
#endif
   {
      _mesa_glthread_finish_before(ctx, "GetIntegerv");
      CALL_GetIntegerv(ctx->Dispatch.Current, (pname, p));
   }
}

/* TODO: Implement glGetBooleanv, glGetFloatv, etc. if needed */

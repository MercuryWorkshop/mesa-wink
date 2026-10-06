/* SPDX-License-Identifier: MIT
 * A real EGL/GL API capture proof, with two distinct frames and live readback. */
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int desktop = argc > 1 && !strcmp(argv[1], "gl");
    EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;
    if (!eglInitialize(d, &major, &minor)) return 1;
    if (!eglBindAPI(desktop ? EGL_OPENGL_API : EGL_OPENGL_ES_API)) return 2;
    EGLint attrs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
        desktop ? EGL_OPENGL_BIT : EGL_OPENGL_ES3_BIT, EGL_RED_SIZE,8,
        EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_DEPTH_SIZE,24,
        EGL_STENCIL_SIZE,8,EGL_NONE };
    EGLConfig cfg; EGLint count;
    if (!eglChooseConfig(d,attrs,&cfg,1,&count) || !count) return 3;
    EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION,3, EGL_CONTEXT_MINOR_VERSION,desktop ? 3 : 0,
        EGL_NONE };
    EGLContext c = eglCreateContext(d,cfg,EGL_NO_CONTEXT,ca);
    EGLint sa[] = { EGL_WIDTH,64,EGL_HEIGHT,48,EGL_NONE };
    EGLSurface s = eglCreatePbufferSurface(d,cfg,sa);
    if (!eglMakeCurrent(d,s,s,c)) return 4;
    printf("renderer: %s\nversion: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    const char *vs = desktop ? "#version 330\nvoid main(){vec2 p[3]=vec2[3](vec2(-.8,-.7),vec2(.8,-.7),vec2(0,.8));gl_Position=vec4(p[gl_VertexID],0,1);}" :
        "#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-.8,-.7),vec2(.8,-.7),vec2(0,.8));gl_Position=vec4(p[gl_VertexID],0,1);}";
    const char *fs = desktop ? "#version 330\nout vec4 o;void main(){o=vec4(.1,.8,.3,1);}" :
        "#version 300 es\nprecision highp float;out vec4 o;void main(){o=vec4(.1,.8,.3,1);}";
    GLuint v=glCreateShader(GL_VERTEX_SHADER), f=glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v,1,&vs,NULL); glCompileShader(v);
    glShaderSource(f,1,&fs,NULL); glCompileShader(f);
    GLuint program=glCreateProgram(); glAttachShader(program,v); glAttachShader(program,f); glLinkProgram(program);
    GLint ok; glGetProgramiv(program,GL_LINK_STATUS,&ok); if (!ok) return 5;
    GLuint vao; glGenVertexArrays(1,&vao); glBindVertexArray(vao); glUseProgram(program);
    glViewport(0,0,64,48);
    /* Upstream retrace sizes EGL drawables lazily on the first draw. Keep
     * setup/warmup in the trace and verify both subsequent live frames. */
    for(int frame=1;frame<=4;frame++) {
        glClearColor(frame%2?.2:.8,.1,.4,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES,0,3);
        unsigned char pixels[64*48*4]; glReadPixels(0,0,64,48,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
        char name[64]; snprintf(name,sizeof name,"live-%d.ppm",frame-2);
        if (frame<3) { eglSwapBuffers(d,s); continue; }
        FILE *out=fopen(name,"wb"); if(!out) return 6;
        fprintf(out,"P6\n64 48\n255\n");
        for(int y=47;y>=0;y--) for(int x=0;x<64;x++) fwrite(pixels+(y*64+x)*4,1,3,out);
        fclose(out);
        if (glGetError()!=GL_NO_ERROR) return 7;
        eglSwapBuffers(d,s);
    }
    eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
    eglDestroyContext(d,c); eglDestroySurface(d,s); eglTerminate(d);
    return 0;
}

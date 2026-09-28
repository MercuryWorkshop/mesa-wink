#version 450
/* A quad from a vertex buffer, moved by a push constant. */
layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 color;
layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_uv;
layout(push_constant) uniform Push {
   vec4 tint;
   vec2 offset;
} pc;
void main()
{
   v_color = color;
   v_uv = pos * 0.5 + 0.5;
   gl_Position = vec4(pos + pc.offset, 0.0, 1.0);
}

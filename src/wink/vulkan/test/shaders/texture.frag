#version 450
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 o;
layout(set = 0, binding = 0) uniform Scale {
   vec4 scale;
} u;
layout(set = 0, binding = 1) uniform sampler2D tex;
void main()
{
   o = texture(tex, v_uv) * u.scale;
}

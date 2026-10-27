#version 450
layout(location = 0) out vec4 out_color;
layout(push_constant) uniform Push { vec4 tint; vec2 offset; } pc;
void main() { out_color = pc.tint; }

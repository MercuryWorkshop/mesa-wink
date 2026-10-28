#version 450
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInputMS input_color;
layout(location = 0) out vec4 out_color;
void main() { out_color = vec4(subpassLoad(input_color, 0).gr, 0, 1); }

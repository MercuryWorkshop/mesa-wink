#version 460
/* Draw parameters and a divisor, as Minecraft's shaders need them: glslang
 * makes gl_VertexID VertexIndex - BaseVertex. A cell per instance, four to
 * a row, a row per draw; its colour an attribute with a divisor. */
layout(location = 0) in vec4 inst_color;
layout(location = 0) out vec3 v_color;
const vec2 corners[4] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1));
void main()
{
   const vec2 c = corners[gl_VertexIndex - gl_BaseVertex];
   const float column = float(gl_InstanceIndex - gl_BaseInstance);
   const vec2 p = vec2((column + c.x) / 4.0, (float(gl_DrawID) + c.y) / 2.0);
   v_color = inst_color.rgb;
   gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

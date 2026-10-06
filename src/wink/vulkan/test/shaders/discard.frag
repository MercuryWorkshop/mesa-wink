#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) out vec4 color;
void main() {
   if (gl_FragCoord.x < 16.0) discard;
   color = vec4(1.0);
}

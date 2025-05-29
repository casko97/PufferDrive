#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;       // Fragment input: vertex attribute: texture coordinates
in vec4 fragColor;          // Fragment input: vertex attribute: color 
in vec3 fragPosition;       // Fragment input: vertex attribute: position

// Input uniform values
uniform sampler2D texture0; // Fragment input: texture

// Output fragment color
out vec4 finalColor;       // Fragment output: pixel color

float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
    // Color based on height (e.g., gradient from blue to red)
    float height = fragPosition.y;
    if (height < 5.0) {
        float val = 0.5 + height/10.0;
        finalColor.rgba = vec4(val, 0.0, 0.0, 1.0);
    } else if (height < 15.0) {
        float val = 0.5 + (height - 5.0)/5.0;
        finalColor.rgba = vec4(val, val/4.0, 0.0, 1.0);
    } else {
        float val = 0.5 + (height - 15.0)/(32.0 - 15.0)/2.0;
        finalColor.rgba = vec4(val, val/2.0, 0.0, 1.0);
    }
}


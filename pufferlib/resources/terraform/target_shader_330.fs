#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;       // Fragment input: vertex attribute: texture coordinates
in vec4 fragColor;          // Fragment input: vertex attribute: color 
in vec3 fragPosition;       // Fragment input: vertex attribute: position

// Input uniform values
uniform sampler2D texture0; // Fragment input: texture

// Output fragment color
out vec4 finalColor;       // Fragment output: pixel color

void main()
{
    // Color based on height (e.g., gradient from blue to red)
    float height = fragPosition.y/32.0;

    float black = 0.25 - 0.25*fract(20.0*height);
    vec4 start_color = vec4(0.0, 0.5, 0.0, 0.4);
    vec4 end_color = vec4(0.0, 1.0, 1.0, 0.4);
    finalColor.rgba = mix(start_color, end_color, height);

    finalColor.rgb += black;
    if (height < 0.001) {
        finalColor.rgb = start_color.rgb;
    }
}


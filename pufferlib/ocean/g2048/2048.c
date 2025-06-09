#include "2048.h"
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <time.h>

// Helper to get a single character from stdin without waiting for Enter
int getch() {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

// Map key to action
int key_to_action(int key) {
    if (key == 'w' || key == 'W' || key == 65) return UP;
    if (key == 's' || key == 'S' || key == 66) return DOWN;
    if (key == 'a' || key == 'A' || key == 68) return LEFT;
    if (key == 'd' || key == 'D' || key == 67) return RIGHT;
    return 0;
}

int main() {
    srand(time(NULL));
    Game env;
    float observations[SIZE * SIZE] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    c_reset(&env);

    // Main game loop
    while (1) {
        c_render(&env);

        int action = 0;
        if (IsWindowReady()) {
            if (IsKeyPressed(KEY_W) || IsKeyPressed(KEY_UP)) action = UP;
            else if (IsKeyPressed(KEY_S) || IsKeyPressed(KEY_DOWN)) action = DOWN;
            else if (IsKeyPressed(KEY_A) || IsKeyPressed(KEY_LEFT)) action = LEFT;
            else if (IsKeyPressed(KEY_D) || IsKeyPressed(KEY_RIGHT)) action = RIGHT;
        } else {
            printf("Move (WASD or arrows): ");
            int key = getch();
            action = key_to_action(key);
        }

        if (action != 0) {
            env.actions[0] = action;
            c_step(&env);
            if (!IsWindowReady()) {
                print_grid(&env);
                printf("Reward: %.0f\n", env.rewards[0]);
            }
        }

        if (IsWindowReady() && WindowShouldClose()) {
            break;
        }
    }

    c_close(&env);
    printf("Game Over! Final Max Tile: %d\n", env.score);
    return 0;
}
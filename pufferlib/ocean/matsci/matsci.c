#include "matsci.h"

int main() {
    Matsci env = {};
    env.observations = (float*)calloc(3, sizeof(float));
    env.actions = (float*)calloc(3, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = rndf(-0.05f, 0.05f);
        env.actions[1] = rndf(-0.05f, 0.05f);
        env.actions[2] = rndf(-0.05f, 0.05f);
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}


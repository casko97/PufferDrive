#include <time.h>
#include "freeway.h"
#include "puffernet.h"
#include<unistd.h>

int main() {
    Weights* weights = load_weights("resources/freeway/puffer_freeway_weights.bin", 137092);
    LinearLSTM* net = make_linearlstm(weights, 1, 34, 3);

    Freeway env = {
        .frameskip=4,
        .width=1216,
        .height=720,
        .player_width=56,
        .player_height=32,
        .car_width=64,
        .car_height=40,
        .lane_size=64,
        .difficulty=0,
        .level=0,
        .use_dense_rewards=1,
        .env_randomization=1,
        .enable_human_player=0,
    };
    allocate(&env);

    env.client = make_client(&env);

    c_reset(&env);
    while (!WindowShouldClose()) {
        env.human_actions[0] = 0.0;
        int* actions = (int*)env.actions;
        forward_linearlstm(net, env.observations, actions);
        env.actions[0] = actions[0];
        if (IsKeyDown(KEY_UP)  || IsKeyDown(KEY_A)) env.human_actions[0] = 1;
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_D)) env.human_actions[0] = 2;
        c_step(&env);
        c_render(&env);
        
    }
    free_allocated(&env);
    close_client(env.client);
}

#include "target.h"
#include "puffernet.h"

void allocate(Target* env) {
}

void free_allocated(Target* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

int main() {
    //Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Target env = {
        .width = 1080,
        .height = 720,
        .num_agents = 8,
        .num_goals = 8
    };
    init(&env);
    env.observations = calloc(env.num_agents*(2*(env.num_agents + env.num_goals)+1), sizeof(float));
    env.actions = calloc(2*env.num_agents, sizeof(float));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
        } else {
            for (int i=0; i<env.num_agents; i++) {
                env.actions[i] = (float)rand()/(float)RAND_MAX - 0.5f ;
            }
            //forward_linearlstm(net, env.observations, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }
    //free_linearlstm(net);
    //free(weights);
    //close_client(client);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}


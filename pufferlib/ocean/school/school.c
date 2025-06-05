/* Pure C demo file for School. Build it with:
 * bash scripts/build_ocean.sh school local (debug)
 * bash scripts/build_ocean.sh school fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "school.h"

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"

int main() {
    // Weights are exported by running puffer export
    //Weights* weights = load_weights("resources/puffer_school_weights.bin", 137743);

    //int logit_sizes[2] = {9, 5};
    //LinearLSTM* net = make_linearlstm(weights, num_agents, num_obs, logit_sizes, 2);

    School env = {
        .width = 1980,
        .height = 1020,
        .size_x = 1,
        .size_y = 1, .size_z = 1,
        .num_agents = 1024,
        .num_factories = 32,
        .num_resources = 8,
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    int num_obs = 3*env.num_resources + 10 + env.num_resources;
    env.observations = calloc(env.num_agents*num_obs, sizeof(float));
    env.actions = calloc(3*env.num_agents, sizeof(int));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    // while(True) will break web builds
    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            Entity* agent = &env.agents[i];
            int item = agent->item;
            float dx = env.observations[num_obs*i + 3*item];
            float dy = env.observations[num_obs*i + 3*item + 1];
            float dz = env.observations[num_obs*i + 3*item + 2];
            env.actions[3*i] = (dx > 0.0f) ? 6 : 2;
            env.actions[3*i + 1] = (dy > 0.0f) ? 6 : 2;
            env.actions[3*i + 2] = (dz > 0.0f) ? 6 : 2;
            //float dpitch = atan2f(dz, sqrtf(dx*dx + dy*dy));
            //float droll = asinf(dz/sqrtf(dx*dx + dy*dy + dz*dz));
            //env.actions[3*i] = 6;
            //env.actions[3*i + 1] = (dpitch > 0.0f) ? 6 : 2;
            //env.actions[3*i + 2] = (droll > 0.0f) ? 6 : 2;
            env.actions[3*i] = rand() % 9;
            env.actions[3*i + 1] = rand() % 9;
            env.actions[3*i + 2] = rand() % 9;
            //env.actions[3*i] = 4.0f;
            //env.actions[3*i + 1] = 4.0f;
            //env.actions[3*i + 2] = 4.0f;
        }

        //forward_linearlstm(net, env.observations, env.actions);
        compute_observations(&env);
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    //free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}


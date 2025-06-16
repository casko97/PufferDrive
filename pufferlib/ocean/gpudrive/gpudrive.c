#include <time.h>
#include <unistd.h>
#include "gpudrive.h"

void demo() {
    
    GPUDrive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
	    .map_name = "resources/gpudrive/binaries/map_942.bin",
        .spawn_immunity_timer = 30
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    //Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int(*)[2])env.actions;
        actions[env.human_agent_idx][0] = 3;
        actions[env.human_agent_idx][1] = 6;
        if(IsKeyDown(KEY_UP)){
            actions[env.human_agent_idx][0] += accel_delta;
            // Cap acceleration to maximum of 6
            if(actions[env.human_agent_idx][0] > 6) {
                actions[env.human_agent_idx][0] = 6;
            }
        }
        if(IsKeyDown(KEY_DOWN)){
            actions[env.human_agent_idx][0] -= accel_delta;
            // Cap acceleration to minimum of 0
            if(actions[env.human_agent_idx][0] < 0) {
                actions[env.human_agent_idx][0] = 0;
            }
        }
        if(IsKeyDown(KEY_LEFT)){
            actions[env.human_agent_idx][1] += steer_delta;
            // Cap steering to minimum of 0
            if(actions[env.human_agent_idx][1] < 0) {
                actions[env.human_agent_idx][1] = 0;
            }
        }
        if(IsKeyDown(KEY_RIGHT)){
            actions[env.human_agent_idx][1] -= steer_delta;
            // Cap steering to maximum of 12
            if(actions[env.human_agent_idx][1] > 12) {
                actions[env.human_agent_idx][1] = 12;
            }
        }   
        if(IsKeyPressed(KEY_TAB)){
            env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
        }
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    GPUDrive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
	    .map_name = "resources/gpudrive/binaries/map_055.bin"
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;
    
    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel;  // -1, 0, or 1
            actions[j][1] = steer;  // Random steering
        }
        
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    demo();
    // performance_test();
    return 0;
}

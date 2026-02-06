#include <stdio.h>
#include <stdlib.h>
#include "pufferlib/ocean/drive/drive.h"
#include "pufferlib/ocean/drive/drivenet.h"
#include "pufferlib/ocean/env_config.h"

int main() {
    printf("=== Testing Action Space Support ===\n\n");
    
    // Parse configuration
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s\n", ini_file);
        return -1;
    }
    
    printf("Config loaded:\n");
    printf("  action_type: %d (%s)\n", conf.action_type, conf.action_type == 0 ? "discrete" : "continuous");
    printf("  dynamics_model: %d (%s)\n\n", conf.dynamics_model, conf.dynamics_model == 0 ? "classic" : "jerk");
    
    // Test loading weights
    const char *weights_file = "pufferlib/resources/drive/puffer_drive_weights.bin";
    printf("Loading weights from: %s\n", weights_file);
    Weights *weights = load_weights(weights_file);
    if (!weights) {
        fprintf(stderr, "Error: Failed to load weights\n");
        return -1;
    }
    printf("Weights loaded successfully: %d floats\n\n", weights->size);
    
    // Initialize network with discrete actions (default in config)
    int num_agents = 1;
    printf("Initializing DriveNet with:\n");
    printf("  num_agents: %d\n", num_agents);
    printf("  dynamics_model: %d\n", conf.dynamics_model);
    printf("  action_type: %d\n\n", conf.action_type);
    
    DriveNet *net = init_drivenet(weights, num_agents, conf.dynamics_model, conf.action_type);
    if (!net) {
        fprintf(stderr, "Error: Failed to initialize DriveNet\n");
        free(weights);
        return -1;
    }
    
    printf("\nDriveNet initialized successfully!\n");
    printf("  action_type: %d\n", net->action_type);
    printf("  action_dim: %d\n", net->action_dim);
    printf("  ego_dim: %d\n\n", net->ego_dim);
    
    // Test forward pass with dummy observations
    int ego_dim = net->ego_dim;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    float *observations = (float *)calloc(num_agents * max_obs, sizeof(float));
    
    // Fill with some dummy data
    for (int i = 0; i < num_agents * max_obs; i++) {
        observations[i] = 0.1f * (i % 10);
    }
    
    printf("Running forward pass...\n");
    
    if (net->action_type == 0) {  // Discrete
        int *actions = (int *)calloc(num_agents, sizeof(int));
        forward(net, observations, actions);
        printf("Discrete action output: %d\n", actions[0]);
        free(actions);
    } else {  // Continuous
        float *actions = (float *)calloc(num_agents * net->action_dim, sizeof(float));
        forward(net, observations, actions);
        printf("Continuous action output: [%.4f, %.4f]\n", actions[0], actions[1]);
        free(actions);
    }
    
    printf("\n=== Test completed successfully! ===\n");
    
    // Cleanup
    free(observations);
    free_drivenet(net);
    free(weights);
    
    return 0;
}

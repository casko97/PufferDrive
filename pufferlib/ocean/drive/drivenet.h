#include <time.h>
#include "drive.h"
#include "puffernet.h"
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NN_INPUT_SIZE 64
#define NN_HIDDEN_SIZE 256

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int raw_ego_dim;
    int raw_partner_features;
    int ego_encoder_input_dim;
    int partner_encoder_input_dim;
    int sim_obs_dim;
    int policy_obs_dim;
    int observation_mode;
    int action_type;  // 0 = discrete, 1 = continuous
    int action_dim;   // Number of action dimensions
    float *policy_observations;
    float *obs_self;
    float *obs_partner;
    float *obs_road;
    float *partner_linear_output;
    float *road_linear_output;
    float *partner_layernorm_output;
    float *road_layernorm_output;
    float *partner_linear_output_two;
    float *road_linear_output_two;
    Linear *ego_encoder;
    Linear *road_encoder;
    Linear *partner_encoder;
    LayerNorm *ego_layernorm;
    LayerNorm *road_layernorm;
    LayerNorm *partner_layernorm;
    Linear *ego_encoder_two;
    Linear *road_encoder_two;
    Linear *partner_encoder_two;
    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1;
    CatDim1 *cat2;
    GELU *gelu;
    Linear *shared_embedding;
    ReLU *relu;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    Multidiscrete *multidiscrete;
};

DriveNet *init_drivenet(Weights *weights, int num_agents, int dynamics_model, int action_type, int observation_mode,
                       int extend_classic_action_space) {
    DriveNet *net = calloc(1, sizeof(DriveNet));
    int base_ego_dim = (dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int augmented_ego_features = (observation_mode == 1 ? (PARTNER_TYPE_CHANNELS + EGO_TRAILER_STATE_FEATURES) : 0);
    int raw_ego_dim = base_ego_dim + augmented_ego_features;
    int raw_partner_features = PARTNER_FEATURES + (observation_mode == 1 ? PARTNER_TYPE_CHANNELS : 0);
    int ego_encoder_input_dim =
        base_ego_dim + (observation_mode == 1 ? (EGO_TRAILER_STATE_FEATURES + POLICY_TYPE_CLASS_COUNT) : 0);
    int partner_encoder_input_dim = PARTNER_FEATURES + (observation_mode == 1 ? POLICY_REAL_TYPE_CLASS_COUNT : 0);
    int road_features = ROAD_FEATURES;
    int input_size = NN_INPUT_SIZE;
    int hidden_size = NN_HIDDEN_SIZE;
    int road_feat_onehot = road_features + 6; // one-hot extra 6 features for road

    net->action_type = action_type;
    net->observation_mode = observation_mode;
    
    // Determine action space size based on dynamics model and action type
    int action_size, logit_sizes[2];
    int action_dim;
    if (action_type == 1) {  // Continuous
        action_size = 2;  // accel/jerk_long + steer/jerk_lat
        action_dim = 2;
    } else {  // Discrete
        if (dynamics_model == CLASSIC || dynamics_model == ARTICULATED) {
            int acceleration_count =
                extend_classic_action_space ? (int)(sizeof(ACCELERATION_VALUES_EXTENDED) / sizeof(float))
                                            : (int)(sizeof(ACCELERATION_VALUES_LEGACY) / sizeof(float));
            action_size = acceleration_count * (int)(sizeof(STEERING_VALUES) / sizeof(float));
            logit_sizes[0] = action_size;
            action_dim = 1;
        } else {                 // JERK
            action_size = 4 * 3; // Joint action space (4 longitudinal × 3 lateral = 12)
            logit_sizes[0] = 4 * 3;
            action_dim = 1;
        }
    }
    net->action_dim = action_dim;

    net->num_agents = num_agents;
    net->raw_ego_dim = raw_ego_dim;
    net->raw_partner_features = raw_partner_features;
    net->ego_encoder_input_dim = ego_encoder_input_dim;
    net->partner_encoder_input_dim = partner_encoder_input_dim;
    net->sim_obs_dim =
        base_ego_dim + PARTNER_FEATURES * max_partners + ROAD_FEATURES * max_road_obs;
    net->policy_obs_dim =
        raw_ego_dim + raw_partner_features * max_partners + ROAD_FEATURES * max_road_obs;
    net->policy_observations = calloc(num_agents * net->policy_obs_dim, sizeof(float));
    net->obs_self = calloc(num_agents * ego_encoder_input_dim, sizeof(float));
    net->obs_partner = calloc(num_agents * max_partners * partner_encoder_input_dim, sizeof(float));
    net->obs_road = calloc(num_agents * max_road_obs * road_feat_onehot, sizeof(float));
    net->partner_linear_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));

    net->ego_encoder = make_linear(weights, num_agents, ego_encoder_input_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->road_encoder = make_linear(weights, num_agents, road_feat_onehot, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_encoder = make_linear(weights, num_agents, partner_encoder_input_dim, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_max = make_max_dim1(num_agents, max_partners, input_size);
    net->road_max = make_max_dim1(num_agents, max_road_obs, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3 * input_size);
    net->shared_embedding = make_linear(weights, num_agents, input_size * 3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, action_size);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, NN_HIDDEN_SIZE);
    memset(net->lstm->state_h, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    memset(net->lstm->state_c, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    
    if (action_type == 0) {  // Discrete only
        net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
    } else {
        net->multidiscrete = NULL;
    }
    
    const char *dynamics_name =
        (dynamics_model == CLASSIC) ? "classic" : ((dynamics_model == ARTICULATED) ? "articulated" : "jerk");
    printf("DriveNet initialized: action_type=%d (%s), action_dim=%d, dynamics_model=%d (%s), observation_mode=%d\n",
           action_type, action_type == 0 ? "discrete" : "continuous", action_dim,
           dynamics_model, dynamics_name, observation_mode);
    
    return net;
}

void free_drivenet(DriveNet *net) {
    free(net->policy_observations);
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->multidiscrete);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

static inline float *prepare_policy_observations(DriveNet *net, Drive *env) {
    if (net->observation_mode != 1) {
        return env->observations;
    }

    int base_ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int sim_partner_dim = max_partners * PARTNER_FEATURES;
    int sim_road_offset = base_ego_dim + sim_partner_dim;
    int sim_road_dim = max_road_obs * ROAD_FEATURES;
    int aug_ego_dim = net->raw_ego_dim;
    int aug_partner_dim = max_partners * net->raw_partner_features;
    int aug_road_offset = aug_ego_dim + aug_partner_dim;
    const float empty_partner_eps = 1e-8f;

    memset(net->policy_observations, 0, net->num_agents * net->policy_obs_dim * sizeof(float));

    for (int b = 0; b < env->active_agent_count; b++) {
        const float *sim_obs = &env->observations[b * net->sim_obs_dim];
        float *policy_obs = &net->policy_observations[b * net->policy_obs_dim];

        memcpy(policy_obs, sim_obs, base_ego_dim * sizeof(float));
        memcpy(&policy_obs[aug_road_offset], &sim_obs[sim_road_offset], sim_road_dim * sizeof(float));

        Entity *ego_entity = &env->entities[env->active_agent_indices[b]];
        int ego_type = map_entity_to_policy_type(env, env->active_agent_indices[b]);
        if (ego_type < 0) {
            ego_type = 0;
        } else if (ego_type >= POLICY_TYPE_CLASS_COUNT) {
            ego_type = POLICY_TYPE_CLASS_COUNT - 1;
        }
        policy_obs[base_ego_dim] = (float)ego_type;
        if (has_valid_ego_trailer_pair(env) && env->active_agent_indices[b] == env->sdc_track_index) {
            float trailer_rel_x = 0.0f, trailer_rel_y = 0.0f;
            float trailer_rel_heading_x = 0.0f, trailer_rel_heading_y = 0.0f;
            get_ego_trailer_obs_features_for_agent(env, env->active_agent_indices[b], &trailer_rel_x, &trailer_rel_y,
                                                   &trailer_rel_heading_x, &trailer_rel_heading_y);
            policy_obs[base_ego_dim + 1] = trailer_rel_x;
            policy_obs[base_ego_dim + 2] = trailer_rel_y;
            policy_obs[base_ego_dim + 3] = trailer_rel_heading_x;
            policy_obs[base_ego_dim + 4] = trailer_rel_heading_y;
        }

        const float *sim_partner_obs = &sim_obs[base_ego_dim];
        float *policy_partner_obs = &policy_obs[aug_ego_dim];
        for (int i = 0; i < max_partners; i++) {
            memcpy(&policy_partner_obs[i * net->raw_partner_features], &sim_partner_obs[i * PARTNER_FEATURES],
                   PARTNER_FEATURES * sizeof(float));
        }

        int partner_slot = 0;
        for (int j = 0; j < MAX_AGENTS && partner_slot < max_partners; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1)
                continue;
            if (env->entities[index].type > 3)
                break;
            if (index == env->active_agent_indices[b])
                continue;
            if (has_valid_ego_trailer_pair(env) && env->active_agent_indices[b] == env->sdc_track_index &&
                index == env->ego_trailer_track_index) {
                continue;
            }

            Entity *other_entity = &env->entities[index];
            if (ego_entity->respawn_timestep != -1 || other_entity->respawn_timestep != -1)
                continue;

            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx * dx + dy * dy);
            if (dist > 2500.0f)
                continue;

            const float *partner_src = &sim_partner_obs[partner_slot * PARTNER_FEATURES];
            int occupied = 0;
            for (int k = 0; k < PARTNER_FEATURES; k++) {
                if (fabsf(partner_src[k]) > empty_partner_eps) {
                    occupied = 1;
                    break;
                }
            }
            if (occupied) {
                int partner_type = map_entity_to_policy_type(env, index);
                if (partner_type < 0) {
                    partner_type = 0;
                } else if (partner_type >= POLICY_TYPE_CLASS_COUNT) {
                    partner_type = POLICY_TYPE_CLASS_COUNT - 1;
                }
                policy_partner_obs[partner_slot * net->raw_partner_features + PARTNER_FEATURES] = (float)partner_type;
            }
            partner_slot++;
        }
    }

    return net->policy_observations;
}

void forward(DriveNet *net, float *observations, void *actions) {
    int raw_ego_dim = net->raw_ego_dim;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int ego_extra_features = (net->observation_mode == 1 ? (PARTNER_TYPE_CHANNELS + EGO_TRAILER_STATE_FEATURES) : 0);
    int base_ego_dim = raw_ego_dim - ego_extra_features;
    int raw_partner_features = net->raw_partner_features;
    int partner_features = net->partner_encoder_input_dim;
    int road_features = ROAD_FEATURES;
    int road_feat_onehot = road_features + 6; // one-hot extra 6 features for road

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * net->ego_encoder_input_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * max_partners * partner_features * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * max_road_obs * road_feat_onehot * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (raw_ego_dim + max_partners * raw_partner_features + max_road_obs * road_features);
        int partner_offset = b_offset + raw_ego_dim;
        int road_offset = b_offset + raw_ego_dim + max_partners * raw_partner_features;

        // Process self observation
        for (int i = 0; i < base_ego_dim; i++) {
            net->obs_self[b * net->ego_encoder_input_dim + i] = observations[b_offset + i];
        }
        if (net->observation_mode == 1) {
            int ego_type = (int)observations[b_offset + base_ego_dim];
            for (int i = 0; i < EGO_TRAILER_STATE_FEATURES; i++) {
                net->obs_self[b * net->ego_encoder_input_dim + base_ego_dim + i] =
                    observations[b_offset + base_ego_dim + 1 + i];
            }
            if (ego_type < 0) {
                ego_type = 0;
            } else if (ego_type >= POLICY_TYPE_CLASS_COUNT) {
                ego_type = POLICY_TYPE_CLASS_COUNT - 1;
            }
            net->obs_self[b * net->ego_encoder_input_dim + base_ego_dim + EGO_TRAILER_STATE_FEATURES + ego_type] = 1.0f;
        }

        // Process partner observation
        for (int i = 0; i < max_partners; i++) {
            float *partner_dst = &net->obs_partner[b * max_partners * partner_features + i * partner_features];
            const float *partner_src = &observations[partner_offset + i * raw_partner_features];
            for (int j = 0; j < PARTNER_FEATURES; j++) {
                partner_dst[j] = partner_src[j];
            }
            if (net->observation_mode == 1) {
                int occupied = 0;
                for (int j = 0; j < PARTNER_FEATURES; j++) {
                    if (fabsf(partner_src[j]) > 1e-8f) {
                        occupied = 1;
                        break;
                    }
                }
                if (occupied) {
                    int partner_type = (int)partner_src[PARTNER_FEATURES];
                    if (partner_type < POLICY_TYPE_VEHICLE_SINGLE) {
                        partner_type = POLICY_TYPE_VEHICLE_SINGLE;
                    } else if (partner_type >= POLICY_TYPE_CLASS_COUNT) {
                        partner_type = POLICY_TYPE_CLASS_COUNT - 1;
                    }
                    partner_dst[PARTNER_FEATURES + (partner_type - POLICY_TYPE_VEHICLE_SINGLE)] = 1.0f;
                }
            }
        }

        // Process road observation
        for (int i = 0; i < MAX_ROAD_SEGMENT_OBSERVATIONS; i++) {
            for (int j = 0; j < 7; j++) {
                net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT + j] =
                    observations[road_offset + i * 7 + j];
            }
            for (int j = 0; j < 7; j++) {
                if (j == observations[road_offset + i * 7 + 6]) {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT +
                                  6 + j] = 1.0f;
                } else {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT +
                                  6 + j] = 0.0f;
                }
            }
        }
    }

    // Forward pass through the network
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            // Get the 7 features for this object
            float *obj_features = &net->obs_partner[b * max_partners * partner_features + obj * partner_features];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                    &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    partner_features, NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *after_first = &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                       &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            // Get the 7 features for this object
            float *obj_features =
                &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                    &net->partner_linear_output_two[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // Process road objects: apply linear to each object individually
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            // Get the 13 features for this object
            float *obj_features = &net->obs_road[b * max_road_obs * ROAD_FEATURES_ONEHOT + obj * ROAD_FEATURES_ONEHOT];
            // Apply linear layer to this object
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                    &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    ROAD_FEATURES_ONEHOT, NN_INPUT_SIZE);
        }
    }

    // Apply layer norm and second linear to each road object
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                       &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(after_first, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    gelu(net->gelu, net->cat2->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get actions based on action type
    if (net->action_type == 0) {  // Discrete
        int *discrete_actions = (int *)actions;
        softmax_multidiscrete(net->multidiscrete, net->actor->output, discrete_actions);
    } else {  // Continuous
        float *continuous_actions = (float *)actions;
        // Use tanh to bound actions to [-1, 1]
        for (int i = 0; i < net->num_agents * net->action_dim; i++) {
            continuous_actions[i] = tanhf(net->actor->output[i]);
        }
    }
}

static inline void forward_drive_env(DriveNet *net, Drive *env, void *actions) {
    float *policy_observations = prepare_policy_observations(net, env);
    forward(net, policy_observations, actions);
}

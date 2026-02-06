/* idm_policy.c - Standalone IDM policy implementation
 * Compile and link with visualizer to use IDM without neural network weights
 */

#include <math.h>
#include <stdio.h>
#include "idm_policy.h"

// Constants from drive.h
#ifndef MAX_AGENTS
#define MAX_AGENTS 32
#endif
#define PARTNER_FEATURES 8
#define ROAD_FEATURES 8
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128
#define MAX_SPEED 100.0f
#define EGO_FEATURES_CLASSIC 8
#define EGO_FEATURES_JERK 11
#define JERK 1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Initialize with default parameters
IDMParams idm_default_params(void) {
    IDMParams params;
    params.desired_velocity = 30.0f;
    params.time_headway = 1.5f;
    params.min_spacing = 2.0f;
    params.max_acceleration = 2.0f;
    params.comfortable_deceleration = 3.0f;
    params.delta = 4.0f;
    params.lateral_gain = 2.0f;
    return params;
}

// IDM forward pass - generates actions from observations
void idm_forward(float *observations, float *actions, int num_agents, 
                 int ego_dim, IDMParams *params) {
    
    if (params == NULL) {
        static IDMParams default_params;
        static int initialized = 0;
        if (!initialized) {
            default_params = idm_default_params();
            initialized = 1;
        }
        params = &default_params;
    }

    int max_partners = MAX_AGENTS - 1;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int obs_size = ego_dim + max_partners * partner_features + 
                   MAX_ROAD_SEGMENT_OBSERVATIONS * road_features;
    
    for (int b = 0; b < num_agents; b++) {
        float *obs = &observations[b * obs_size];
        
        // Extract ego state
        float goal_dx = obs[0] * 200.0f;  // Denormalize
        float goal_dy = obs[1] * 200.0f;
        float ego_speed = obs[3] * MAX_SPEED;
        
        // Find lead vehicle
        float lead_dist = 1e6f;
        float lead_speed = 0.0f;
        int partner_offset = ego_dim;
        
        for (int i = 0; i < max_partners; i++) {
            int idx = partner_offset + i * partner_features;
            float dx = obs[idx + 0] * 50.0f;
            float dy = obs[idx + 1] * 50.0f;
            
            if (dx == 0.0f && dy == 0.0f) continue;
            
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dx > 0.0f && dist < lead_dist) {
                lead_dist = dist;
                lead_speed = obs[idx + 7] * MAX_SPEED;
            }
        }
        
        // IDM longitudinal control
        float delta_v = ego_speed - lead_speed;
        float sqrt_ab = sqrtf(params->max_acceleration * params->comfortable_deceleration);
        float s_star = params->min_spacing + ego_speed * params->time_headway + 
                       (ego_speed * delta_v) / (2.0f * sqrt_ab);
        
        if (s_star < params->min_spacing) {
            s_star = params->min_spacing;
        }
        
        float speed_ratio = ego_speed / params->desired_velocity;
        float accel_free = params->max_acceleration * (1.0f - powf(speed_ratio, params->delta));
        
        float accel_interaction = 0.0f;
        if (lead_dist < 100.0f) {
            float dist_clamped = fmaxf(lead_dist, 0.1f);
            accel_interaction = -params->max_acceleration * powf(s_star / dist_clamped, 2.0f);
        }
        
        float accel = accel_free + accel_interaction;
        
        if (accel < -params->comfortable_deceleration) {
            accel = -params->comfortable_deceleration;
        }
        if (accel > params->max_acceleration) {
            accel = params->max_acceleration;
        }
        
        // Path planning: find nearest road lane ahead that leads toward goal
        float target_x = goal_dx;
        float target_y = goal_dy;
        int road_offset = ego_dim + (MAX_AGENTS - 1) * PARTNER_FEATURES;
        float min_dist = 1e6f;
        float goal_angle = atan2f(goal_dy, goal_dx);
        
        for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
            int idx = road_offset + k * ROAD_FEATURES;
            float rx = obs[idx + 0] * 50.0f;
            float ry = obs[idx + 1] * 50.0f;
            int road_type = (int)obs[idx + 7];
            
            if (rx == 0.0f && ry == 0.0f) continue;
            if (road_type + 4 != 4) continue;  // Only ROAD_LANE (type 4)
            if (rx < 0.0f) continue;  // Only ahead
            
            // Check if lane direction aligns with goal (within 60 degrees)
            float lane_angle = atan2f(ry, rx);
            float angle_diff = fabsf(lane_angle - goal_angle);
            if (angle_diff > M_PI) angle_diff = 2.0f * M_PI - angle_diff;
            if (angle_diff > M_PI / 3.0f) continue;
            
            float dist = sqrtf(rx * rx + ry * ry);
            if (dist < min_dist && dist < 30.0f) {
                min_dist = dist;
                target_x = rx;
                target_y = ry;
            }
        }
        
        // Lateral control toward target
        float target_heading = atan2f(target_y, target_x);
        float heading_error = target_heading;
        
        while (heading_error > M_PI) heading_error -= 2.0f * M_PI;
        while (heading_error < -M_PI) heading_error += 2.0f * M_PI;
        
        float steer = params->lateral_gain * heading_error;
        
        if (steer < -1.0f) steer = -1.0f;
        if (steer > 1.0f) steer = 1.0f;
        
        float accel_norm = accel / params->max_acceleration;
        if (accel_norm < -1.0f) accel_norm = -1.0f;
        if (accel_norm > 1.0f) accel_norm = 1.0f;
        
        actions[b * 2 + 0] = accel_norm;
        actions[b * 2 + 1] = steer;
    }
}

void print_idm_params(IDMParams *params) {
    printf("IDM Parameters:\n");
    printf("  Desired velocity: %.1f m/s\n", params->desired_velocity);
    printf("  Time headway: %.1f s\n", params->time_headway);
    printf("  Min spacing: %.1f m\n", params->min_spacing);
    printf("  Max acceleration: %.1f m/s²\n", params->max_acceleration);
    printf("  Comfortable deceleration: %.1f m/s²\n", params->comfortable_deceleration);
    printf("  Lateral gain: %.1f\n", params->lateral_gain);
}

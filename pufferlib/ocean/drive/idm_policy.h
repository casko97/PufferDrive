/* idm_policy.h - Header for standalone IDM policy */

#ifndef IDM_POLICY_H
#define IDM_POLICY_H

typedef struct {
    float desired_velocity;
    float time_headway;
    float min_spacing;
    float max_acceleration;
    float comfortable_deceleration;
    float delta;
    float lateral_gain;
} IDMParams;

IDMParams idm_default_params(void);
void idm_forward(float *observations, float *actions, int num_agents, int ego_dim, IDMParams *params);
void print_idm_params(IDMParams *params);

#endif // IDM_POLICY_H

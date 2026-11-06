#ifndef PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_DYNAMICS_H
#define PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_DYNAMICS_H

#include <math.h>

#include "tractor_trailer.h"

#define ARTICULATED_MAX_SUBSTEP_SECONDS 0.02f
#define ARTICULATED_MAX_SUBSTEP_DISTANCE 0.25f

typedef struct ArticulatedStepInput {
    float speed_start;
    float speed_end;
    float steering_start;
    float steering_end;
    float dt;
    float max_articulation_angle;
} ArticulatedStepInput;

typedef struct ArticulatedKinematicRates {
    float tractor_axle_vx;
    float tractor_axle_vy;
    float tractor_yaw_rate;
    float hitch_vx;
    float hitch_vy;
    float trailer_axle_vx;
    float trailer_axle_vy;
    float trailer_yaw_rate;
} ArticulatedKinematicRates;

typedef struct ArticulatedEntityVelocities {
    float tractor_vx;
    float tractor_vy;
    float tractor_yaw_rate;
    float trailer_vx;
    float trailer_vy;
    float trailer_yaw_rate;
} ArticulatedEntityVelocities;

static inline float articulated_wrap_angle(float angle) {
    return atan2f(sinf(angle), cosf(angle));
}

static inline float articulated_lerp(float start, float end, float fraction) {
    return start + fraction * (end - start);
}

static inline float articulated_angle(const ArticulatedVehicleState *state) {
    return articulated_wrap_angle(state->trailer_axle.heading - state->tractor_rear_axle.heading);
}

static inline void articulated_reproject_trailer_axle(ArticulatedVehicleState *state,
                                                       const ArticulatedGeometry *geometry) {
    ArticulatedPoint hitch = tractor_connection_point(state->tractor_rear_axle, geometry);
    state->trailer_axle =
        trailer_axle_pose_from_connection(hitch, state->trailer_axle.heading, geometry);
}

static inline void articulated_apply_angle_limit(ArticulatedVehicleState *state,
                                                  const ArticulatedGeometry *geometry,
                                                  float max_articulation_angle) {
    if (max_articulation_angle > 0.0f) {
        float articulation = articulated_angle(state);
        if (articulation > max_articulation_angle) {
            state->trailer_axle.heading =
                articulated_wrap_angle(state->tractor_rear_axle.heading + max_articulation_angle);
        } else if (articulation < -max_articulation_angle) {
            state->trailer_axle.heading =
                articulated_wrap_angle(state->tractor_rear_axle.heading - max_articulation_angle);
        }
    }
    articulated_reproject_trailer_axle(state, geometry);
}

static inline ArticulatedKinematicRates articulated_kinematic_rates(
    const ArticulatedVehicleState *state, const ArticulatedGeometry *geometry, float speed,
    float steering_angle) {
    const float tractor_heading = state->tractor_rear_axle.heading;
    const float trailer_heading = state->trailer_axle.heading;
    const float tractor_cos = cosf(tractor_heading);
    const float tractor_sin = sinf(tractor_heading);
    const float trailer_cos = cosf(trailer_heading);
    const float trailer_sin = sinf(trailer_heading);

    ArticulatedKinematicRates rates = {0};
    rates.tractor_axle_vx = speed * tractor_cos;
    rates.tractor_axle_vy = speed * tractor_sin;
    rates.tractor_yaw_rate = speed * tanf(steering_angle) / geometry->wheelbase;

    // p_h = p_r - a e(psi), including the hitch velocity induced by tractor yaw.
    rates.hitch_vx =
        rates.tractor_axle_vx + geometry->tractor_rear_axle_to_hitch * rates.tractor_yaw_rate * tractor_sin;
    rates.hitch_vy =
        rates.tractor_axle_vy - geometry->tractor_rear_axle_to_hitch * rates.tractor_yaw_rate * tractor_cos;

    const float hitch_dx =
        state->tractor_rear_axle.x - geometry->tractor_rear_axle_to_hitch * tractor_cos -
        state->trailer_axle.x;
    const float hitch_dy =
        state->tractor_rear_axle.y - geometry->tractor_rear_axle_to_hitch * tractor_sin -
        state->trailer_axle.y;
    const float b_squared = geometry->trailer_axle_to_hitch * geometry->trailer_axle_to_hitch;
    rates.trailer_yaw_rate =
        (-hitch_dy * rates.hitch_vx + hitch_dx * rates.hitch_vy) / b_squared;

    const float trailer_longitudinal_speed =
        rates.hitch_vx * trailer_cos + rates.hitch_vy * trailer_sin;
    rates.trailer_axle_vx = trailer_longitudinal_speed * trailer_cos;
    rates.trailer_axle_vy = trailer_longitudinal_speed * trailer_sin;
    return rates;
}

static inline int articulated_substep_count(float dt, float speed_start, float speed_end,
                                             float max_substep_seconds, float max_substep_distance) {
    if (dt == 0.0f)
        return 0;

    if (max_substep_seconds <= 0.0f)
        max_substep_seconds = ARTICULATED_MAX_SUBSTEP_SECONDS;
    if (max_substep_distance <= 0.0f)
        max_substep_distance = ARTICULATED_MAX_SUBSTEP_DISTANCE;

    const float duration = fabsf(dt);
    const float maximum_speed = fmaxf(fabsf(speed_start), fabsf(speed_end));
    const int time_steps = (int)ceilf(duration / max_substep_seconds);
    const int distance_steps = (int)ceilf(maximum_speed * duration / max_substep_distance);
    int substeps = time_steps > distance_steps ? time_steps : distance_steps;
    return substeps > 0 ? substeps : 1;
}

static inline int articulated_integrate_rk2_with_limits(
    ArticulatedVehicleState *state, const ArticulatedGeometry *geometry,
    const ArticulatedStepInput *input, float max_substep_seconds, float max_substep_distance) {
    if (state == NULL || geometry == NULL || input == NULL || geometry->wheelbase <= 0.0f ||
        geometry->trailer_axle_to_hitch <= 0.0f)
        return 0;

    const int substeps = articulated_substep_count(
        input->dt, input->speed_start, input->speed_end, max_substep_seconds, max_substep_distance);
    if (substeps == 0) {
        state->speed = input->speed_end;
        state->steering_angle = input->steering_end;
        articulated_apply_angle_limit(state, geometry, input->max_articulation_angle);
        return 0;
    }

    const float substep_dt = input->dt / (float)substeps;
    for (int step = 0; step < substeps; step++) {
        const float start_fraction = (float)step / (float)substeps;
        const float midpoint_fraction = ((float)step + 0.5f) / (float)substeps;
        const float speed_start =
            articulated_lerp(input->speed_start, input->speed_end, start_fraction);
        const float steering_start =
            articulated_lerp(input->steering_start, input->steering_end, start_fraction);
        const float speed_midpoint =
            articulated_lerp(input->speed_start, input->speed_end, midpoint_fraction);
        const float steering_midpoint =
            articulated_lerp(input->steering_start, input->steering_end, midpoint_fraction);

        ArticulatedKinematicRates start_rates =
            articulated_kinematic_rates(state, geometry, speed_start, steering_start);
        ArticulatedVehicleState midpoint = *state;
        midpoint.tractor_rear_axle.x += 0.5f * substep_dt * start_rates.tractor_axle_vx;
        midpoint.tractor_rear_axle.y += 0.5f * substep_dt * start_rates.tractor_axle_vy;
        midpoint.tractor_rear_axle.heading = articulated_wrap_angle(
            midpoint.tractor_rear_axle.heading + 0.5f * substep_dt * start_rates.tractor_yaw_rate);
        midpoint.trailer_axle.x += 0.5f * substep_dt * start_rates.trailer_axle_vx;
        midpoint.trailer_axle.y += 0.5f * substep_dt * start_rates.trailer_axle_vy;
        midpoint.trailer_axle.heading = articulated_wrap_angle(
            midpoint.trailer_axle.heading + 0.5f * substep_dt * start_rates.trailer_yaw_rate);
        articulated_reproject_trailer_axle(&midpoint, geometry);

        ArticulatedKinematicRates midpoint_rates =
            articulated_kinematic_rates(&midpoint, geometry, speed_midpoint, steering_midpoint);
        state->tractor_rear_axle.x += substep_dt * midpoint_rates.tractor_axle_vx;
        state->tractor_rear_axle.y += substep_dt * midpoint_rates.tractor_axle_vy;
        state->tractor_rear_axle.heading = articulated_wrap_angle(
            state->tractor_rear_axle.heading + substep_dt * midpoint_rates.tractor_yaw_rate);
        state->trailer_axle.x += substep_dt * midpoint_rates.trailer_axle_vx;
        state->trailer_axle.y += substep_dt * midpoint_rates.trailer_axle_vy;
        state->trailer_axle.heading = articulated_wrap_angle(
            state->trailer_axle.heading + substep_dt * midpoint_rates.trailer_yaw_rate);
        articulated_apply_angle_limit(state, geometry, input->max_articulation_angle);
    }

    state->speed = input->speed_end;
    state->steering_angle = input->steering_end;
    return substeps;
}

static inline int articulated_integrate_rk2(ArticulatedVehicleState *state,
                                             const ArticulatedGeometry *geometry,
                                             const ArticulatedStepInput *input) {
    return articulated_integrate_rk2_with_limits(
        state, geometry, input, ARTICULATED_MAX_SUBSTEP_SECONDS,
        ARTICULATED_MAX_SUBSTEP_DISTANCE);
}

static inline ArticulatedEntityVelocities articulated_entity_velocities(
    const ArticulatedVehicleState *state, const ArticulatedGeometry *geometry) {
    ArticulatedKinematicRates rates =
        articulated_kinematic_rates(state, geometry, state->speed, state->steering_angle);
    const float tractor_heading = state->tractor_rear_axle.heading;
    const float trailer_heading = state->trailer_axle.heading;
    return (ArticulatedEntityVelocities){
        .tractor_vx =
            rates.tractor_axle_vx -
            geometry->tractor_axle_to_center * rates.tractor_yaw_rate * sinf(tractor_heading),
        .tractor_vy =
            rates.tractor_axle_vy +
            geometry->tractor_axle_to_center * rates.tractor_yaw_rate * cosf(tractor_heading),
        .tractor_yaw_rate = rates.tractor_yaw_rate,
        .trailer_vx =
            rates.trailer_axle_vx -
            geometry->trailer_axle_to_center * rates.trailer_yaw_rate * sinf(trailer_heading),
        .trailer_vy =
            rates.trailer_axle_vy +
            geometry->trailer_axle_to_center * rates.trailer_yaw_rate * cosf(trailer_heading),
        .trailer_yaw_rate = rates.trailer_yaw_rate,
    };
}

#endif

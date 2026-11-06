#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pufferlib/ocean/drive/drive.h"

#define TEST_EPSILON 1e-4f

static void assert_close(float actual, float expected, float tolerance) {
    assert(fabsf(actual - expected) <= tolerance);
}

static ArticulatedGeometry default_geometry(void) {
    ArticulatedGeometry geometry = {0};
    assert(articulated_geometry_from_params(
               DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry) ==
           ARTICULATED_GEOMETRY_OK);
    return geometry;
}

static ArticulatedVehicleState aligned_state(const ArticulatedGeometry *geometry,
                                             float x, float y, float heading,
                                             float speed, float steering) {
    ArticulatedVehicleState state = {
        .tractor_rear_axle = {.x = x, .y = y, .z = 0.0f, .heading = heading},
        .speed = speed,
        .steering_angle = steering,
    };
    ArticulatedPoint hitch = tractor_connection_point(
        state.tractor_rear_axle, geometry);
    state.trailer_axle =
        trailer_axle_pose_from_connection(hitch, heading, geometry);
    return state;
}

static float hitch_error(const ArticulatedVehicleState *state,
                         const ArticulatedGeometry *geometry) {
    ArticulatedPoint tractor_hitch =
        tractor_connection_point(state->tractor_rear_axle, geometry);
    ArticulatedPoint trailer_hitch =
        trailer_connection_point(state->trailer_axle, geometry);
    return hypotf(tractor_hitch.x - trailer_hitch.x,
                  tractor_hitch.y - trailer_hitch.y);
}

static void test_substep_policy(void) {
    assert(articulated_substep_count(0.1f, 0.0f, 0.0f, 0.02f, 0.25f) == 5);
    assert(articulated_substep_count(0.1f, 30.0f, 30.0f, 0.02f, 0.25f) == 12);
    assert(articulated_substep_count(0.0f, 30.0f, 30.0f, 0.02f, 0.25f) == 0);
}

static void test_analytic_straight_motion(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState state =
        aligned_state(&geometry, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f);
    float initial_trailer_x = state.trailer_axle.x;
    ArticulatedStepInput input = {
        .speed_start = 5.0f,
        .speed_end = 5.0f,
        .steering_start = 0.0f,
        .steering_end = 0.0f,
        .dt = 1.0f,
    };

    assert(articulated_integrate_rk2(&state, &geometry, &input) == 50);
    assert_close(state.tractor_rear_axle.x, 5.0f, TEST_EPSILON);
    assert_close(state.tractor_rear_axle.y, 0.0f, TEST_EPSILON);
    assert_close(state.tractor_rear_axle.heading, 0.0f, TEST_EPSILON);
    assert_close(state.trailer_axle.x, initial_trailer_x + 5.0f, TEST_EPSILON);
    assert_close(state.trailer_axle.y, 0.0f, TEST_EPSILON);
    assert_close(state.trailer_axle.heading, 0.0f, TEST_EPSILON);
    assert(hitch_error(&state, &geometry) <= TEST_EPSILON);
}

static void test_analytic_tractor_turn_and_trailer_following(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState state =
        aligned_state(&geometry, 0.0f, 0.0f, 0.0f, 8.0f, 0.25f);
    const float curvature = tanf(0.25f) / geometry.wheelbase;
    const float theta = 8.0f * curvature;
    ArticulatedStepInput input = {
        .speed_start = 8.0f,
        .speed_end = 8.0f,
        .steering_start = 0.25f,
        .steering_end = 0.25f,
        .dt = 1.0f,
    };

    articulated_integrate_rk2(&state, &geometry, &input);
    assert_close(state.tractor_rear_axle.x, sinf(theta) / curvature, 2e-3f);
    assert_close(state.tractor_rear_axle.y, (1.0f - cosf(theta)) / curvature, 2e-3f);
    assert_close(state.tractor_rear_axle.heading, theta, 2e-4f);
    assert(state.trailer_axle.heading > 0.0f);
    assert(state.trailer_axle.heading < state.tractor_rear_axle.heading);
    assert(hitch_error(&state, &geometry) <= TEST_EPSILON);
}

static void test_signed_reverse_motion(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState state =
        aligned_state(&geometry, 0.0f, 0.0f, 0.0f, -4.0f, 0.3f);
    ArticulatedStepInput input = {
        .speed_start = -4.0f,
        .speed_end = -4.0f,
        .steering_start = 0.3f,
        .steering_end = 0.3f,
        .dt = 0.5f,
    };

    articulated_integrate_rk2(&state, &geometry, &input);
    assert(state.tractor_rear_axle.x < 0.0f);
    assert(state.tractor_rear_axle.heading < 0.0f);
    assert(state.trailer_axle.heading > 0.0f);
    assert(hitch_error(&state, &geometry) <= TEST_EPSILON);
}

static float state_distance(const ArticulatedVehicleState *a,
                            const ArticulatedVehicleState *b) {
    float sum = 0.0f;
    sum += hypotf(a->tractor_rear_axle.x - b->tractor_rear_axle.x,
                  a->tractor_rear_axle.y - b->tractor_rear_axle.y);
    sum += fabsf(articulated_wrap_angle(
        a->tractor_rear_axle.heading - b->tractor_rear_axle.heading));
    sum += hypotf(a->trailer_axle.x - b->trailer_axle.x,
                  a->trailer_axle.y - b->trailer_axle.y);
    sum += fabsf(articulated_wrap_angle(
        a->trailer_axle.heading - b->trailer_axle.heading));
    return sum;
}

static void test_refinement_and_constraint_projection(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState initial =
        aligned_state(&geometry, 1.0f, -2.0f, 0.2f, 12.0f, 0.5f);
    ArticulatedVehicleState coarse = initial;
    ArticulatedVehicleState medium = initial;
    ArticulatedVehicleState reference = initial;
    ArticulatedStepInput input = {
        .speed_start = 12.0f,
        .speed_end = 12.0f,
        .steering_start = 0.5f,
        .steering_end = 0.5f,
        .dt = 1.0f,
    };

    articulated_integrate_rk2_with_limits(
        &coarse, &geometry, &input, 0.1f, 1000.0f);
    articulated_integrate_rk2_with_limits(
        &medium, &geometry, &input, 0.02f, 1000.0f);
    articulated_integrate_rk2_with_limits(
        &reference, &geometry, &input, 0.0025f, 1000.0f);
    assert(state_distance(&medium, &reference) <
           state_distance(&coarse, &reference));

    ArticulatedVehicleState projected = initial;
    for (int step = 0; step < 100; step++) {
        float steering = 0.35f * sinf(0.17f * (float)step);
        ArticulatedStepInput varying_input = {
            .speed_start = 7.0f,
            .speed_end = 7.0f,
            .steering_start = steering,
            .steering_end = steering,
            .dt = 0.1f,
        };
        articulated_integrate_rk2(&projected, &geometry, &varying_input);
        assert(hitch_error(&projected, &geometry) <= TEST_EPSILON);
    }
}

static void test_articulation_limit(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState state =
        aligned_state(&geometry, 0.0f, 0.0f, 0.0f, -6.0f, 0.5f);
    for (int step = 0; step < 20; step++) {
        ArticulatedStepInput input = {
            .speed_start = -6.0f,
            .speed_end = -6.0f,
            .steering_start = 0.5f,
            .steering_end = 0.5f,
            .dt = 0.1f,
            .max_articulation_angle = 0.2f,
        };
        articulated_integrate_rk2(&state, &geometry, &input);
        assert(fabsf(articulated_angle(&state)) <= 0.20001f);
        assert(hitch_error(&state, &geometry) <= TEST_EPSILON);
    }
    assert_close(fabsf(articulated_angle(&state)), 0.2f, 1e-3f);
}

static void test_physically_consistent_entity_velocities(void) {
    ArticulatedGeometry geometry = default_geometry();
    ArticulatedVehicleState state =
        aligned_state(&geometry, 2.0f, -1.0f, 0.4f, 7.0f, 0.3f);
    ArticulatedStepInput input = {
        .speed_start = 7.0f,
        .speed_end = 7.0f,
        .steering_start = 0.3f,
        .steering_end = 0.3f,
        .dt = 0.4f,
    };
    articulated_integrate_rk2(&state, &geometry, &input);

    ArticulatedKinematicRates rates = articulated_kinematic_rates(
        &state, &geometry, state.speed, state.steering_angle);
    float trailer_lateral_velocity =
        -sinf(state.trailer_axle.heading) * rates.trailer_axle_vx +
        cosf(state.trailer_axle.heading) * rates.trailer_axle_vy;
    assert_close(trailer_lateral_velocity, 0.0f, TEST_EPSILON);

    ArticulatedEntityVelocities velocities =
        articulated_entity_velocities(&state, &geometry);
    assert(isfinite(velocities.tractor_vx));
    assert(isfinite(velocities.tractor_vy));
    assert(isfinite(velocities.trailer_vx));
    assert(isfinite(velocities.trailer_vy));
    assert(fabsf(velocities.tractor_vy - rates.tractor_axle_vy) > 1e-3f);
}

static Drive drive_for_action_test(int dynamics_model) {
    Drive env = {0};
    env.entities = calloc(2, sizeof(Entity));
    assert(env.entities != NULL);
    env.articulated_ego_valid = 1;
    env.tractor_idx = 0;
    env.trailer_idx = 1;
    env.dt = 0.1f;
    env.action_type = 1;
    env.dynamics_model = dynamics_model;
    env.articulated_geometry = default_geometry();
    env.articulated_state =
        aligned_state(&env.articulated_geometry, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f);
    write_articulated_state_to_entities(&env);
    env.entities[env.tractor_idx].wheelbase = env.articulated_geometry.wheelbase;
    return env;
}

static void test_classic_and_jerk_action_semantics(void) {
    Drive classic = drive_for_action_test(CLASSIC);
    float classic_action[2] = {0.5f, 0.2f};
    classic.actions = classic_action;
    move_dynamics(&classic, 0, classic.tractor_idx);
    assert_close(classic.articulated_state.speed, 5.2f, TEST_EPSILON);
    assert_close(classic.articulated_state.steering_angle, 0.2f, TEST_EPSILON);
    assert(hitch_error(&classic.articulated_state,
                       &classic.articulated_geometry) <= TEST_EPSILON);
    free(classic.entities);

    Drive jerk = drive_for_action_test(JERK);
    float jerk_action[2] = {0.0f, 1.0f};
    jerk.actions = jerk_action;
    move_dynamics(&jerk, 0, jerk.tractor_idx);
    assert_close(jerk.articulated_state.speed, 5.0f, TEST_EPSILON);
    assert(jerk.articulated_state.steering_angle > 0.0f);
    assert(jerk.articulated_state.steering_angle <= 0.06001f);
    assert_close(jerk.entities[jerk.tractor_idx].a_lat, 0.4f, 2e-3f);
    assert(hitch_error(&jerk.articulated_state,
                       &jerk.articulated_geometry) <= TEST_EPSILON);
    free(jerk.entities);
}

int main(void) {
    test_substep_policy();
    test_analytic_straight_motion();
    test_analytic_tractor_turn_and_trailer_following();
    test_signed_reverse_motion();
    test_refinement_and_constraint_projection();
    test_articulation_limit();
    test_physically_consistent_entity_velocities();
    test_classic_and_jerk_action_semantics();
    puts("tractor-trailer dynamics tests passed");
    return 0;
}

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pufferlib/ocean/drive/drive.h"

#define TEST_EPSILON 1e-4f

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) <= TEST_EPSILON);
}

static ArticulatedGeometry default_geometry(void) {
    ArticulatedGeometry geometry = {0};
    assert(articulated_geometry_from_params(
               DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry) ==
           ARTICULATED_GEOMETRY_OK);
    return geometry;
}

static void test_invalid_state_is_zeroed(void) {
    Drive env = {0};
    int valid = 99;
    float state[ARTICULATED_EGO_STATE_SIZE];
    for (int i = 0; i < ARTICULATED_EGO_STATE_SIZE; i++) {
        state[i] = 123.0f;
    }

    c_get_articulated_ego_state(&env, &valid, state);

    assert(valid == 0);
    for (int i = 0; i < ARTICULATED_EGO_STATE_SIZE; i++) {
        assert(state[i] == 0.0f);
    }
}

static void test_complete_articulated_state(void) {
    Drive env = {0};
    env.articulated_ego_valid = 1;
    env.tractor_idx = 0;
    env.trailer_idx = 1;
    env.num_entities = 2;
    env.world_mean_x = 100.0f;
    env.world_mean_y = -50.0f;
    env.entities = calloc(2, sizeof(Entity));
    assert(env.entities != NULL);
    env.articulated_geometry = default_geometry();

    env.articulated_state.tractor_rear_axle = (ArticulatedPose){
        .x = 1.0f,
        .y = 2.0f,
        .z = 0.5f,
        .heading = 0.3f,
    };
    const ArticulatedPoint hitch = tractor_connection_point(
        env.articulated_state.tractor_rear_axle, &env.articulated_geometry);
    env.articulated_state.trailer_axle = trailer_axle_pose_from_connection(
        hitch, 0.1f, &env.articulated_geometry);
    env.articulated_state.speed = -2.0f;
    env.articulated_state.steering_angle = 0.25f;

    Entity *tractor = &env.entities[env.tractor_idx];
    Entity *trailer = &env.entities[env.trailer_idx];
    const ArticulatedPose tractor_center = axle_pose_to_entity_center(
        env.articulated_state.tractor_rear_axle,
        env.articulated_geometry.tractor_axle_to_center);
    const ArticulatedPose trailer_center = axle_pose_to_entity_center(
        env.articulated_state.trailer_axle,
        env.articulated_geometry.trailer_axle_to_center);

    tractor->x = tractor_center.x;
    tractor->y = tractor_center.y;
    tractor->z = tractor_center.z;
    tractor->heading = tractor_center.heading;
    tractor->vx = -1.5f;
    tractor->vy = -0.4f;
    tractor->vz = 0.1f;
    tractor->length = env.articulated_geometry.tractor_length;
    tractor->width = env.articulated_geometry.tractor_width;
    tractor->height = env.articulated_geometry.tractor_height;

    trailer->x = trailer_center.x;
    trailer->y = trailer_center.y;
    trailer->z = trailer_center.z;
    trailer->heading = trailer_center.heading;
    trailer->vx = -1.8f;
    trailer->vy = -0.2f;
    trailer->vz = -0.1f;
    trailer->length = env.articulated_geometry.trailer_length;
    trailer->width = env.articulated_geometry.trailer_width;
    trailer->height = env.articulated_geometry.trailer_height;

    int valid = 0;
    float state[ARTICULATED_EGO_STATE_SIZE] = {0};
    c_get_articulated_ego_state(&env, &valid, state);

    assert(valid == 1);
    assert_close(state[ART_STATE_TRACTOR_CENTER_X], tractor->x + 100.0f);
    assert_close(state[ART_STATE_TRACTOR_CENTER_Y], tractor->y - 50.0f);
    assert_close(state[ART_STATE_TRACTOR_CENTER_Z], tractor->z);
    assert_close(state[ART_STATE_TRACTOR_CENTER_HEADING], tractor->heading);
    assert_close(state[ART_STATE_TRAILER_CENTER_X], trailer->x + 100.0f);
    assert_close(state[ART_STATE_TRAILER_CENTER_Y], trailer->y - 50.0f);
    assert_close(state[ART_STATE_TRAILER_CENTER_Z], trailer->z);
    assert_close(state[ART_STATE_TRAILER_CENTER_HEADING], trailer->heading);

    assert_close(state[ART_STATE_TRACTOR_REAR_AXLE_X], 101.0f);
    assert_close(state[ART_STATE_TRACTOR_REAR_AXLE_Y], -48.0f);
    assert_close(state[ART_STATE_TRACTOR_REAR_AXLE_HEADING], 0.3f);
    assert_close(
        state[ART_STATE_TRAILER_AXLE_X],
        env.articulated_state.trailer_axle.x + 100.0f);
    assert_close(
        state[ART_STATE_TRAILER_AXLE_Y],
        env.articulated_state.trailer_axle.y - 50.0f);
    assert_close(state[ART_STATE_TRAILER_AXLE_HEADING], 0.1f);

    assert_close(state[ART_STATE_HITCH_X], hitch.x + 100.0f);
    assert_close(state[ART_STATE_HITCH_Y], hitch.y - 50.0f);
    assert_close(state[ART_STATE_HITCH_Z], hitch.z);
    const ArticulatedPoint trailer_hitch = trailer_connection_point(
        env.articulated_state.trailer_axle, &env.articulated_geometry);
    assert_close(trailer_hitch.x, hitch.x);
    assert_close(trailer_hitch.y, hitch.y);
    assert_close(trailer_hitch.z, hitch.z);

    assert_close(state[ART_STATE_TRACTOR_VX], -1.5f);
    assert_close(state[ART_STATE_TRACTOR_VY], -0.4f);
    assert_close(state[ART_STATE_TRACTOR_VZ], 0.1f);
    assert_close(state[ART_STATE_TRAILER_VX], -1.8f);
    assert_close(state[ART_STATE_TRAILER_VY], -0.2f);
    assert_close(state[ART_STATE_TRAILER_VZ], -0.1f);
    assert_close(state[ART_STATE_SPEED], -2.0f);
    assert_close(state[ART_STATE_STEERING_ANGLE], 0.25f);
    assert_close(state[ART_STATE_ARTICULATION_ANGLE], -0.2f);
    assert_close(
        state[ART_STATE_TRACTOR_LENGTH],
        env.articulated_geometry.tractor_length);
    assert_close(
        state[ART_STATE_TRACTOR_WIDTH],
        env.articulated_geometry.tractor_width);
    assert_close(
        state[ART_STATE_TRACTOR_HEIGHT],
        env.articulated_geometry.tractor_height);
    assert_close(
        state[ART_STATE_TRAILER_LENGTH],
        env.articulated_geometry.trailer_length);
    assert_close(
        state[ART_STATE_TRAILER_WIDTH],
        env.articulated_geometry.trailer_width);
    assert_close(
        state[ART_STATE_TRAILER_HEIGHT],
        env.articulated_geometry.trailer_height);

    free(env.entities);
}

int main(void) {
    test_invalid_state_is_zeroed();
    test_complete_articulated_state();
    puts("tractor-trailer state export tests passed");
    return 0;
}

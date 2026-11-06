#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "pufferlib/ocean/drive/tractor_trailer.h"

#define TEST_EPSILON 1e-5f

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) <= TEST_EPSILON);
}

static void test_default_geometry(void) {
    ArticulatedGeometry geometry = {0};
    ArticulatedGeometryStatus status =
        articulated_geometry_from_params(DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry);

    assert(status == ARTICULATED_GEOMETRY_OK);
    assert_close(geometry.wheelbase, 3.6f);
    assert_close(geometry.tractor_rear_axle_to_hitch, 0.6f);
    assert_close(geometry.trailer_front_bumper_to_hitch, 2.2f);
    assert_close(geometry.trailer_axle_to_hitch, 8.3f);
    assert_close(geometry.tractor_nominal_length, 6.3f);
    assert_close(geometry.trailer_nominal_length, 12.5f);
    assert_close(geometry.tractor_axle_to_center, 1.65f);
    assert_close(geometry.trailer_axle_to_center, 4.25f);
    assert_close(geometry.tractor_length, 6.3f);
    assert_close(geometry.trailer_length, 12.5f);
    assert_close(geometry.tractor_width, 2.55f);
    assert_close(geometry.trailer_width, 2.55f);
    assert_close(geometry.tractor_height, 3.5f);
    assert_close(geometry.trailer_height, 3.5f);
    assert(geometry.tractor_length_mismatch == 0);
    assert(geometry.trailer_length_mismatch == 0);
}

static void test_pose_round_trip(float heading, float offset) {
    ArticulatedPose axle = {.x = 11.25f, .y = -7.5f, .z = 2.0f, .heading = heading};
    ArticulatedPose center = axle_pose_to_entity_center(axle, offset);
    ArticulatedPose recovered = entity_center_pose_to_axle(center, offset);

    assert_close(recovered.x, axle.x);
    assert_close(recovered.y, axle.y);
    assert_close(recovered.z, axle.z);
    assert_close(recovered.heading, axle.heading);
}

static void test_pose_conversions(void) {
    test_pose_round_trip(0.0f, 1.65f);
    test_pose_round_trip(0.75f, 4.25f);
    test_pose_round_trip(-2.4f, 4.25f);
}

static void test_connection_point_coincidence(float tractor_heading, float trailer_heading) {
    ArticulatedGeometry geometry = {0};
    assert(articulated_geometry_from_params(DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry) ==
           ARTICULATED_GEOMETRY_OK);

    ArticulatedPose tractor_axle = {.x = 3.25f, .y = -4.5f, .z = 1.2f, .heading = tractor_heading};
    ArticulatedPoint tractor_connection = tractor_connection_point(tractor_axle, &geometry);
    ArticulatedPose trailer_axle =
        trailer_axle_pose_from_connection(tractor_connection, trailer_heading, &geometry);
    ArticulatedPoint trailer_connection = trailer_connection_point(trailer_axle, &geometry);

    assert_close(trailer_connection.x, tractor_connection.x);
    assert_close(trailer_connection.y, tractor_connection.y);
    assert_close(trailer_connection.z, tractor_connection.z);
}

static void test_connection_points(void) {
    test_connection_point_coincidence(0.0f, 0.0f);
    test_connection_point_coincidence(0.7f, -0.2f);
    test_connection_point_coincidence(-2.0f, 2.4f);
}

static void test_invalid_geometry(void) {
    ArticulatedGeometry geometry = {0};
    float params[NON_KINEMATIC_VEHICLE_PARAM_COUNT];

    for (size_t i = 0; i < NON_KINEMATIC_VEHICLE_PARAM_COUNT; i++) {
        params[i] = DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS[i];
    }
    params[NKVP_TRACTOR_WIDTH] = 0.0f;
    assert(articulated_geometry_from_params(params, &geometry) ==
           ARTICULATED_GEOMETRY_NON_POSITIVE_DIMENSION);

    params[NKVP_TRACTOR_WIDTH] = DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS[NKVP_TRACTOR_WIDTH];
    params[NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE] = 0.0f;
    assert(articulated_geometry_from_params(params, &geometry) ==
           ARTICULATED_GEOMETRY_INVALID_TRACTOR_LONGITUDINAL_GEOMETRY);

    params[NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE] =
        DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS[NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE];
    params[NKVP_TRAILER_FRONT_BUMPER_TO_HITCH] = params[NKVP_TRAILER_AXLE_TO_FRONT_BUMPER];
    assert(articulated_geometry_from_params(params, &geometry) ==
           ARTICULATED_GEOMETRY_INVALID_TRAILER_LONGITUDINAL_GEOMETRY);

    assert(articulated_geometry_from_params(NULL, &geometry) == ARTICULATED_GEOMETRY_NULL_ARGUMENT);
    assert(articulated_geometry_from_params(params, NULL) == ARTICULATED_GEOMETRY_NULL_ARGUMENT);
}

int main(void) {
    test_default_geometry();
    test_pose_conversions();
    test_connection_points();
    test_invalid_geometry();
    puts("tractor-trailer geometry tests passed");
    return 0;
}

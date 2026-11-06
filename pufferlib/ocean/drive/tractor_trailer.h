#ifndef PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_H
#define PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_H

#include <math.h>
#include <stddef.h>

#define NON_KINEMATIC_VEHICLE_PARAM_COUNT 13
#define TRACTOR_TRAILER_GEOMETRY_EPSILON 1e-5f

typedef enum NonKinematicVehicleParam {
    NKVP_TRACTOR_NOMINAL_LENGTH = 0,
    NKVP_TRAILER_NOMINAL_LENGTH = 1,
    NKVP_TRACTOR_WIDTH = 2,
    NKVP_TRAILER_WIDTH = 3,
    NKVP_TRACTOR_HEIGHT = 4,
    NKVP_TRAILER_HEIGHT = 5,
    NKVP_TRACTOR_REAR_AXLE_TO_HITCH = 6,
    NKVP_TRAILER_FRONT_BUMPER_TO_HITCH = 7,
    NKVP_TRACTOR_REAR_AXLE_TO_REAR_BUMPER = 8,
    NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE = 9,
    NKVP_TRACTOR_FRONT_AXLE_TO_FRONT_BUMPER = 10,
    NKVP_TRAILER_AXLE_TO_REAR_BUMPER = 11,
    NKVP_TRAILER_AXLE_TO_FRONT_BUMPER = 12,
} NonKinematicVehicleParam;

static const float DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS[NON_KINEMATIC_VEHICLE_PARAM_COUNT] = {
    [NKVP_TRACTOR_NOMINAL_LENGTH] = 6.3f,
    [NKVP_TRAILER_NOMINAL_LENGTH] = 12.5f,
    [NKVP_TRACTOR_WIDTH] = 2.55f,
    [NKVP_TRAILER_WIDTH] = 2.55f,
    [NKVP_TRACTOR_HEIGHT] = 3.5f,
    [NKVP_TRAILER_HEIGHT] = 3.5f,
    [NKVP_TRACTOR_REAR_AXLE_TO_HITCH] = 0.6f,
    [NKVP_TRAILER_FRONT_BUMPER_TO_HITCH] = 2.2f,
    [NKVP_TRACTOR_REAR_AXLE_TO_REAR_BUMPER] = 1.5f,
    [NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE] = 3.6f,
    [NKVP_TRACTOR_FRONT_AXLE_TO_FRONT_BUMPER] = 1.2f,
    [NKVP_TRAILER_AXLE_TO_REAR_BUMPER] = 2.0f,
    [NKVP_TRAILER_AXLE_TO_FRONT_BUMPER] = 10.5f,
};

typedef struct ArticulatedPose {
    float x;
    float y;
    float z;
    float heading;
} ArticulatedPose;

typedef struct ArticulatedPoint {
    float x;
    float y;
    float z;
} ArticulatedPoint;

typedef struct ArticulatedGeometry {
    float tractor_nominal_length;
    float trailer_nominal_length;
    float tractor_length;
    float trailer_length;
    float tractor_width;
    float trailer_width;
    float tractor_height;
    float trailer_height;
    float wheelbase;
    float tractor_rear_axle_to_hitch;
    float trailer_front_bumper_to_hitch;
    float trailer_axle_to_hitch;
    float tractor_axle_to_center;
    float trailer_axle_to_center;
    int tractor_length_mismatch;
    int trailer_length_mismatch;
} ArticulatedGeometry;

typedef struct ArticulatedVehicleState {
    ArticulatedPose tractor_rear_axle;
    ArticulatedPose trailer_axle;
    float speed;
    float steering_angle;
} ArticulatedVehicleState;

typedef enum ArticulatedGeometryStatus {
    ARTICULATED_GEOMETRY_OK = 0,
    ARTICULATED_GEOMETRY_NULL_ARGUMENT = 1,
    ARTICULATED_GEOMETRY_NON_POSITIVE_DIMENSION = 2,
    ARTICULATED_GEOMETRY_INVALID_TRACTOR_LONGITUDINAL_GEOMETRY = 3,
    ARTICULATED_GEOMETRY_INVALID_TRAILER_LONGITUDINAL_GEOMETRY = 4,
} ArticulatedGeometryStatus;

static inline ArticulatedGeometryStatus articulated_geometry_from_params(
    const float params[NON_KINEMATIC_VEHICLE_PARAM_COUNT], ArticulatedGeometry *geometry) {
    if (params == NULL || geometry == NULL) {
        return ARTICULATED_GEOMETRY_NULL_ARGUMENT;
    }

    const float tractor_nominal_length = params[NKVP_TRACTOR_NOMINAL_LENGTH];
    const float trailer_nominal_length = params[NKVP_TRAILER_NOMINAL_LENGTH];
    const float tractor_width = params[NKVP_TRACTOR_WIDTH];
    const float trailer_width = params[NKVP_TRAILER_WIDTH];
    const float tractor_height = params[NKVP_TRACTOR_HEIGHT];
    const float trailer_height = params[NKVP_TRAILER_HEIGHT];
    if (tractor_nominal_length <= 0.0f || trailer_nominal_length <= 0.0f || tractor_width <= 0.0f ||
        trailer_width <= 0.0f || tractor_height <= 0.0f || trailer_height <= 0.0f) {
        return ARTICULATED_GEOMETRY_NON_POSITIVE_DIMENSION;
    }

    const float tractor_rear_extent = params[NKVP_TRACTOR_REAR_AXLE_TO_REAR_BUMPER];
    const float wheelbase = params[NKVP_TRACTOR_REAR_AXLE_TO_FRONT_AXLE];
    const float tractor_front_overhang = params[NKVP_TRACTOR_FRONT_AXLE_TO_FRONT_BUMPER];
    const float tractor_rear_axle_to_hitch = params[NKVP_TRACTOR_REAR_AXLE_TO_HITCH];
    if (tractor_rear_extent < 0.0f || wheelbase <= 0.0f || tractor_front_overhang < 0.0f ||
        tractor_rear_axle_to_hitch < 0.0f) {
        return ARTICULATED_GEOMETRY_INVALID_TRACTOR_LONGITUDINAL_GEOMETRY;
    }

    const float trailer_rear_extent = params[NKVP_TRAILER_AXLE_TO_REAR_BUMPER];
    const float trailer_front_extent = params[NKVP_TRAILER_AXLE_TO_FRONT_BUMPER];
    const float trailer_front_bumper_to_hitch = params[NKVP_TRAILER_FRONT_BUMPER_TO_HITCH];
    const float trailer_axle_to_hitch = trailer_front_extent - trailer_front_bumper_to_hitch;
    if (trailer_rear_extent < 0.0f || trailer_front_extent <= 0.0f || trailer_front_bumper_to_hitch < 0.0f ||
        trailer_axle_to_hitch <= 0.0f) {
        return ARTICULATED_GEOMETRY_INVALID_TRAILER_LONGITUDINAL_GEOMETRY;
    }

    const float tractor_front_extent = wheelbase + tractor_front_overhang;
    const float tractor_length = tractor_rear_extent + tractor_front_extent;
    const float trailer_length = trailer_rear_extent + trailer_front_extent;
    *geometry = (ArticulatedGeometry){
        .tractor_nominal_length = tractor_nominal_length,
        .trailer_nominal_length = trailer_nominal_length,
        .tractor_length = tractor_length,
        .trailer_length = trailer_length,
        .tractor_width = tractor_width,
        .trailer_width = trailer_width,
        .tractor_height = tractor_height,
        .trailer_height = trailer_height,
        .wheelbase = wheelbase,
        .tractor_rear_axle_to_hitch = tractor_rear_axle_to_hitch,
        .trailer_front_bumper_to_hitch = trailer_front_bumper_to_hitch,
        .trailer_axle_to_hitch = trailer_axle_to_hitch,
        .tractor_axle_to_center = 0.5f * (tractor_front_extent - tractor_rear_extent),
        .trailer_axle_to_center = 0.5f * (trailer_front_extent - trailer_rear_extent),
        .tractor_length_mismatch = fabsf(tractor_nominal_length - tractor_length) > TRACTOR_TRAILER_GEOMETRY_EPSILON,
        .trailer_length_mismatch = fabsf(trailer_nominal_length - trailer_length) > TRACTOR_TRAILER_GEOMETRY_EPSILON,
    };
    return ARTICULATED_GEOMETRY_OK;
}

static inline ArticulatedPose axle_pose_to_entity_center(ArticulatedPose axle_pose, float axle_to_center) {
    axle_pose.x += axle_to_center * cosf(axle_pose.heading);
    axle_pose.y += axle_to_center * sinf(axle_pose.heading);
    return axle_pose;
}

static inline ArticulatedPose entity_center_pose_to_axle(ArticulatedPose center_pose, float axle_to_center) {
    center_pose.x -= axle_to_center * cosf(center_pose.heading);
    center_pose.y -= axle_to_center * sinf(center_pose.heading);
    return center_pose;
}

static inline ArticulatedPoint tractor_connection_point(ArticulatedPose tractor_rear_axle,
                                                         const ArticulatedGeometry *geometry) {
    return (ArticulatedPoint){
        .x = tractor_rear_axle.x - geometry->tractor_rear_axle_to_hitch * cosf(tractor_rear_axle.heading),
        .y = tractor_rear_axle.y - geometry->tractor_rear_axle_to_hitch * sinf(tractor_rear_axle.heading),
        .z = tractor_rear_axle.z,
    };
}

static inline ArticulatedPoint trailer_connection_point(ArticulatedPose trailer_axle,
                                                         const ArticulatedGeometry *geometry) {
    return (ArticulatedPoint){
        .x = trailer_axle.x + geometry->trailer_axle_to_hitch * cosf(trailer_axle.heading),
        .y = trailer_axle.y + geometry->trailer_axle_to_hitch * sinf(trailer_axle.heading),
        .z = trailer_axle.z,
    };
}

static inline ArticulatedPose trailer_axle_pose_from_connection(ArticulatedPoint connection, float trailer_heading,
                                                                const ArticulatedGeometry *geometry) {
    return (ArticulatedPose){
        .x = connection.x - geometry->trailer_axle_to_hitch * cosf(trailer_heading),
        .y = connection.y - geometry->trailer_axle_to_hitch * sinf(trailer_heading),
        .z = connection.z,
        .heading = trailer_heading,
    };
}

#endif

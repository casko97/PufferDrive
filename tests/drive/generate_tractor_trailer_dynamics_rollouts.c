#include <math.h>
#include <stdio.h>

#include "pufferlib/ocean/drive/tractor_trailer_dynamics.h"

#define ROLLOUT_FRAMES 41
#define ROLLOUT_DT 0.1f

static ArticulatedVehicleState aligned_state(const ArticulatedGeometry *geometry,
                                             float speed, float steering) {
    ArticulatedVehicleState state = {
        .tractor_rear_axle = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .heading = 0.0f},
        .speed = speed,
        .steering_angle = steering,
    };
    ArticulatedPoint hitch =
        tractor_connection_point(state.tractor_rear_axle, geometry);
    state.trailer_axle =
        trailer_axle_pose_from_connection(hitch, 0.0f, geometry);
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

static void emit_rollout(const char *name, float speed, float steering,
                         const ArticulatedGeometry *geometry) {
    ArticulatedVehicleState state =
        aligned_state(geometry, speed, steering);
    for (int frame = 0; frame < ROLLOUT_FRAMES; frame++) {
        printf("%s,%d,%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9g\n",
               name, frame, frame * ROLLOUT_DT,
               state.tractor_rear_axle.x, state.tractor_rear_axle.y,
               state.tractor_rear_axle.heading, state.trailer_axle.x,
               state.trailer_axle.y, state.trailer_axle.heading,
               state.speed, state.steering_angle,
               hitch_error(&state, geometry));
        ArticulatedStepInput input = {
            .speed_start = speed,
            .speed_end = speed,
            .steering_start = steering,
            .steering_end = steering,
            .dt = ROLLOUT_DT,
        };
        articulated_integrate_rk2(&state, geometry, &input);
    }
}

int main(void) {
    ArticulatedGeometry geometry = {0};
    if (articulated_geometry_from_params(
            DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry) !=
        ARTICULATED_GEOMETRY_OK)
        return 1;

    puts("scenario,frame,time,xr,yr,psi,xt,yt,psi_t,speed,steering,hitch_error");
    emit_rollout("straight_forward", 5.0f, 0.0f, &geometry);
    emit_rollout("constant_steering", 6.0f, 0.25f, &geometry);
    emit_rollout("reverse_steering", -3.0f, 0.3f, &geometry);
    return 0;
}

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pufferlib/ocean/drive/drive.h"

#define TEST_EPSILON 1e-4f
#define TEST_TRAJECTORY_SIZE 3

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) <= TEST_EPSILON);
}

static void allocate_trajectory(Entity *entity) {
    entity->array_size = TEST_TRAJECTORY_SIZE;
    entity->traj_x = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_y = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_z = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_vx = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_vy = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_vz = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_heading = calloc(TEST_TRAJECTORY_SIZE, sizeof(float));
    entity->traj_valid = calloc(TEST_TRAJECTORY_SIZE, sizeof(int));
    assert(entity->traj_x != NULL);
    assert(entity->traj_y != NULL);
    assert(entity->traj_z != NULL);
    assert(entity->traj_vx != NULL);
    assert(entity->traj_vy != NULL);
    assert(entity->traj_vz != NULL);
    assert(entity->traj_heading != NULL);
    assert(entity->traj_valid != NULL);
}

static Drive make_test_env(int ego_vehicle_type, int sdc_track_index) {
    Drive env = {0};
    env.ego_vehicle_type = ego_vehicle_type;
    env.sdc_track_index = sdc_track_index;
    env.init_steps = 1;
    env.control_mode = CONTROL_SDC_ONLY;
    env.init_mode = INIT_ALL_VALID;
    env.max_controlled_agents = MAX_AGENTS;
    env.num_objects = 1;
    env.num_roads = 1;
    env.num_entities = 2;
    env.entities = calloc((size_t)env.num_entities, sizeof(Entity));
    assert(env.entities != NULL);

    Entity *tractor = &env.entities[0];
    allocate_trajectory(tractor);
    tractor->type = VEHICLE;
    tractor->id = 17;
    tractor->scenario_id = 23;
    tractor->articulated_parent_idx = -1;
    tractor->goal_position_x = 100.0f;
    tractor->goal_position_y = 20.0f;
    for (int t = 0; t < TEST_TRAJECTORY_SIZE; t++) {
        tractor->traj_x[t] = 10.0f + 2.0f * (float)t;
        tractor->traj_y[t] = -4.0f + 0.5f * (float)t;
        tractor->traj_z[t] = 0.25f;
        tractor->traj_vx[t] = 6.0f + (float)t;
        tractor->traj_vy[t] = 0.25f * (float)t;
        tractor->traj_heading[t] = 0.15f + 0.1f * (float)t;
        tractor->traj_valid[t] = 1;
    }

    Entity *road = &env.entities[1];
    allocate_trajectory(road);
    road->type = ROAD_LANE;
    road->id = 99;
    for (int t = 0; t < TEST_TRAJECTORY_SIZE; t++) {
        road->traj_valid[t] = 1;
    }
    return env;
}

static void destroy_test_env(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
}

static void assert_trajectory_hitches_coincide(const Drive *env) {
    const Entity *tractor = &env->entities[env->tractor_idx];
    const Entity *trailer = &env->entities[env->trailer_idx];
    for (int t = 0; t < tractor->array_size; t++) {
        ArticulatedPose tractor_center = {
            .x = tractor->traj_x[t],
            .y = tractor->traj_y[t],
            .z = tractor->traj_z[t],
            .heading = tractor->traj_heading[t],
        };
        ArticulatedPose trailer_center = {
            .x = trailer->traj_x[t],
            .y = trailer->traj_y[t],
            .z = trailer->traj_z[t],
            .heading = trailer->traj_heading[t],
        };
        ArticulatedPose tractor_axle =
            entity_center_pose_to_axle(tractor_center, env->articulated_geometry.tractor_axle_to_center);
        ArticulatedPose trailer_axle =
            entity_center_pose_to_axle(trailer_center, env->articulated_geometry.trailer_axle_to_center);
        ArticulatedPoint tractor_hitch = tractor_connection_point(tractor_axle, &env->articulated_geometry);
        ArticulatedPoint trailer_hitch = trailer_connection_point(trailer_axle, &env->articulated_geometry);
        assert_close(tractor_hitch.x, trailer_hitch.x);
        assert_close(tractor_hitch.y, trailer_hitch.y);
        assert_close(tractor_hitch.z, trailer_hitch.z);
    }
}

static void test_disabled_and_missing_sdc_fallback(void) {
    Drive car_env = make_test_env(EGO_VEHICLE_CAR, 0);
    assert(initialize_articulated_ego(&car_env) == 0);
    assert(car_env.articulated_ego_valid == 0);
    assert(car_env.num_objects == 1);
    assert(car_env.num_entities == 2);
    destroy_test_env(&car_env);

    Drive missing_sdc_env = make_test_env(EGO_VEHICLE_TRACTOR_TRAILER, -1);
    assert(initialize_articulated_ego(&missing_sdc_env) == 0);
    assert(missing_sdc_env.articulated_ego_valid == 0);
    assert(missing_sdc_env.num_objects == 1);
    assert(missing_sdc_env.num_entities == 2);
    destroy_test_env(&missing_sdc_env);
}

static void test_entity_lifecycle(void) {
    Drive env = make_test_env(EGO_VEHICLE_TRACTOR_TRAILER, 0);
    assert(initialize_articulated_ego(&env) == 1);
    assert(env.articulated_ego_valid == 1);
    assert(env.tractor_idx == 0);
    assert(env.trailer_idx == 1);
    assert(env.num_objects == 2);
    assert(env.num_entities == 3);
    assert(env.entities[2].type == ROAD_LANE);
    assert(env.entities[2].id == 99);

    Entity *tractor = &env.entities[env.tractor_idx];
    Entity *trailer = &env.entities[env.trailer_idx];
    assert(tractor->articulated_role == ARTICULATED_ROLE_TRACTOR);
    assert(trailer->articulated_role == ARTICULATED_ROLE_TRAILER);
    assert(trailer->articulated_parent_idx == env.tractor_idx);
    assert_close(tractor->length, 6.3f);
    assert_close(trailer->length, 12.5f);
    assert_close(tractor->wheelbase, 3.6f);
    assert_trajectory_hitches_coincide(&env);

    set_active_agents(&env);
    assert(env.active_agent_count == 1);
    assert(env.active_agent_indices[0] == env.tractor_idx);
    assert(env.static_agent_count == 1);
    assert(env.static_agent_indices[0] == env.trailer_idx);
    assert(env.num_actors == 2);

    trailer->collided_before_goal = 1;
    set_start_position(&env);
    tractor = &env.entities[env.tractor_idx];
    trailer = &env.entities[env.trailer_idx];
    assert_close(tractor->wheelbase, 3.6f);
    assert_close(trailer->x, trailer->traj_x[env.init_steps]);
    assert_close(trailer->y, trailer->traj_y[env.init_steps]);
    assert_close(trailer->vx, trailer->traj_vx[env.init_steps]);
    assert_close(trailer->vy, trailer->traj_vy[env.init_steps]);
    assert(trailer->collided_before_goal == 0);
    ArticulatedPose expected_trailer_axle = entity_center_pose_to_axle(
        (ArticulatedPose){
            .x = trailer->x,
            .y = trailer->y,
            .z = trailer->z,
            .heading = trailer->heading,
        },
        env.articulated_geometry.trailer_axle_to_center);
    assert_close(env.articulated_state.trailer_axle.x, expected_trailer_axle.x);
    assert_close(env.articulated_state.trailer_axle.y, expected_trailer_axle.y);

    trailer->x = 1234.0f;
    trailer->y = -4321.0f;
    trailer->collision_state = VEHICLE_COLLISION;
    trailer->removed = 1;
    env.timestep = 44;
    respawn_agent(&env, env.tractor_idx);
    assert_close(trailer->x, trailer->traj_x[0]);
    assert_close(trailer->y, trailer->traj_y[0]);
    assert_close(trailer->vx, trailer->traj_vx[0]);
    assert_close(trailer->vy, trailer->traj_vy[0]);
    assert(trailer->collision_state == NO_COLLISION);
    assert(trailer->removed == 0);
    assert(trailer->respawn_timestep == env.timestep);
    assert_close(env.articulated_state.trailer_axle.x,
                 trailer->traj_x[0] - env.articulated_geometry.trailer_axle_to_center *
                                          cosf(trailer->traj_heading[0]));

    destroy_test_env(&env);
}

int main(void) {
    test_disabled_and_missing_sdc_fallback();
    test_entity_lifecycle();
    puts("tractor-trailer lifecycle tests passed");
    return 0;
}

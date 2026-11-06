#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pufferlib/ocean/drive/drive.h"

#define TEST_ENTITY_CAPACITY 40

static void set_box(Entity *entity, float x, float y, float length, float width) {
    entity->type = VEHICLE;
    entity->x = x;
    entity->y = y;
    entity->length = length;
    entity->width = width;
    entity->heading = 0.0f;
    entity->heading_x = 1.0f;
    entity->heading_y = 0.0f;
    entity->valid = 1;
    entity->respawn_timestep = -1;
}

static void initialize_empty_grid(Drive *env) {
    env->grid_map = calloc(1, sizeof(GridMap));
    assert(env->grid_map != NULL);
    GridMap *grid = env->grid_map;
    grid->top_left_x = -50.0f;
    grid->top_left_y = 50.0f;
    grid->bottom_right_x = 50.0f;
    grid->bottom_right_y = -50.0f;
    grid->grid_cols = 20;
    grid->grid_rows = 20;
    int cell_count = grid->grid_cols * grid->grid_rows;
    grid->cells = calloc((size_t)cell_count, sizeof(GridMapEntity *));
    grid->cell_entities_count = calloc((size_t)cell_count, sizeof(int));
    grid->neighbor_cache_entities =
        calloc((size_t)cell_count, sizeof(GridMapEntity *));
    grid->neighbor_cache_count = calloc((size_t)cell_count, sizeof(int));
    assert(grid->cells != NULL);
    assert(grid->cell_entities_count != NULL);
    assert(grid->neighbor_cache_entities != NULL);
    assert(grid->neighbor_cache_count != NULL);
}

static void destroy_grid(GridMap *grid) {
    if (grid == NULL)
        return;
    int cell_count = grid->grid_cols * grid->grid_rows;
    for (int index = 0; index < cell_count; index++) {
        free(grid->cells[index]);
        free(grid->neighbor_cache_entities[index]);
    }
    free(grid->cells);
    free(grid->cell_entities_count);
    free(grid->neighbor_cache_entities);
    free(grid->neighbor_cache_count);
    free(grid);
}

static Drive make_collision_fixture(void) {
    Drive env = {0};
    env.entities = calloc(TEST_ENTITY_CAPACITY, sizeof(Entity));
    assert(env.entities != NULL);
    env.num_entities = 3;
    env.num_objects = 3;
    env.articulated_ego_valid = 1;
    env.tractor_idx = 0;
    env.trailer_idx = 1;
    assert(articulated_geometry_from_params(
               DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS,
               &env.articulated_geometry) == ARTICULATED_GEOMETRY_OK);

    set_box(&env.entities[0], 0.0f, 0.0f, 6.3f, 2.55f);
    set_box(&env.entities[1], -6.3f, 0.0f, 12.5f, 2.55f);
    set_box(&env.entities[2], -10.0f, 0.0f, 2.0f, 2.0f);
    env.entities[0].articulated_role = ARTICULATED_ROLE_TRACTOR;
    env.entities[1].articulated_role = ARTICULATED_ROLE_TRAILER;
    env.entities[1].articulated_parent_idx = 0;
    env.articulated_state.speed = 5.0f;

    env.active_agent_count = 1;
    env.static_agent_count = 2;
    env.num_actors = 3;
    env.active_agent_indices = malloc(sizeof(int));
    env.static_agent_indices = malloc(2 * sizeof(int));
    assert(env.active_agent_indices != NULL);
    assert(env.static_agent_indices != NULL);
    env.active_agent_indices[0] = 0;
    env.static_agent_indices[0] = 1;
    env.static_agent_indices[1] = 2;
    initialize_empty_grid(&env);
    return env;
}

static void destroy_fixture(Drive *env) {
    for (int index = 0; index < env->num_entities; index++)
        free_entity(&env->entities[index]);
    free(env->entities);
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->observations);
    destroy_grid(env->grid_map);
}

static void test_mutual_collision_ignored_and_trailer_collision_aggregated(void) {
    Drive env = make_collision_fixture();
    assert(check_aabb_collision(&env.entities[0], &env.entities[1]) == 1);
    assert(collision_check(&env, env.tractor_idx) == -1);
    assert(collision_check(&env, env.trailer_idx) == 2);

    env.collision_behavior = STOP_AGENT;
    compute_agent_metrics(&env, env.tractor_idx);
    assert(env.entities[0].collision_state == VEHICLE_COLLISION);
    assert(env.entities[1].collision_state == VEHICLE_COLLISION);
    assert(env.entities[0].stopped == 1);
    assert(env.entities[1].stopped == 1);
    assert(env.articulated_state.speed == 0.0f);
    destroy_fixture(&env);

    env = make_collision_fixture();
    env.collision_behavior = REMOVE_AGENT;
    compute_agent_metrics(&env, env.tractor_idx);
    assert(env.entities[0].removed == 1);
    assert(env.entities[1].removed == 1);
    assert(env.entities[2].removed == 1);
    assert(env.entities[0].x == INVALID_POSITION);
    assert(env.entities[1].x == INVALID_POSITION);
    destroy_fixture(&env);
}

static void add_trailer_only_road_edge(Drive *env) {
    Entity *edge = &env->entities[2];
    edge->type = ROAD_EDGE;
    edge->array_size = 2;
    edge->traj_x = malloc(2 * sizeof(float));
    edge->traj_y = malloc(2 * sizeof(float));
    assert(edge->traj_x != NULL);
    assert(edge->traj_y != NULL);
    edge->traj_x[0] = edge->traj_x[1] = -10.0f;
    edge->traj_y[0] = -5.0f;
    edge->traj_y[1] = 5.0f;

    int trailer_cell = getGridIndex(env, env->entities[1].x, env->entities[1].y);
    env->grid_map->cell_entities_count[trailer_cell] = 1;
    env->grid_map->cells[trailer_cell] = malloc(sizeof(GridMapEntity));
    assert(env->grid_map->cells[trailer_cell] != NULL);
    env->grid_map->cells[trailer_cell][0] =
        (GridMapEntity){.entity_idx = 2, .geometry_idx = 0};
}

static void test_trailer_only_offroad_aggregated(void) {
    Drive env = make_collision_fixture();
    env.num_objects = 2;
    env.static_agent_count = 1;
    env.num_actors = 2;
    add_trailer_only_road_edge(&env);
    env.offroad_behavior = STOP_AGENT;

    compute_agent_metrics(&env, env.tractor_idx);
    assert(env.entities[0].collision_state == OFFROAD);
    assert(env.entities[1].collision_state == OFFROAD);
    assert(env.entities[0].metrics_array[OFFROAD_IDX] == 1.0f);
    assert(env.entities[1].metrics_array[OFFROAD_IDX] == 1.0f);
    assert(env.entities[0].stopped == 1);
    assert(env.entities[1].stopped == 1);
    destroy_fixture(&env);
}

static void prepare_observation_grid(Drive *env, int entity_idx) {
    int grid_index =
        getGridIndex(env, env->entities[entity_idx].x, env->entities[entity_idx].y);
    if (env->grid_map->neighbor_cache_entities[grid_index] == NULL) {
        env->grid_map->neighbor_cache_entities[grid_index] =
            calloc(1, sizeof(GridMapEntity));
        assert(env->grid_map->neighbor_cache_entities[grid_index] != NULL);
    }
}

static void test_owner_observation_filter_and_other_agent_visibility(void) {
    Drive env = make_collision_fixture();
    env.entities[2].x = 20.0f;
    env.active_agent_count = 2;
    env.static_agent_count = 1;
    env.num_actors = 3;
    env.active_agent_indices = realloc(env.active_agent_indices, 2 * sizeof(int));
    assert(env.active_agent_indices != NULL);
    env.active_agent_indices[0] = 0;
    env.active_agent_indices[1] = 2;
    env.static_agent_indices[0] = 1;
    env.dynamics_model = CLASSIC;
    prepare_observation_grid(&env, 0);
    prepare_observation_grid(&env, 2);

    int observation_size =
        EGO_FEATURES_CLASSIC + PARTNER_FEATURES * (MAX_AGENTS - 1) +
        ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    env.observations =
        calloc((size_t)env.active_agent_count * observation_size, sizeof(float));
    assert(env.observations != NULL);
    compute_observations(&env);

    float *tractor_observation = env.observations;
    assert(fabsf(tractor_observation[EGO_FEATURES_CLASSIC] - 0.4f) < 1e-6f);
    assert(tractor_observation[EGO_FEATURES_CLASSIC + PARTNER_FEATURES] == 0.0f);

    float *other_observation = env.observations + observation_size;
    int first_partner = EGO_FEATURES_CLASSIC;
    int second_partner = first_partner + PARTNER_FEATURES;
    assert(fabsf(other_observation[first_partner] + 0.4f) < 1e-6f);
    assert(fabsf(other_observation[second_partner] + 0.526f) < 1e-5f);
    assert(fabsf(other_observation[second_partner + 3] -
                 12.5f / MAX_VEH_LEN) < 1e-6f);
    destroy_fixture(&env);
}

static void allocate_single_sample_trajectory(Entity *entity) {
    entity->array_size = 1;
    entity->traj_x = calloc(1, sizeof(float));
    entity->traj_y = calloc(1, sizeof(float));
    entity->traj_z = calloc(1, sizeof(float));
    entity->traj_vx = calloc(1, sizeof(float));
    entity->traj_vy = calloc(1, sizeof(float));
    entity->traj_vz = calloc(1, sizeof(float));
    entity->traj_heading = calloc(1, sizeof(float));
    entity->traj_valid = calloc(1, sizeof(int));
    assert(entity->traj_valid != NULL);
    entity->traj_valid[0] = 1;
}

static void test_actor_capacity_reserves_trailer_slot(void) {
    Drive env = {0};
    env.num_entities = MAX_AGENTS + 2;
    env.num_objects = env.num_entities;
    env.num_agents = MAX_AGENTS;
    env.init_mode = INIT_ALL_VALID;
    env.control_mode = CONTROL_SDC_ONLY;
    env.sdc_track_index = 0;
    env.articulated_ego_valid = 1;
    env.tractor_idx = 0;
    env.trailer_idx = 1;
    env.entities = calloc((size_t)env.num_entities, sizeof(Entity));
    assert(env.entities != NULL);
    for (int index = 0; index < env.num_entities; index++) {
        allocate_single_sample_trajectory(&env.entities[index]);
        env.entities[index].type = VEHICLE;
        env.entities[index].respawn_timestep = -1;
    }

    set_active_agents(&env);
    assert(env.num_actors == MAX_AGENTS);
    assert(env.active_agent_count == 1);
    assert(env.static_agent_count == MAX_AGENTS - 1);
    assert(env.static_agent_indices[0] == env.trailer_idx);
    for (int index = 0; index < env.num_entities; index++)
        free_entity(&env.entities[index]);
    free(env.entities);
    free(env.active_agent_indices);
    free(env.static_agent_indices);
    free(env.expert_static_agent_indices);
}

static void test_rendering_inherits_owner_style(void) {
    Drive env = make_collision_fixture();
    env.entities[env.tractor_idx].collision_state = VEHICLE_COLLISION;
    env.entities[env.trailer_idx].collision_state = NO_COLLISION;
    assert(render_active_agent_slot(&env, env.tractor_idx) == 0);
    assert(render_active_agent_slot(&env, env.trailer_idx) == 0);
    assert(render_collision_state(&env, env.trailer_idx) == VEHICLE_COLLISION);
    assert(renders_as_articulated_rectangle(&env, env.trailer_idx) == 1);
    assert(env.active_agent_count == 1);
    destroy_fixture(&env);
}

int main(void) {
    test_mutual_collision_ignored_and_trailer_collision_aggregated();
    test_trailer_only_offroad_aggregated();
    test_owner_observation_filter_and_other_agent_visibility();
    test_actor_capacity_reserves_trailer_slot();
    test_rendering_inherits_owner_style();
    puts("tractor-trailer interaction tests passed");
    return 0;
}

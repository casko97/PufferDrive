#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define JERK 1
#define ARTICULATED 2

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3

// Optional binary extension block marker/version (written by drive.py).
#define DRIVE_BIN_EXT_MAGIC 0x54524C52
#define DRIVE_BIN_EXT_VERSION 2

// Grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL                                                                                          \
    30 // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in
       // gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Observation constants
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128

// Maximum number of agents per scene
#ifndef MAX_AGENTS
#define MAX_AGENTS 32
#endif
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define ROAD_FEATURES 7
#define ROAD_FEATURES_ONEHOT 13
#define PARTNER_FEATURES 7
#define PARTNER_TYPE_CHANNELS 1

// Ego features depend on dynamics model
#define EGO_FEATURES_CLASSIC 7
#define EGO_FEATURES_JERK 10

// Observation normalization constants
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f
#define MAX_PARTNER_OBS_DIST2 2500.0f

// Policy-facing type ids used by the current Python policy code.
#define POLICY_TYPE_PADDED 0
#define POLICY_TYPE_VEHICLE_SINGLE 1
#define POLICY_TYPE_PEDESTRIAN 2
#define POLICY_TYPE_CYCLIST 3
#define POLICY_TYPE_VEHICLE_MULTI 4
#define POLICY_TYPE_CLASS_COUNT 5
#define POLICY_REAL_TYPE_CLASS_COUNT (POLICY_TYPE_CLASS_COUNT - 1)
#define EGO_TRAILER_STATE_FEATURES 4

// Goal behavior
#define GOAL_RESPAWN 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES_LEGACY[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float ACCELERATION_VALUES_EXTENDED[9] = {-6.0000f, -4.0000f, -2.0000f, -1.0000f, -0.0000f,
                                                      1.0000f,  2.0000f,  4.0000f,  6.0000f};
static const float STEERING_VALUES[13] = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
                                          0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},  // Middle row (including center)
    {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  // Fourth row
    {-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}   // Bottom row
};

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};
const Color LIGHTYELLOW = (Color){255, 255, 152, 255};
const Color SOFT_YELLOW = (Color){245, 245, 220, 255};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log {
    float episode_return;
    float episode_length;
    float score;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    float offroad_rate;
    float collision_rate;
    float completion_rate;
    float offroad_per_agent;
    float collisions_per_agent;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float speed_at_goal;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
};

typedef struct Entity Entity;
struct Entity {
    int scenario_id;
    int type;
    int id;
    unsigned long long source_track_id_hash;
    int is_trailer;
    int parent_track_index;
    int array_size;
    float *traj_x;
    float *traj_y;
    float *traj_z;
    float *traj_vx;
    float *traj_vy;
    float *traj_vz;
    float *traj_heading;
    int *traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;
    float init_goal_y;
    int mark_as_expert;
    int collision_state;
    int combo_collision_any;
    int combo_collision_tractor_body;
    int combo_collision_trailer_body;
    float metrics_array[5]; // metrics_array: [collision, offroad, reached_goal, lane_aligned
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int current_lane_idx;
    int valid;
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    int current_goal_reached;
    int active_agent;
    int stopped;
    int removed;

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;
    float articulation_angle;
};

void free_entity(Entity *entity) {
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

// Utility functions
float relative_distance(float a, float b) {
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

float normalize_heading(float heading);

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_idx;
    int geometry_idx;
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int *cell_entities_count; // number of entities in each cell of the GridMap
    GridMapEntity **cells;    // list of gridEntities in each cell of the GridMap
    // Extras/Optimizations
    int vision_range;
    int *neighbor_cache_count;               // number of entities in each cells neighbor cache
    GridMapEntity **neighbor_cache_entities; // preallocated array to hold neighbor entities
};

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    Log log;
    Log *logs;
    int num_agents;
    int active_agent_count;
    int *active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity *entities;
    int num_entities;
    int num_actors;
    int num_objects;
    int num_roads;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    int vision_range;
    GridMap *grid_map;
    int *neighbor_offsets;
    int episode_length;
    int termination_mode;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    float goal_speed;
    int max_controlled_agents;
    int logs_capacity;
    int goal_behavior;
    float goal_target_distance;
    char *ini_file;
    char *scenario_id;
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int has_ego_trailer;
    int ego_trailer_track_index;
    float non_kinematic_vehicle_params[13];
    int override_non_kinematic_vehicle_params;
    float non_kinematic_vehicle_params_override[13];
    int force_zero_trailer_articulation_at_init;
    int invalid_initial_trailer_state;
    int num_tracks_to_predict;
    int *tracks_to_predict_indices;
    int init_mode;
    int control_mode;
    int observation_mode;
    int extend_classic_action_space;
};

static inline int classic_acceleration_count(const Drive *env) {
    return env->extend_classic_action_space ? (int)(sizeof(ACCELERATION_VALUES_EXTENDED) / sizeof(float))
                                            : (int)(sizeof(ACCELERATION_VALUES_LEGACY) / sizeof(float));
}

static inline const float *classic_acceleration_values(const Drive *env) {
    return env->extend_classic_action_space ? ACCELERATION_VALUES_EXTENDED : ACCELERATION_VALUES_LEGACY;
}

static inline int classic_joint_action_count(const Drive *env) {
    return classic_acceleration_count(env) * (int)(sizeof(STEERING_VALUES) / sizeof(float));
}

Entity *load_map_binary(const char *filename, Drive *env);

static inline int load_runtime_non_kinematic_params_from_reference_bin(const char *reference_bin, float *out_params) {
    Drive tmp_env = {0};
    Entity *tmp_entities = load_map_binary(reference_bin, &tmp_env);
    if (tmp_entities == NULL) {
        return 0;
    }

    for (int i = 0; i < 13; i++) {
        out_params[i] = tmp_env.non_kinematic_vehicle_params[i];
    }

    for (int i = 0; i < tmp_env.num_entities; i++) {
        free_entity(&tmp_entities[i]);
    }
    free(tmp_entities);
    free(tmp_env.tracks_to_predict_indices);
    return 1;
}

static inline int legacy_bin_has_truck_like_sdc(Drive *env, Entity *entities) {
    // Legacy bins lack trailer metadata. For compatibility, classify by SDC shape only.
    // We allow large background vehicles, but reject clearly truck-like SDC footprints.
    const float max_non_truck_sdc_length = 8.0f;
    const float max_non_truck_sdc_width = 2.6f;
    const float max_non_truck_sdc_height = 3.5f;
    if (env->sdc_track_index < 0 || env->sdc_track_index >= env->num_objects) {
        return 0;
    }
    Entity *sdc = &entities[env->sdc_track_index];
    if (sdc->type != VEHICLE) {
        return 0;
    }
    return (sdc->length > max_non_truck_sdc_length || sdc->width > max_non_truck_sdc_width ||
            sdc->height > max_non_truck_sdc_height);
}

static inline int maybe_add_runtime_sdc_trailer(Drive *env, Entity **entities_ptr) {
    if (!env->override_non_kinematic_vehicle_params) {
        return 1;
    }
    if (env->has_ego_trailer && env->sdc_track_index >= 0 && env->sdc_track_index < env->num_objects &&
        env->ego_trailer_track_index >= 0 && env->ego_trailer_track_index < env->num_objects &&
        env->ego_trailer_track_index != env->sdc_track_index) {
        return 1;
    }
    if (env->sdc_track_index < 0 || env->sdc_track_index >= env->num_objects) {
        return 1;
    }

    Entity *entities = *entities_ptr;
    int old_num_objects = env->num_objects;
    int old_num_entities = env->num_entities;

    Entity *resized = (Entity *)realloc(entities, (old_num_entities + 1) * sizeof(Entity));
    if (resized == NULL) {
        return 0;
    }
    entities = resized;
    memmove(&entities[old_num_objects + 1], &entities[old_num_objects],
            (old_num_entities - old_num_objects) * sizeof(Entity));

    Entity *sdc = &entities[env->sdc_track_index];
    Entity *trailer = &entities[old_num_objects];
    memset(trailer, 0, sizeof(Entity));

    trailer->scenario_id = sdc->scenario_id;
    trailer->type = VEHICLE;
    trailer->id = -1000000 - old_num_objects;
    trailer->source_track_id_hash = 0ULL;
    trailer->is_trailer = 1;
    trailer->parent_track_index = env->sdc_track_index;
    trailer->array_size = sdc->array_size;

    int size = trailer->array_size;
    trailer->traj_x = (float *)malloc(size * sizeof(float));
    trailer->traj_y = (float *)malloc(size * sizeof(float));
    trailer->traj_z = (float *)malloc(size * sizeof(float));
    trailer->traj_vx = (float *)malloc(size * sizeof(float));
    trailer->traj_vy = (float *)malloc(size * sizeof(float));
    trailer->traj_vz = (float *)malloc(size * sizeof(float));
    trailer->traj_heading = (float *)malloc(size * sizeof(float));
    trailer->traj_valid = (int *)malloc(size * sizeof(int));
    if (trailer->traj_x == NULL || trailer->traj_y == NULL || trailer->traj_z == NULL || trailer->traj_vx == NULL ||
        trailer->traj_vy == NULL || trailer->traj_vz == NULL || trailer->traj_heading == NULL ||
        trailer->traj_valid == NULL) {
        free_entity(trailer);
        return 0;
    }

    memcpy(trailer->traj_x, sdc->traj_x, size * sizeof(float));
    memcpy(trailer->traj_y, sdc->traj_y, size * sizeof(float));
    memcpy(trailer->traj_z, sdc->traj_z, size * sizeof(float));
    memcpy(trailer->traj_vx, sdc->traj_vx, size * sizeof(float));
    memcpy(trailer->traj_vy, sdc->traj_vy, size * sizeof(float));
    memcpy(trailer->traj_vz, sdc->traj_vz, size * sizeof(float));
    memcpy(trailer->traj_heading, sdc->traj_heading, size * sizeof(float));
    memcpy(trailer->traj_valid, sdc->traj_valid, size * sizeof(int));

    trailer->width = env->non_kinematic_vehicle_params[3];
    trailer->length = env->non_kinematic_vehicle_params[1];
    trailer->height = env->non_kinematic_vehicle_params[5];
    trailer->goal_position_x = sdc->goal_position_x;
    trailer->goal_position_y = sdc->goal_position_y;
    trailer->goal_position_z = sdc->goal_position_z;
    trailer->mark_as_expert = 1;

    env->has_ego_trailer = 1;
    env->ego_trailer_track_index = old_num_objects;
    env->num_objects = old_num_objects + 1;
    env->num_entities = old_num_entities + 1;

    *entities_ptr = entities;
    return 1;
}

static inline bool has_valid_ego_trailer_pair(Drive *env) {
    if (!env->has_ego_trailer)
        return false;
    if (env->sdc_track_index < 0 || env->sdc_track_index >= env->num_objects)
        return false;
    if (env->ego_trailer_track_index < 0 || env->ego_trailer_track_index >= env->num_objects)
        return false;
    if (env->ego_trailer_track_index == env->sdc_track_index)
        return false;
    return true;
}

static inline bool is_ego_or_trailer_pair(Drive *env, int a, int b) {
    if (!has_valid_ego_trailer_pair(env))
        return false;
    return ((a == env->sdc_track_index && b == env->ego_trailer_track_index) ||
            (a == env->ego_trailer_track_index && b == env->sdc_track_index));
}

static inline void freeze_ego_trailer_entity(Drive *env) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (trailer->removed || trailer->x == INVALID_POSITION)
        return;

    trailer->vx = 0.0f;
    trailer->vy = 0.0f;
}

static inline void update_ego_trailer_pose(Drive *env) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *tractor = &env->entities[env->sdc_track_index];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (tractor->removed || trailer->removed)
        return;
    if (tractor->x == INVALID_POSITION || trailer->x == INVALID_POSITION)
        return;
    if (tractor->stopped) {
        freeze_ego_trailer_entity(env);
        return;
    }

    // Use dataset non-kinematic geometry parameters from extension v2.
    float tractor2hitch = env->non_kinematic_vehicle_params[6];
    float trailer2hitch = env->non_kinematic_vehicle_params[7];
    float effective_length = fmaxf(0.5f, trailer->length - trailer2hitch);
    float theta_tractor = tractor->heading;
    float theta_trailer = trailer->heading;

    // Update trailer yaw using articulation kinematics.
    float tractor_speed = sqrtf(tractor->vx * tractor->vx + tractor->vy * tractor->vy);
    float tractor_signed_speed =
        copysignf(tractor_speed, tractor->vx * tractor->heading_x + tractor->vy * tractor->heading_y);
    float articulation = theta_tractor - theta_trailer;
    if (articulation > M_PI)
        articulation -= 2.0f * M_PI;
    if (articulation < -M_PI)
        articulation += 2.0f * M_PI;
    theta_trailer = theta_trailer + (tractor_signed_speed / effective_length) * sinf(articulation) * env->dt;
    if (theta_trailer > M_PI)
        theta_trailer -= 2.0f * M_PI;
    if (theta_trailer < -M_PI)
        theta_trailer += 2.0f * M_PI;

    // Compute trailer center from hitch point and updated trailer yaw.
    float x_hitch = tractor->x + (-tractor->length * 0.5f + tractor2hitch) * cosf(theta_tractor);
    float y_hitch = tractor->y + (-tractor->length * 0.5f + tractor2hitch) * sinf(theta_tractor);
    float x_rear = x_hitch - effective_length * cosf(theta_trailer);
    float y_rear = y_hitch - effective_length * sinf(theta_trailer);
    float new_x = x_rear + 0.5f * trailer->length * cosf(theta_trailer);
    float new_y = y_rear + 0.5f * trailer->length * sinf(theta_trailer);

    float old_x = trailer->x;
    float old_y = trailer->y;
    trailer->x = new_x;
    trailer->y = new_y;
    trailer->heading = theta_trailer;
    trailer->heading_x = cosf(theta_trailer);
    trailer->heading_y = sinf(theta_trailer);
    trailer->vx = (trailer->x - old_x) / fmaxf(env->dt, 1e-4f);
    trailer->vy = (trailer->y - old_y) / fmaxf(env->dt, 1e-4f);
}

static inline void sync_ego_trailer_entity_to_trajectory_step(Drive *env, int step) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (step < 0)
        step = 0;
    if (step >= trailer->array_size)
        step = trailer->array_size - 1;

    if (trailer->traj_valid && trailer->traj_valid[step] == 0) {
        trailer->x = INVALID_POSITION;
        trailer->y = INVALID_POSITION;
        trailer->z = 0.0f;
        trailer->vx = 0.0f;
        trailer->vy = 0.0f;
        trailer->vz = 0.0f;
        trailer->heading = 0.0f;
        trailer->heading_x = 1.0f;
        trailer->heading_y = 0.0f;
        trailer->valid = 0;
        return;
    }

    trailer->x = trailer->traj_x[step];
    trailer->y = trailer->traj_y[step];
    trailer->z = trailer->traj_z[step];
    trailer->vx = trailer->traj_vx[step];
    trailer->vy = trailer->traj_vy[step];
    trailer->vz = trailer->traj_vz[step];
    trailer->heading = trailer->traj_heading[step];
    trailer->heading_x = cosf(trailer->heading);
    trailer->heading_y = sinf(trailer->heading);
    trailer->valid = trailer->traj_valid[step];
}

static inline void project_articulated_ego_trailer_pose(Drive *env, int update_velocity_from_pose_delta) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *tractor = &env->entities[env->sdc_track_index];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (tractor->removed || trailer->removed)
        return;
    if (tractor->x == INVALID_POSITION)
        return;
    if (tractor->stopped) {
        freeze_ego_trailer_entity(env);
        return;
    }

    float tractor2hitch = env->non_kinematic_vehicle_params[6];
    float trailer2hitch = env->non_kinematic_vehicle_params[7];
    float effective_length = fmaxf(0.5f, trailer->length - trailer2hitch);
    float theta_tractor = tractor->heading;
    float theta_trailer = normalize_heading(theta_tractor - tractor->articulation_angle);

    float x_hitch = tractor->x + (-tractor->length * 0.5f + tractor2hitch) * cosf(theta_tractor);
    float y_hitch = tractor->y + (-tractor->length * 0.5f + tractor2hitch) * sinf(theta_tractor);
    float x_rear = x_hitch - effective_length * cosf(theta_trailer);
    float y_rear = y_hitch - effective_length * sinf(theta_trailer);
    float new_x = x_rear + 0.5f * trailer->length * cosf(theta_trailer);
    float new_y = y_rear + 0.5f * trailer->length * sinf(theta_trailer);

    float old_x = trailer->x;
    float old_y = trailer->y;
    trailer->x = new_x;
    trailer->y = new_y;
    trailer->heading = theta_trailer;
    trailer->heading_x = cosf(theta_trailer);
    trailer->heading_y = sinf(theta_trailer);

    if (update_velocity_from_pose_delta) {
        trailer->vx = (trailer->x - old_x) / fmaxf(env->dt, 1e-4f);
        trailer->vy = (trailer->y - old_y) / fmaxf(env->dt, 1e-4f);
    } else {
        trailer->vx = tractor->vx;
        trailer->vy = tractor->vy;
    }
}

static inline void initialize_articulated_ego_trailer_state(Drive *env, int force_zero_articulation) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *tractor = &env->entities[env->sdc_track_index];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (tractor->removed || trailer->removed)
        return;
    if (tractor->x == INVALID_POSITION || trailer->x == INVALID_POSITION)
        return;

    tractor->articulation_angle =
        force_zero_articulation ? 0.0f : normalize_heading(tractor->heading - trailer->heading);
    trailer->articulation_angle = tractor->articulation_angle;
    project_articulated_ego_trailer_pose(env, 0);
}

static inline void apply_non_kinematic_runtime_override(Drive *env) {
    if (!env->override_non_kinematic_vehicle_params)
        return;
    for (int i = 0; i < 13; i++) {
        env->non_kinematic_vehicle_params[i] = env->non_kinematic_vehicle_params_override[i];
    }

    if (env->sdc_track_index < 0 || env->sdc_track_index >= env->num_objects)
        return;
    Entity *tractor = &env->entities[env->sdc_track_index];
    tractor->length = env->non_kinematic_vehicle_params[0];
    tractor->width = env->non_kinematic_vehicle_params[2];
    tractor->height = env->non_kinematic_vehicle_params[4];

    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    trailer->length = env->non_kinematic_vehicle_params[1];
    trailer->width = env->non_kinematic_vehicle_params[3];
    trailer->height = env->non_kinematic_vehicle_params[5];
}

static inline void force_zero_trailer_articulation_pose_from_params(Drive *env) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *tractor = &env->entities[env->sdc_track_index];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    if (tractor->removed || trailer->removed)
        return;
    if (tractor->x == INVALID_POSITION || trailer->x == INVALID_POSITION)
        return;

    float theta = tractor->heading;
    float tractor2hitch = env->non_kinematic_vehicle_params[6];
    float trailer2hitch = env->non_kinematic_vehicle_params[7];
    float effective_length = fmaxf(0.5f, trailer->length - trailer2hitch);

    float x_hitch = tractor->x + (-tractor->length * 0.5f + tractor2hitch) * cosf(theta);
    float y_hitch = tractor->y + (-tractor->length * 0.5f + tractor2hitch) * sinf(theta);
    float x_rear = x_hitch - effective_length * cosf(theta);
    float y_rear = y_hitch - effective_length * sinf(theta);

    trailer->x = x_rear + 0.5f * trailer->length * cosf(theta);
    trailer->y = y_rear + 0.5f * trailer->length * sinf(theta);
    trailer->heading = theta;
    trailer->heading_x = cosf(theta);
    trailer->heading_y = sinf(theta);
    trailer->vx = tractor->vx;
    trailer->vy = tractor->vy;
}

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];

        env->log.goals_reached_this_episode += e->goals_reached_this_episode;
        env->log.goals_sampled_this_episode += e->goals_sampled_this_episode;

        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float offroad_per_agent = env->logs[i].offroad_per_agent;
        env->log.offroad_per_agent += offroad_per_agent;
        float collisions_per_agent = env->logs[i].collisions_per_agent;
        env->log.collisions_per_agent += collisions_per_agent;

        float frac_goal_reached = e->goals_reached_this_episode / e->goals_sampled_this_episode;

        // Update score, which is an aggregate measure whether the agent fully solved its task
        // Note: When resampling goals, performance is relative to the number of goals sampled
        float threshold = 0.99f; // Default threshold for 1 goal
        if (e->goals_sampled_this_episode == 2.0f) {
            threshold = 0.5f; // Require ≥50% completion for 2 goals
        } else if (e->goals_sampled_this_episode < 5.0f) {
            threshold = 0.8f; // Require ≥80% completion for 3-4 goals
        } else {
            threshold = 0.9f; // Require ≥90% completion for 5+ goals
        }

        int collision_occurred =
            (env->goal_behavior == GOAL_RESPAWN) ? e->collided_before_goal : env->logs[i].collision_rate;
        if (frac_goal_reached > threshold && !collision_occurred) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && frac_goal_reached < 1.0f) {
            env->log.dnf_rate += 1.0f;
        }
        int lane_aligned = env->logs[i].lane_alignment_rate;
        env->log.lane_alignment_rate += lane_aligned;
        env->log.speed_at_goal += env->logs[i].speed_at_goal;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        env->log.n += 1;
    }
}

Entity *load_map_binary(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    env->has_ego_trailer = 0;
    env->ego_trailer_track_index = -1;
    for (int i = 0; i < 13; i++) {
        env->non_kinematic_vehicle_params[i] = 0.0f;
    }

    // Read sdc_track_index
    fread(&env->sdc_track_index, sizeof(int), 1, file);

    // Read tracks_to_predict
    fread(&env->num_tracks_to_predict, sizeof(int), 1, file);
    if (env->num_tracks_to_predict > 0) {
        env->tracks_to_predict_indices = (int *)malloc(env->num_tracks_to_predict * sizeof(int));

        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            fread(&env->tracks_to_predict_indices[i], sizeof(int), 1, file);
        }
    } else {
        env->tracks_to_predict_indices = NULL;
    }

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].scenario_id, sizeof(int), 1, file);
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].id, sizeof(int), 1, file);
        entities[i].source_track_id_hash = 0ULL;
        entities[i].is_trailer = 0;
        entities[i].parent_track_index = -1;
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float *)malloc(size * sizeof(float));
        entities[i].traj_y = (float *)malloc(size * sizeof(float));
        entities[i].traj_z = (float *)malloc(size * sizeof(float));
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float *)malloc(size * sizeof(float));
            entities[i].traj_vy = (float *)malloc(size * sizeof(float));
            entities[i].traj_vz = (float *)malloc(size * sizeof(float));
            entities[i].traj_heading = (float *)malloc(size * sizeof(float));
            entities[i].traj_valid = (int *)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
    }

    // Optional extension block (v2). Legacy bins may end here.
    int extension_magic = 0;
    size_t read_magic = fread(&extension_magic, sizeof(int), 1, file);
    if (read_magic != 1) {
        if (legacy_bin_has_truck_like_sdc(env, entities)) {
            fclose(file);
            for (int i = 0; i < env->num_entities; i++) {
                free_entity(&entities[i]);
            }
            free(entities);
            free(env->tracks_to_predict_indices);
            env->tracks_to_predict_indices = NULL;
            raise_error_with_message(
                ERROR_INITIALIZATION_FAILED,
                "Legacy map binary without extension is only supported for non-truck scenes");
        }
        if (!maybe_add_runtime_sdc_trailer(env, &entities)) {
            fclose(file);
            for (int i = 0; i < env->num_entities; i++) {
                free_entity(&entities[i]);
            }
            free(entities);
            free(env->tracks_to_predict_indices);
            env->tracks_to_predict_indices = NULL;
            raise_error_with_message(ERROR_INITIALIZATION_FAILED, "Failed to allocate runtime synthetic trailer");
        }
        env->entities = entities;
        apply_non_kinematic_runtime_override(env);
        fclose(file);
        return entities;
    }
    if (extension_magic != DRIVE_BIN_EXT_MAGIC) {
        // Legacy format: no trailer extension metadata block.
        if (legacy_bin_has_truck_like_sdc(env, entities)) {
            fclose(file);
            for (int i = 0; i < env->num_entities; i++) {
                free_entity(&entities[i]);
            }
            free(entities);
            free(env->tracks_to_predict_indices);
            env->tracks_to_predict_indices = NULL;
            raise_error_with_message(
                ERROR_INITIALIZATION_FAILED,
                "Legacy map binary without extension is only supported for non-truck scenes");
        }
        if (!maybe_add_runtime_sdc_trailer(env, &entities)) {
            fclose(file);
            for (int i = 0; i < env->num_entities; i++) {
                free_entity(&entities[i]);
            }
            free(entities);
            free(env->tracks_to_predict_indices);
            env->tracks_to_predict_indices = NULL;
            raise_error_with_message(ERROR_INITIALIZATION_FAILED, "Failed to allocate runtime synthetic trailer");
        }
        env->entities = entities;
        apply_non_kinematic_runtime_override(env);
        fclose(file);
        return entities;
    }

    int extension_version = 0;
    if (fread(&extension_version, sizeof(int), 1, file) != 1 || extension_version != DRIVE_BIN_EXT_VERSION) {
        fclose(file);
        free(entities);
        raise_error_with_message(ERROR_INITIALIZATION_FAILED,
                                 "Map binary extension version mismatch: expected %d got %d", DRIVE_BIN_EXT_VERSION,
                                 extension_version);
    }

    fread(&env->has_ego_trailer, sizeof(int), 1, file);
    fread(&env->ego_trailer_track_index, sizeof(int), 1, file);

    int object_meta_count = 0;
    fread(&object_meta_count, sizeof(int), 1, file);
    for (int i = 0; i < object_meta_count; i++) {
        unsigned long long source_hash = 0ULL;
        int is_trailer = 0;
        int parent_track_index = -1;
        fread(&source_hash, sizeof(unsigned long long), 1, file);
        fread(&is_trailer, sizeof(int), 1, file);
        fread(&parent_track_index, sizeof(int), 1, file);
        if (i < env->num_objects) {
            entities[i].source_track_id_hash = source_hash;
            entities[i].is_trailer = is_trailer;
            entities[i].parent_track_index = parent_track_index;
        }
    }

    int vehicle_param_count = 0;
    fread(&vehicle_param_count, sizeof(int), 1, file);
    if (vehicle_param_count != 13) {
        fclose(file);
        free(entities);
        raise_error_with_message(ERROR_INITIALIZATION_FAILED,
                                 "Map binary non_kinematic_vehicle_params count mismatch: expected 13 got %d",
                                 vehicle_param_count);
    }
    for (int i = 0; i < 13; i++) {
        fread(&env->non_kinematic_vehicle_params[i], sizeof(float), 1, file);
    }
    if (!maybe_add_runtime_sdc_trailer(env, &entities)) {
        fclose(file);
        for (int i = 0; i < env->num_entities; i++) {
            free_entity(&entities[i]);
        }
        free(entities);
        free(env->tracks_to_predict_indices);
        env->tracks_to_predict_indices = NULL;
        raise_error_with_message(ERROR_INITIALIZATION_FAILED, "Failed to allocate runtime synthetic trailer");
    }
    env->entities = entities;
    apply_non_kinematic_runtime_override(env);

    fclose(file);
    return entities;
}

void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity *e = &env->entities[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size)
            step = e->array_size - 1;
        if (step < 0)
            step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];
        if (e->type > CYCLIST || e->type == 0) {
            continue;
        }
        if (is_active == 0) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[env->init_steps];
            e->vy = e->traj_vy[env->init_steps];
            e->vz = e->traj_vz[env->init_steps];
        }
        e->heading = e->traj_heading[env->init_steps];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[env->init_steps];
        e->collision_state = 0;
        e->combo_collision_any = 0;
        e->combo_collision_tractor_body = 0;
        e->combo_collision_trailer_body = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->respawn_timestep = -1;
        e->stopped = 0;
        e->removed = 0;
        e->respawn_count = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        e->wheelbase = 0.6f * e->length;
        e->articulation_angle = 0.0f;
    }
    if (env->dynamics_model == ARTICULATED && has_valid_ego_trailer_pair(env)) {
        initialize_articulated_ego_trailer_state(env, env->force_zero_trailer_articulation_at_init);
    } else if (env->force_zero_trailer_articulation_at_init) {
        force_zero_trailer_articulation_pose_from_params(env);
    }
}

int getGridIndex(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x ||
        env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1; // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;     // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y; // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);        // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);        // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1; // Return -1 for out of bounds
    }
    int index = (gridY * env->grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx, int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
               "%d\n",
               grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

void init_grid_map(Drive *env) {
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap *)malloc(sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            // Check all points in the trajectory for road elements
            Entity *e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION)
                    continue;
                if (e->traj_y[j] == INVALID_POSITION)
                    continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if (e->traj_x[j] < top_left_x)
                    top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x)
                    bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] > top_left_y)
                    top_left_y = e->traj_y[j];
                if (e->traj_y[j] < bottom_right_y)
                    bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **)calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    int cell_entities_insert_index[grid_cell_count]; // Helper array for insertion index
    memset(cell_entities_insert_index, 0, grid_cell_count * sizeof(int));

    // Initialize grid cells
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        env->grid_map->cells[grid_index] =
            (GridMapEntity *)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for (int i = 0; i < grid_cell_count; i++) {
        if (cell_entities_insert_index[i] != 0) {
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Populate grid cells
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 &&
            env->entities[i].type < 7) { // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
}

void init_neighbor_offsets(Drive *env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->grid_map->vision_range * env->grid_map->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->grid_map->vision_range * env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range / 2 && abs(y) <= env->grid_map->vision_range / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **)calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *)calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] =
            (GridMapEntity *)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx], env->grid_map->cells[src_idx],
                   grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, GridMapEntity *entities, int max_entities) {
    GridMap *grid_map = env->grid_map;
    if (cell_idx < 0 || cell_idx >= (grid_map->grid_cols * grid_map->grid_rows)) {
        return 0; // Invalid cell index
    }

    int count = grid_map->neighbor_cache_count[cell_idx];
    // Limit to available space
    if (count > max_entities) {
        count = max_entities;
    }
    memcpy(entities, grid_map->neighbor_cache_entities[cell_idx], count * sizeof(GridMapEntity));
    return count;
}

void set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                // Assume a validity flag exists (e.g., valid[j]); adjust if not available
                if (env->entities[i].traj_valid[j]) { // Add validity check if applicable
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION)
                    continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

static inline void move_expert_trajectory_only(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        agent->vz = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    if (agent->traj_valid && agent->traj_valid[t] == 0) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        agent->vz = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->vx = agent->traj_vx[t];
    agent->vy = agent->traj_vy[t];
    agent->vz = agent->traj_vz[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    move_expert_trajectory_only(env, agent_idx);

    // Keep replayed trailer pose consistent with articulated geometry in extension v2.
    if (has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index) {
        if (env->dynamics_model == ARTICULATED) {
            sync_ego_trailer_entity_to_trajectory_step(env, env->timestep);
            initialize_articulated_ego_trailer_state(env, env->force_zero_trailer_articulation_at_init);
        } else {
            update_ego_trailer_pose(env);
        }
    }
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive *env, float x, float y, GridMapEntity *entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity *car1, Entity *car2) {
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

    // Check each axis
    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2) {
            return 0; // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1; // Collision
}

int collision_check_with_sources(Drive *env, int agent_idx, int *agent_body_collided, int *trailer_body_collided) {
    Entity *agent = &env->entities[agent_idx];
    Entity *ego_trailer = NULL;
    bool check_ego_trailer = has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index;
    if (check_ego_trailer) {
        ego_trailer = &env->entities[env->ego_trailer_track_index];
    }

    if (agent->x == INVALID_POSITION)
        return -1;

    if (agent_body_collided != NULL)
        *agent_body_collided = 0;
    if (trailer_body_collided != NULL)
        *trailer_body_collided = 0;

    int car_collided_with_index = -1;

    if (agent->respawn_timestep != -1)
        return car_collided_with_index; // Skip respawning entities

    for (int i = 0; i < MAX_AGENTS; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        if (is_ego_or_trailer_pair(env, agent_idx, index))
            continue;
        Entity *entity = &env->entities[index];
        if (entity->respawn_timestep != -1)
            continue; // Skip respawning entities
        float x1 = entity->x;
        float y1 = entity->y;
        float dist_agent = ((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        float dist_trailer = dist_agent;
        if (check_ego_trailer && ego_trailer != NULL) {
            dist_trailer = ((x1 - ego_trailer->x) * (x1 - ego_trailer->x) + (y1 - ego_trailer->y) * (y1 - ego_trailer->y));
        }
        if (dist_agent > 225.0f && dist_trailer > 225.0f)
            continue;
        int agent_hit = check_aabb_collision(agent, entity);
        int trailer_hit = 0;
        if (check_ego_trailer && ego_trailer != NULL) {
            trailer_hit = check_aabb_collision(ego_trailer, entity);
        }
        if (agent_hit || trailer_hit) {
            if (agent_body_collided != NULL && agent_hit)
                *agent_body_collided = 1;
            if (trailer_body_collided != NULL && trailer_hit)
                *trailer_body_collided = 1;
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

int collision_check(Drive *env, int agent_idx) {
    return collision_check_with_sources(env, agent_idx, NULL, NULL);
}

static inline int has_invalid_initial_sdc_trailer_collision(Drive *env) {
    if (!env->override_non_kinematic_vehicle_params)
        return 0;
    if (!has_valid_ego_trailer_pair(env))
        return 0;
    if (env->sdc_track_index < 0 || env->sdc_track_index >= env->num_objects)
        return 0;
    if (env->ego_trailer_track_index < 0 || env->ego_trailer_track_index >= env->num_objects)
        return 0;

    Entity *ego_trailer = &env->entities[env->ego_trailer_track_index];
    if (ego_trailer->x == INVALID_POSITION || ego_trailer->y == INVALID_POSITION)
        return 0;

    for (int i = 0; i < env->num_actors; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->active_agent_count + env->static_agent_count) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index < 0 || index >= env->num_objects)
            continue;
        if (index == env->sdc_track_index || index == env->ego_trailer_track_index)
            continue;
        Entity *other = &env->entities[index];
        if (other->x == INVALID_POSITION || other->y == INVALID_POSITION)
            continue;
        if (other->removed)
            continue;
        if (check_aabb_collision(ego_trailer, other)) {
            return 1;
        }
    }

    // Mirror runtime off-road semantics for trailer body at initial state:
    // invalid if trailer box intersects any ROAD_EDGE segment.
    float half_length = ego_trailer->length / 2.0f;
    float half_width = ego_trailer->width / 2.0f;
    float cos_heading = cosf(ego_trailer->heading);
    float sin_heading = sinf(ego_trailer->heading);
    float trailer_corners[4][2];
    for (int i = 0; i < 4; i++) {
        trailer_corners[i][0] = ego_trailer->x + (offsets[i][0] * half_length * cos_heading -
                                                  offsets[i][1] * half_width * sin_heading);
        trailer_corners[i][1] = ego_trailer->y + (offsets[i][0] * half_length * sin_heading +
                                                  offsets[i][1] * half_width * cos_heading);
    }
    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type != ROAD_EDGE)
            continue;
        if (road->array_size < 2)
            continue;
        for (int j = 0; j < road->array_size - 1; j++) {
            float start[2] = {road->traj_x[j], road->traj_y[j]};
            float end[2] = {road->traj_x[j + 1], road->traj_y[j + 1]};
            for (int k = 0; k < 4; k++) {
                int next = (k + 1) % 4;
                if (check_line_intersection(trailer_corners[k], trailer_corners[next], start, end)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}

int check_lane_aligned(Entity *car, Entity *lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->array_size < 2)
        return 0;

    // Clamp geometry index to valid segment range [0, array_size-2]
    if (geometry_idx < 0)
        geometry_idx = 0;
    if (geometry_idx >= lane->array_size - 1)
        geometry_idx = lane->array_size - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->traj_x[geometry_idx] - lane->traj_x[geometry_idx - 1];
        heading_y1 = lane->traj_y[geometry_idx] - lane->traj_y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
        heading_y1 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];
    }

    float heading_x2 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
    float heading_y2 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI)
        heading -= 2.0f * M_PI;
    if (heading < -M_PI)
        heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    if (heading_diff > M_PI)
        heading_diff = 2.0f * M_PI - heading_diff;

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

void reset_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->collision_state = 0;
    agent->combo_collision_any = 0;
    agent->combo_collision_tractor_body = 0;
    agent->combo_collision_trailer_body = 0;
}

float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // The segment is a point
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }

    // Calculate the t that minimizes the distance
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to the segment
    if (t < 0)
        t = 0;
    else if (t > 1)
        t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

void compute_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    bool check_ego_trailer = has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index;
    Entity *ego_trailer = check_ego_trailer ? &env->entities[env->ego_trailer_track_index] : NULL;

    reset_agent_metrics(env, agent_idx);

    if (agent->x == INVALID_POSITION)
        return; // invalid agent position

    int collided = 0;
    int offroad_agent_body_collided = 0;
    int offroad_trailer_body_collided = 0;
    int vehicle_agent_body_collided = 0;
    int vehicle_trailer_body_collided = 0;
    int final_agent_body_collided = 0;
    int final_trailer_body_collided = 0;
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float min_distance = (float)INT16_MAX;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    float trailer_corners[4][2];
    if (check_ego_trailer && ego_trailer != NULL) {
        float trailer_half_length = ego_trailer->length / 2.0f;
        float trailer_half_width = ego_trailer->width / 2.0f;
        float trailer_cos_heading = cosf(ego_trailer->heading);
        float trailer_sin_heading = sinf(ego_trailer->heading);
        for (int i = 0; i < 4; i++) {
            trailer_corners[i][0] = ego_trailer->x +
                                    (offsets[i][0] * trailer_half_length * trailer_cos_heading -
                                     offsets[i][1] * trailer_half_width * trailer_sin_heading);
            trailer_corners[i][1] = ego_trailer->y +
                                    (offsets[i][0] * trailer_half_length * trailer_sin_heading +
                                     offsets[i][1] * trailer_half_width * trailer_cos_heading);
        }
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 50]; // Tractor neighborhood + optional trailer neighborhood
    int list_size =
        checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 50, collision_offsets, 25);
    if (check_ego_trailer && ego_trailer != NULL) {
        int remaining = (MAX_ENTITIES_PER_CELL * 50) - list_size;
        if (remaining > 0) {
            list_size += checkNeighbors(env, ego_trailer->x, ego_trailer->y, &entity_list[list_size], remaining,
                                        collision_offsets, 25);
        }
    }
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        if (entity_list[i].entity_idx == agent_idx)
            continue;
        Entity *entity;
        entity = &env->entities[entity_list[i].entity_idx];

        // Check for offroad collision with road edges
        if (entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    offroad_agent_body_collided = 1;
                    break;
                }
                if (check_ego_trailer && ego_trailer != NULL &&
                    check_line_intersection(trailer_corners[k], trailer_corners[next], start, end)) {
                    collided = OFFROAD;
                    offroad_trailer_body_collided = 1;
                    break;
                }
            }
        }

        if (collided == OFFROAD)
            break;

        // Find closest point on the road centerline to the agent
        if (entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->x, agent->y, start[0], start[1], end[0], end[1]);
            float heading_diff = fabsf(atan2f(end[1] - start[1], end[0] - start[0]) - agent->heading);

            // Normalize heading difference to [0, pi]
            if (heading_diff > M_PI)
                heading_diff = 2.0f * M_PI - heading_diff;

            // Penalize if heading differs by more than 30 degrees
            if (heading_diff > (M_PI / 6.0f))
                dist += 3.0f;

            if (dist < min_distance) {
                min_distance = dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane and set current lane
    // 4.0m threshold: agents more than 4 meters from any lane are considered off-road
    if (min_distance > 4.0f || closest_lane_entity_idx == -1) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->current_lane_idx = -1;
    } else {
        agent->current_lane_idx = closest_lane_entity_idx;
        int lane_aligned =
            check_lane_aligned(agent, &env->entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;
    }

    // Check for vehicle collisions
    int car_collided_with_index =
        collision_check_with_sources(env, agent_idx, &vehicle_agent_body_collided, &vehicle_trailer_body_collided);
    if (car_collided_with_index != -1)
        collided = VEHICLE_COLLISION;

    agent->collision_state = collided;
    if (collided == VEHICLE_COLLISION) {
        final_agent_body_collided = vehicle_agent_body_collided;
        final_trailer_body_collided = vehicle_trailer_body_collided;
    } else if (collided == OFFROAD) {
        final_agent_body_collided = offroad_agent_body_collided;
        final_trailer_body_collided = offroad_trailer_body_collided;
    }
    if (check_ego_trailer) {
        agent->combo_collision_any = (collided > 0) ? 1 : 0;
        agent->combo_collision_tractor_body = final_agent_body_collided;
        agent->combo_collision_trailer_body = final_trailer_body_collided;
    } else {
        agent->combo_collision_any = (collided > 0) ? 1 : 0;
        agent->combo_collision_tractor_body = (collided > 0) ? 1 : 0;
        agent->combo_collision_trailer_body = 0;
    }

    if (collided == VEHICLE_COLLISION) {
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
            if (agent_idx == env->sdc_track_index)
                freeze_ego_trailer_entity(env);
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            Entity *agent_collided = &env->entities[car_collided_with_index];
            agent->removed = 1;
            agent_collided->removed = 1;
            agent->x = agent->y = -10000.0f;
            agent_collided->x = agent_collided->y = -10000.0f;
        }
    }
    if (collided == OFFROAD) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
            if (agent_idx == env->sdc_track_index)
                freeze_ego_trailer_entity(env);
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
            agent->x = agent->y = -10000.0f;
        }
    }

    return;
}

bool should_control_agent(Drive *env, int agent_idx) {

    // Check if we have room for more agents or are already at capacity
    if (env->active_agent_count >= env->num_agents) {
        return false;
    }

    Entity *entity = &env->entities[agent_idx];

    // Keep ego trailer as a dependent body, never directly policy-controlled.
    if (has_valid_ego_trailer_pair(env) && agent_idx == env->ego_trailer_track_index) {
        return false;
    }

    if (env->control_mode == CONTROL_SDC_ONLY) {
        return agent_idx == env->sdc_track_index;
    }

    bool is_vehicle = (entity->type == VEHICLE);
    bool is_ped_or_bike = (entity->type == PEDESTRIAN || entity->type == CYCLIST);
    bool type_is_valid = false;

    switch (env->control_mode) {
    case CONTROL_WOSAC:
        // Valid types only, ignore expert flag and goal distance
        return (is_vehicle || is_ped_or_bike);

    case CONTROL_VEHICLES:
        type_is_valid = is_vehicle;
        break;

    default:
        type_is_valid = (is_vehicle || is_ped_or_bike);
        break;
    }

    // Filter invalid types or experts
    if (!type_is_valid || entity->mark_as_expert) {
        return false;
    }

    // Check distance to goal in agent's local frame
    float cos_heading = cosf(entity->traj_heading[0]);
    float sin_heading = sinf(entity->traj_heading[0]);
    float goal_dx = entity->goal_position_x - entity->traj_x[0];
    float goal_dy = entity->goal_position_y - entity->traj_y[0];

    // Transform to agent's local frame
    float local_goal_x = goal_dx * cos_heading + goal_dy * sin_heading;
    float local_goal_y = -goal_dx * sin_heading + goal_dy * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, local_goal_x, local_goal_y);

    return distance_to_goal >= MIN_DISTANCE_TO_GOAL;
}

void set_active_agents(Drive *env) {

    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_actors = 0;                // Total agents created

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    if (env->num_agents == 0) {
        env->num_agents = MAX_AGENTS;
    }

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_objects && env->num_actors < MAX_AGENTS; i++) {

        Entity *entity = &env->entities[i];

        // Skip if not valid at initialization
        if (entity->traj_valid[env->init_steps] != 1) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (env->init_mode == INIT_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (entity->type == VEHICLE);
        } else { // Control all agents
            should_create = (entity->type == VEHICLE || entity->type == PEDESTRIAN || entity->type == CYCLIST);
        }
        // Ensure ego trailer is spawned for collision/offroad checks even when not controlled.
        if (has_valid_ego_trailer_pair(env) && i == env->ego_trailer_track_index) {
            should_create = true;
        }

        if (!should_create)
            continue;

        env->num_actors++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = false;

        is_controlled = should_control_agent(env, i);

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->entities[i].active_agent = 0;
            if (env->entities[i].mark_as_expert == 1 || env->active_agent_count == env->num_agents) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *)malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for (int i = 0; i < env->static_agent_count; i++) {
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }

    return;
}

void remove_bad_trajectories(Drive *env) {

    if (env->control_mode != CONTROL_WOSAC) {
        return; // Leave all trajectories in WOSAC control mode
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < env->episode_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i])
                continue;
            env->entities[static_agent_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_agent_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

void init_goal_positions(Drive *env) {
    for (int x = 0; x < env->active_agent_count; x++) {
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }
}

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (startup_timing_enabled) {
        startup_timing_init_calls++;
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        env->entities = load_map_binary(env->map_name, env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_load_map_binary += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        set_means(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_set_means += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        init_grid_map(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_init_grid_map += elapsed_seconds(t0, t1);
    } else {
        env->entities = load_map_binary(env->map_name, env);
        set_means(env);
        init_grid_map(env);
    }
    int vision_range = env->vision_range > 0 ? env->vision_range : 21;
    if (vision_range < 3) {
        vision_range = 3;
    }
    if (vision_range % 2 == 0) {
        vision_range -= 1;
    }
    env->grid_map->vision_range = vision_range;
    if (startup_timing_enabled) {
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        init_neighbor_offsets(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_init_neighbor_offsets += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        cache_neighbor_offsets(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_cache_neighbor_offsets += elapsed_seconds(t0, t1);
    } else {
        init_neighbor_offsets(env);
        cache_neighbor_offsets(env);
    }
    env->logs_capacity = 0;
    if (startup_timing_enabled) {
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        set_active_agents(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_set_active_agents += elapsed_seconds(t0, t1);
    } else {
        set_active_agents(env);
    }
    env->logs_capacity = env->active_agent_count;
    if (startup_timing_enabled) {
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        remove_bad_trajectories(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_remove_bad_trajectories += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        set_start_position(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_set_start_position += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        env->invalid_initial_trailer_state = has_invalid_initial_sdc_trailer_collision(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_invalid_initial_trailer_state += elapsed_seconds(t0, t1);
    } else {
        remove_bad_trajectories(env);
        set_start_position(env);
        env->invalid_initial_trailer_state = has_invalid_initial_sdc_trailer_collision(env);
    }
    if (env->invalid_initial_trailer_state) {
        raise_error_with_message(ERROR_INITIALIZATION_FAILED,
                                 "Invalid initial state: SDC trailer collision/off-road at scenario start");
    }
    if (startup_timing_enabled) {
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        init_goal_positions(env);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_init_goal_positions += elapsed_seconds(t0, t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        startup_timing_alloc_logs += elapsed_seconds(t0, t1);
    } else {
        init_goal_positions(env);
        env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
    }
}

void c_close(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->neighbor_offsets);

    for (int i = 0; i < grid_cell_count; i++) {
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->tracks_to_predict_indices);
    free(env->ini_file);
}

void allocate(Drive *env) {
    init(env);
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed)
        return maxSpeed;
    if (speed < -maxSpeed)
        return -maxSpeed;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI)
        heading -= 2 * M_PI;
    if (heading < -M_PI)
        heading += 2 * M_PI;
    return heading;
}

float normalize_value(float value, float min, float max) { return (value - min) / (max - min); }

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int use_articulated_ego =
        (env->dynamics_model == ARTICULATED && has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index);
    if (agent->removed)
        return;

    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        return;
    }

    if (env->dynamics_model == CLASSIC || use_articulated_ego || env->dynamics_model == ARTICULATED) {
        // Classic dynamics model
        const float *acceleration_values = classic_acceleration_values(env);
        int acceleration_count = classic_acceleration_count(env);
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= acceleration_values[acceleration_count - 1];
            steering *= STEERING_VALUES[12];
        } else { // discrete
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *)env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = acceleration_values[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        // Calculate current speed (signed based on direction relative to heading)
        float speed_magnitude = sqrtf(vx * vx + vy * vy);
        float v_dot_heading = vx * agent->heading_x + vy * agent->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Update speed with acceleration
        signed_speed = signed_speed + acceleration * env->dt;
        signed_speed = clipSpeed(signed_speed);
        // Compute yaw rate
        float beta = tanh(.5 * tanf(steering));

        // New heading
        float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / agent->length;

        // New velocity
        float new_vx = signed_speed * cosf(heading + beta);
        float new_vy = signed_speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx * env->dt);
        y = y + (new_vy * env->dt);
        heading = heading + yaw_rate * env->dt;

        heading = normalize_heading(heading);

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
        if (use_articulated_ego) {
            Entity *trailer = &env->entities[env->ego_trailer_track_index];
            float tractor2hitch = env->non_kinematic_vehicle_params[6];
            float trailer2hitch = env->non_kinematic_vehicle_params[7];
            float hitch_offset = -agent->length * 0.5f + tractor2hitch;
            float effective_length = fmaxf(0.5f, trailer->length - trailer2hitch);
            float theta_trailer = normalize_heading((heading - yaw_rate * env->dt) - agent->articulation_angle);
            float x_hitch_dot = new_vx - yaw_rate * hitch_offset * sinf(heading);
            float y_hitch_dot = new_vy + yaw_rate * hitch_offset * cosf(heading);
            float theta_trailer_dot =
                (-sinf(theta_trailer) * x_hitch_dot + cosf(theta_trailer) * y_hitch_dot) / effective_length;
            theta_trailer = normalize_heading(theta_trailer + theta_trailer_dot * env->dt);
            agent->articulation_angle = normalize_heading(agent->heading - theta_trailer);
            trailer->articulation_angle = agent->articulation_angle;
            project_articulated_ego_trailer_pose(env, 1);
        }
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action_array_f[action_idx][0]; // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]); // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3]; // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else { // discrete
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *)env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int a_long_idx = action_val / num_lat;
            int a_lat_idx = action_val % num_lat;
            a_long = JERK_LONG[a_long_idx];
            a_lat = JERK_LAT[a_lat_idx];
        }

        // Calculate new acceleration
        float a_long_new = agent->a_long + a_long * env->dt;
        float a_lat_new = agent->a_lat + a_lat * env->dt;

        // Make it easy to stop with 0 accel
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, -5.0f, 2.5f);
        }

        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
        }

        // Calculate new velocity
        float v_dot_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float signed_v = copysignf(sqrtf(agent->vx * agent->vx + agent->vy * agent->vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;

        // Make it easy to stop with 0 vel
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_new, -2.0f, 20.0f);
        }

        // Calculate new steering angle
        float signed_curvature = a_lat_new / fmaxf(v_new * v_new, 1e-5f);
        signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);
        float steering_angle = atanf(signed_curvature * agent->wheelbase);
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Update curvature and accel to account for limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * agent->heading_x - dy_local * agent->heading_y;
        float dy = dx_local * agent->heading_y + dy_local * agent->heading_x;

        // Update everything
        agent->x += dx;
        agent->y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->heading = normalize_heading(agent->heading + theta);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = v_new * agent->heading_x;
        agent->vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    return;
}

static inline int get_track_id_or_placeholder(Drive *env, int agent_idx) {
    if (env->tracks_to_predict_indices == NULL || env->num_tracks_to_predict == 0) {
        return -1;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict_indices[k] == agent_idx) {
            return env->tracks_to_predict_indices[k];
        }
    }
    return -1;
}

static inline int map_entity_to_policy_type(Drive *env, int entity_idx) {
    Entity *entity = &env->entities[entity_idx];
    if (entity->type == VEHICLE) {
        if (entity->is_trailer) {
            return POLICY_TYPE_VEHICLE_MULTI;
        }
        if (has_valid_ego_trailer_pair(env) &&
            (entity_idx == env->sdc_track_index || entity_idx == env->ego_trailer_track_index)) {
            return POLICY_TYPE_VEHICLE_MULTI;
        }
        return POLICY_TYPE_VEHICLE_SINGLE;
    }
    if (entity->type == PEDESTRIAN) {
        return POLICY_TYPE_PEDESTRIAN;
    }
    if (entity->type == CYCLIST) {
        return POLICY_TYPE_CYCLIST;
    }
    return POLICY_TYPE_PADDED;
}

static inline void get_ego_trailer_obs_features_for_agent(Drive *env, int agent_idx, float *trailer_rel_x_out,
                                                          float *trailer_rel_y_out, float *trailer_rel_heading_x_out,
                                                          float *trailer_rel_heading_y_out) {
    *trailer_rel_x_out = 0.0f;
    *trailer_rel_y_out = 0.0f;
    *trailer_rel_heading_x_out = 0.0f;
    *trailer_rel_heading_y_out = 0.0f;
    if (!has_valid_ego_trailer_pair(env) || agent_idx != env->sdc_track_index) {
        return;
    }

    Entity *ego = &env->entities[agent_idx];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    float dx = trailer->x - ego->x;
    float dy = trailer->y - ego->y;
    float rel_x = dx * ego->heading_x + dy * ego->heading_y;
    float rel_y = -dx * ego->heading_y + dy * ego->heading_x;
    float rel_heading_x = trailer->heading_x * ego->heading_x + trailer->heading_y * ego->heading_y;
    float rel_heading_y = trailer->heading_y * ego->heading_x - trailer->heading_x * ego->heading_y;

    *trailer_rel_x_out = rel_x * 0.02f;
    *trailer_rel_y_out = rel_y * 0.02f;
    *trailer_rel_heading_x_out = rel_heading_x;
    *trailer_rel_heading_y_out = rel_heading_y;
}

void c_get_ego_trailer_obs_features(Drive *env, float *trailer_rel_x_out, float *trailer_rel_y_out,
                                    float *trailer_rel_heading_x_out, float *trailer_rel_heading_y_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        get_ego_trailer_obs_features_for_agent(env, agent_idx, &trailer_rel_x_out[i], &trailer_rel_y_out[i],
                                               &trailer_rel_heading_x_out[i], &trailer_rel_heading_y_out[i]);
    }
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->x + env->world_mean_x;
        y_out[i] = agent->y + env->world_mean_y;
        z_out[i] = agent->z;
        heading_out[i] = agent->heading;
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        length_out[i] = agent->length;
        width_out[i] = agent->width;
    }
}

void c_get_global_agent_types(Drive *env, int *type_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        type_out[i] = map_entity_to_policy_type(env, agent_idx);
    }
}

void c_get_partner_types(Drive *env, int *type_out) {
    memset(type_out, 0, env->active_agent_count * (MAX_AGENTS - 1) * sizeof(int));
    int (*types)[MAX_AGENTS - 1] = (int (*)[MAX_AGENTS - 1])type_out;
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3) {
            continue;
        }

        int partner_idx = 0;
        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1) {
                continue;
            }
            if (env->entities[index].type > 3) {
                break;
            }
            if (index == env->active_agent_indices[i]) {
                continue;
            }
            if (has_valid_ego_trailer_pair(env) && env->active_agent_indices[i] == env->sdc_track_index &&
                index == env->ego_trailer_track_index) {
                continue;
            }
            Entity *other_entity = &env->entities[index];
            if (ego_entity->respawn_timestep != -1 || other_entity->respawn_timestep != -1) {
                continue;
            }
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx * dx + dy * dy);
            if (dist > 2500.0f) {
                continue;
            }
            if (partner_idx >= (MAX_AGENTS - 1)) {
                break;
            }
            types[i][partner_idx] = map_entity_to_policy_type(env, index);
            partner_idx++;
        }
    }
}

void c_get_active_agent_info(Drive *env, int *scenario_id_out, int *id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        scenario_id_out[i] = agent->scenario_id;
        id_out[i] = agent->id;
    }
}

void c_get_sdc_trailer_state(Drive *env, int *has_trailer_out, float *x_out, float *y_out, float *z_out,
                             float *heading_out, int *id_out, float *length_out, float *width_out) {
    if (!has_valid_ego_trailer_pair(env)) {
        *has_trailer_out = 0;
        *x_out = NAN;
        *y_out = NAN;
        *z_out = NAN;
        *heading_out = NAN;
        *id_out = -1;
        *length_out = 0.0f;
        *width_out = 0.0f;
        return;
    }

    Entity *trailer = &env->entities[env->ego_trailer_track_index];
    *has_trailer_out = 1;
    *x_out = trailer->x + env->world_mean_x;
    *y_out = trailer->y + env->world_mean_y;
    *z_out = trailer->z;
    *heading_out = trailer->heading;
    *id_out = trailer->id;
    *length_out = trailer->length;
    *width_out = trailer->width;
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, int *is_vehicle_out, int *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        is_vehicle_out[i] = agent->type == VEHICLE;
        scenario_id_out[i] = agent->scenario_id;

        for (int t = env->init_steps; t < agent->array_size; t++) {
            int out_idx = i * (agent->array_size - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->traj_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->traj_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->traj_z[t];
            heading_out[out_idx] = agent->traj_heading[t];
            valid_out[out_idx] = agent->traj_valid[t];
        }
    }
}

void c_copy_observations(Drive *env, float *obs_out) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int obs_dim = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    memcpy(obs_out, env->observations, env->active_agent_count * obs_dim * sizeof(float));
}

static inline void set_entity_state_to_logged_timestep(Entity *entity, int timestep) {
    if (entity->type > CYCLIST || entity->type == NONE) {
        return;
    }
    if (timestep < 0 || timestep >= entity->array_size || entity->traj_valid[timestep] == 0) {
        entity->x = INVALID_POSITION;
        entity->y = INVALID_POSITION;
        entity->z = 0.0f;
        entity->vx = 0.0f;
        entity->vy = 0.0f;
        entity->vz = 0.0f;
        entity->heading = 0.0f;
        entity->heading_x = 1.0f;
        entity->heading_y = 0.0f;
        entity->valid = 0;
    } else {
        entity->x = entity->traj_x[timestep];
        entity->y = entity->traj_y[timestep];
        entity->z = entity->traj_z[timestep];
        entity->vx = entity->traj_vx[timestep];
        entity->vy = entity->traj_vy[timestep];
        entity->vz = entity->traj_vz[timestep];
        entity->heading = entity->traj_heading[timestep];
        entity->heading_x = cosf(entity->heading);
        entity->heading_y = sinf(entity->heading);
        entity->valid = entity->traj_valid[timestep];
    }
    entity->collision_state = 0;
    entity->combo_collision_any = 0;
    entity->combo_collision_tractor_body = 0;
    entity->combo_collision_trailer_body = 0;
    entity->metrics_array[COLLISION_IDX] = 0.0f;
    entity->metrics_array[OFFROAD_IDX] = 0.0f;
    entity->metrics_array[REACHED_GOAL_IDX] = 0.0f;
    entity->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
    entity->respawn_timestep = -1;
    entity->respawn_count = 0;
    entity->collided_before_goal = 0;
    entity->current_goal_reached = 0;
    entity->stopped = 0;
    entity->removed = 0;
    entity->a_long = 0.0f;
    entity->a_lat = 0.0f;
    entity->jerk_long = 0.0f;
    entity->jerk_lat = 0.0f;
    entity->steering_angle = 0.0f;
}

void c_set_logged_timestep(Drive *env, int timestep) {
    if (timestep < env->init_steps) {
        timestep = env->init_steps;
    }
    if (timestep >= env->episode_length) {
        timestep = env->episode_length - 1;
    }

    env->timestep = timestep;

    for (int i = 0; i < env->num_objects; i++) {
        set_entity_state_to_logged_timestep(&env->entities[i], timestep);
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        compute_agent_metrics(env, agent_idx);
    }

    compute_observations(env);
}

typedef struct BCState {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    float length;
    float width;
    float a_long;
    float a_lat;
    float steering_angle;
    int stopped;
    int removed;
} BCState;

static inline BCState bc_state_from_entity(const Entity *entity) {
    BCState state = {
        .x = entity->x,
        .y = entity->y,
        .z = entity->z,
        .vx = entity->vx,
        .vy = entity->vy,
        .vz = entity->vz,
        .heading = entity->heading,
        .heading_x = entity->heading_x,
        .heading_y = entity->heading_y,
        .length = entity->length,
        .width = entity->width,
        .a_long = entity->a_long,
        .a_lat = entity->a_lat,
        .steering_angle = entity->steering_angle,
        .stopped = entity->stopped,
        .removed = entity->removed,
    };
    return state;
}

static inline void bc_simulate_discrete_classic(const Drive *env, const BCState *src, int action_val, BCState *dst) {
    *dst = *src;
    if (dst->removed) {
        return;
    }
    if (dst->stopped) {
        dst->vx = 0.0f;
        dst->vy = 0.0f;
        return;
    }

    const float *acceleration_values = classic_acceleration_values(env);
    int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
    int acceleration_index = action_val / num_steer;
    int steering_index = action_val % num_steer;
    float acceleration = acceleration_values[acceleration_index];
    float steering = STEERING_VALUES[steering_index];

    float speed_magnitude = sqrtf(dst->vx * dst->vx + dst->vy * dst->vy);
    float v_dot_heading = dst->vx * dst->heading_x + dst->vy * dst->heading_y;
    float signed_speed = copysignf(speed_magnitude, v_dot_heading);
    signed_speed = clipSpeed(signed_speed + acceleration * env->dt);

    float beta = tanh(.5f * tanf(steering));
    float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / fmaxf(dst->length, 1e-4f);
    float new_vx = signed_speed * cosf(dst->heading + beta);
    float new_vy = signed_speed * sinf(dst->heading + beta);

    dst->x = dst->x + (new_vx * env->dt);
    dst->y = dst->y + (new_vy * env->dt);
    dst->heading = dst->heading + yaw_rate * env->dt;
    dst->heading_x = cosf(dst->heading);
    dst->heading_y = sinf(dst->heading);
    dst->vx = new_vx;
    dst->vy = new_vy;
    dst->steering_angle = steering;
}

static inline float bc_signed_speed(const BCState *state) {
    float speed_magnitude = sqrtf(state->vx * state->vx + state->vy * state->vy);
    float v_dot_heading = state->vx * state->heading_x + state->vy * state->heading_y;
    return copysignf(speed_magnitude, v_dot_heading);
}

static inline int bc_reached_goal(const Drive *env, const Entity *agent, const BCState *state) {
    float distance_to_goal =
        relative_distance_2d(state->x, state->y, agent->goal_position_x, agent->goal_position_y);
    float current_speed = sqrtf(state->vx * state->vx + state->vy * state->vy);
    return (distance_to_goal < env->goal_radius) && (current_speed <= env->goal_speed);
}

static inline float bc_compute_step_cost(const BCState *pred, const Entity *gt, float w_lat, float w_lon,
                                         float w_heading, float w_speed, float *lat_cost_out, float *lon_cost_out) {
    float dx = pred->x - gt->x;
    float dy = pred->y - gt->y;
    float cos_h = cosf(gt->heading);
    float sin_h = sinf(gt->heading);
    float lon = dx * cos_h + dy * sin_h;
    float lat = -dx * sin_h + dy * cos_h;

    float lat_cost = w_lat * lat * lat;
    float lon_cost = w_lon * lon * lon;
    if (lat_cost_out != NULL) {
        *lat_cost_out = lat_cost;
    }
    if (lon_cost_out != NULL) {
        *lon_cost_out = lon_cost;
    }

    float cost = lat_cost + lon_cost;
    if (w_heading > 0.0f) {
        float d_heading = pred->heading - gt->heading;
        if (d_heading > M_PI)
            d_heading -= 2.0f * M_PI;
        if (d_heading < -M_PI)
            d_heading += 2.0f * M_PI;
        cost += w_heading * d_heading * d_heading;
    }
    if (w_speed > 0.0f) {
        float gt_speed_magnitude = sqrtf(gt->vx * gt->vx + gt->vy * gt->vy);
        float gt_v_dot_heading = gt->vx * cos_h + gt->vy * sin_h;
        float gt_signed_speed = copysignf(gt_speed_magnitude, gt_v_dot_heading);
        float d_speed = bc_signed_speed(pred) - gt_signed_speed;
        cost += w_speed * d_speed * d_speed;
    }
    return cost;
}

static inline float bc_compute_action_transition_cost(const Drive *env, int prev_action, int action, float w_steer_change,
                                                      float w_accel_change, float w_steer_flip) {
    if ((w_steer_change <= 0.0f && w_accel_change <= 0.0f && w_steer_flip <= 0.0f) || prev_action < 0 ||
        action < 0) {
        return 0.0f;
    }

    int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
    const float *acceleration_values = classic_acceleration_values(env);
    int prev_acceleration_index = prev_action / num_steer;
    int prev_steering_index = prev_action % num_steer;
    int acceleration_index = action / num_steer;
    int steering_index = action % num_steer;

    float transition_cost = 0.0f;
    if (w_steer_change > 0.0f) {
        float d_steer = STEERING_VALUES[steering_index] - STEERING_VALUES[prev_steering_index];
        transition_cost += w_steer_change * d_steer * d_steer;
    }
    if (w_accel_change > 0.0f) {
        float d_accel = acceleration_values[acceleration_index] - acceleration_values[prev_acceleration_index];
        transition_cost += w_accel_change * d_accel * d_accel;
    }
    if (w_steer_flip > 0.0f) {
        float prev_steer = STEERING_VALUES[prev_steering_index];
        float steer = STEERING_VALUES[steering_index];
        if (fabsf(prev_steer) > 1e-6f && fabsf(steer) > 1e-6f && (prev_steer * steer) < 0.0f) {
            transition_cost += w_steer_flip;
        }
    }
    return transition_cost;
}

static inline float bc_compute_progress_cost(const BCState *prev, const BCState *pred, const Entity *gt_prev,
                                             const Entity *gt_next, float w_progress, float w_reverse) {
    float cost = 0.0f;
    float gt_dx = gt_next->x - gt_prev->x;
    float gt_dy = gt_next->y - gt_prev->y;
    float gt_step_norm = sqrtf(gt_dx * gt_dx + gt_dy * gt_dy);
    if (w_progress > 0.0f && gt_step_norm > 1e-6f) {
        float dir_x = gt_dx / gt_step_norm;
        float dir_y = gt_dy / gt_step_norm;
        float pred_dx = pred->x - prev->x;
        float pred_dy = pred->y - prev->y;
        float pred_progress = pred_dx * dir_x + pred_dy * dir_y;
        float progress_gap = pred_progress - gt_step_norm;
        cost += w_progress * progress_gap * progress_gap;
    }
    if (w_reverse > 0.0f) {
        float signed_speed = bc_signed_speed(pred);
        if (signed_speed < 0.0f) {
            cost += w_reverse * signed_speed * signed_speed;
        }
    }
    return cost;
}

static inline float bc_compute_reference_control_cost(const Drive *env, const BCState *prev, int action,
                                                      const Entity *gt_prev, const Entity *gt_next,
                                                      float w_ref_accel, float w_ref_steer) {
    if (action < 0 || (w_ref_accel <= 0.0f && w_ref_steer <= 0.0f)) {
        return 0.0f;
    }

    int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
    const float *acceleration_values = classic_acceleration_values(env);
    int acceleration_index = action / num_steer;
    int steering_index = action % num_steer;
    float action_accel = acceleration_values[acceleration_index];
    float action_steer = STEERING_VALUES[steering_index];

    float cost = 0.0f;
    if (w_ref_accel > 0.0f) {
        float gt_prev_cos = cosf(gt_prev->heading);
        float gt_prev_sin = sinf(gt_prev->heading);
        float gt_next_cos = cosf(gt_next->heading);
        float gt_next_sin = sinf(gt_next->heading);
        float gt_prev_speed = copysignf(sqrtf(gt_prev->vx * gt_prev->vx + gt_prev->vy * gt_prev->vy),
                                        gt_prev->vx * gt_prev_cos + gt_prev->vy * gt_prev_sin);
        float gt_next_speed = copysignf(sqrtf(gt_next->vx * gt_next->vx + gt_next->vy * gt_next->vy),
                                        gt_next->vx * gt_next_cos + gt_next->vy * gt_next_sin);
        float ref_accel = (gt_next_speed - gt_prev_speed) / fmaxf(env->dt, 1e-5f);
        ref_accel = clip(ref_accel, acceleration_values[0], acceleration_values[classic_acceleration_count(env) - 1]);
        float d_accel = action_accel - ref_accel;
        cost += w_ref_accel * d_accel * d_accel;
    }

    if (w_ref_steer > 0.0f) {
        float gt_prev_cos = cosf(gt_prev->heading);
        float gt_prev_sin = sinf(gt_prev->heading);
        float gt_prev_speed = copysignf(sqrtf(gt_prev->vx * gt_prev->vx + gt_prev->vy * gt_prev->vy),
                                        gt_prev->vx * gt_prev_cos + gt_prev->vy * gt_prev_sin);
        float gt_next_cos = cosf(gt_next->heading);
        float gt_next_sin = sinf(gt_next->heading);
        float gt_next_speed = copysignf(sqrtf(gt_next->vx * gt_next->vx + gt_next->vy * gt_next->vy),
                                        gt_next->vx * gt_next_cos + gt_next->vy * gt_next_sin);
        float signed_speed = 0.5f * (gt_prev_speed + gt_next_speed);
        float d_heading = gt_next->heading - gt_prev->heading;
        if (d_heading > M_PI)
            d_heading -= 2.0f * M_PI;
        if (d_heading < -M_PI)
            d_heading += 2.0f * M_PI;
        float gt_step_dx = gt_next->x - gt_prev->x;
        float gt_step_dy = gt_next->y - gt_prev->y;
        float gt_step_norm = sqrtf(gt_step_dx * gt_step_dx + gt_step_dy * gt_step_dy);

        float ref_steer = 0.0f;
        if (fabsf(signed_speed) > 1e-4f) {
            float ref_yaw_rate = d_heading / fmaxf(env->dt, 1e-5f);
            float steering_angle = atanf((ref_yaw_rate * fmaxf(prev->length, 1e-4f)) / signed_speed);
            ref_steer = clip(steering_angle, STEERING_VALUES[0], STEERING_VALUES[num_steer - 1]);
        }
        float d_steer = action_steer - ref_steer;
        cost += w_ref_steer * d_steer * d_steer;
        // For nearly straight GT segments, strongly prefer near-zero steering to suppress left/right chatter.
        if (gt_step_norm > 1e-3f && fabsf(d_heading) < 0.02f && fabsf(ref_steer) < 0.08f) {
            cost += (4.0f * w_ref_steer) * action_steer * action_steer;
        }
    }
    return cost;
}

typedef struct BCBeamNode {
    BCState state;
    float cumulative_cost;
    float step_cost;
    float cumulative_lat_cost;
    float cumulative_lon_cost;
    float step_lat_cost;
    float step_lon_cost;
    int parent;
    int action;
    int valid;
    int terminal;
    int depth;
} BCBeamNode;

static inline int compare_bc_beam_nodes(const void *a, const void *b) {
    const BCBeamNode *lhs = (const BCBeamNode *)a;
    const BCBeamNode *rhs = (const BCBeamNode *)b;
    if (lhs->cumulative_cost < rhs->cumulative_cost)
        return -1;
    if (lhs->cumulative_cost > rhs->cumulative_cost)
        return 1;
    return 0;
}

int c_fit_discrete_action_sequence(Drive *env, int agent_slot, int beam_width, int planning_horizon, float w_lat,
                                   float w_lon, float w_heading, float w_speed, float w_steer_change,
                                   float w_accel_change, float w_reverse, float w_progress, float w_steer_flip,
                                   float w_ref_accel, float w_ref_steer, int *actions_out, float *step_costs_out,
                                   float *step_lat_costs_out, float *step_lon_costs_out, float *total_cost_out,
                                   float *total_lat_cost_out, float *total_lon_cost_out) {
    if (env->action_type != 0 || env->dynamics_model != CLASSIC) {
        return -1;
    }
    if (agent_slot < 0 || agent_slot >= env->active_agent_count) {
        return -2;
    }

    int agent_idx = env->active_agent_indices[agent_slot];
    Entity *agent = &env->entities[agent_idx];
    int start_t = env->init_steps;
    int end_t = agent->array_size - 1;
    if (planning_horizon > 0) {
        end_t = start_t + planning_horizon;
        if (end_t > agent->array_size - 1) {
            end_t = agent->array_size - 1;
        }
    }
    while (end_t > start_t && agent->traj_valid[end_t] == 0) {
        end_t--;
    }
    int num_steps = end_t - start_t;
    if (num_steps <= 0) {
        return 0;
    }

    int num_actions = classic_joint_action_count(env);
    int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
    int max_beam = beam_width > 0 ? beam_width : 1;

    BCBeamNode *nodes = (BCBeamNode *)calloc((num_steps + 1) * max_beam, sizeof(BCBeamNode));
    BCBeamNode *candidates = (BCBeamNode *)calloc(max_beam * num_actions, sizeof(BCBeamNode));
    if (nodes == NULL || candidates == NULL) {
        free(nodes);
        free(candidates);
        return -3;
    }

    nodes[0].state = bc_state_from_entity(agent);
    nodes[0].cumulative_cost = 0.0f;
    nodes[0].step_cost = 0.0f;
    nodes[0].cumulative_lat_cost = 0.0f;
    nodes[0].cumulative_lon_cost = 0.0f;
    nodes[0].step_lat_cost = 0.0f;
    nodes[0].step_lon_cost = 0.0f;
    nodes[0].parent = -1;
    nodes[0].action = -1;
    nodes[0].valid = 1;
    nodes[0].terminal = 0;
    nodes[0].depth = 0;

    int prev_count = 1;
    int final_step_count = num_steps;
    int final_best_idx = -1;
    for (int step = 0; step < num_steps; step++) {
        int candidate_count = 0;
        Entity gt_step = {0};
        Entity gt_prev = {0};
        gt_prev.x = agent->traj_x[start_t + step];
        gt_prev.y = agent->traj_y[start_t + step];
        gt_prev.heading = agent->traj_heading[start_t + step];
        gt_prev.vx = agent->traj_vx[start_t + step];
        gt_prev.vy = agent->traj_vy[start_t + step];
        gt_step.x = agent->traj_x[start_t + step + 1];
        gt_step.y = agent->traj_y[start_t + step + 1];
        gt_step.heading = agent->traj_heading[start_t + step + 1];
        gt_step.vx = agent->traj_vx[start_t + step + 1];
        gt_step.vy = agent->traj_vy[start_t + step + 1];
        gt_step.heading_x = cosf(gt_step.heading);
        gt_step.heading_y = sinf(gt_step.heading);
        float gt_step_heading_delta = gt_step.heading - gt_prev.heading;
        if (gt_step_heading_delta > M_PI)
            gt_step_heading_delta -= 2.0f * M_PI;
        if (gt_step_heading_delta < -M_PI)
            gt_step_heading_delta += 2.0f * M_PI;
        int lookahead_t = start_t + step + 4;
        if (lookahead_t > end_t) {
            lookahead_t = end_t;
        }
        float gt_window_heading_delta = agent->traj_heading[lookahead_t] - gt_prev.heading;
        if (gt_window_heading_delta > M_PI)
            gt_window_heading_delta -= 2.0f * M_PI;
        if (gt_window_heading_delta < -M_PI)
            gt_window_heading_delta += 2.0f * M_PI;

        for (int prev_idx = 0; prev_idx < prev_count; prev_idx++) {
            BCBeamNode *prev = &nodes[step * max_beam + prev_idx];
            if (!prev->valid) {
                continue;
            }
            float prev_dx = prev->state.x - gt_prev.x;
            float prev_dy = prev->state.y - gt_prev.y;
            float prev_cos_h = cosf(gt_prev.heading);
            float prev_sin_h = sinf(gt_prev.heading);
            float prev_lat_error = -prev_dx * prev_sin_h + prev_dy * prev_cos_h;
            float prev_lat_abs = fabsf(prev_lat_error);
            if (prev->terminal) {
                BCBeamNode *cand = &candidates[candidate_count++];
                *cand = *prev;
                cand->parent = prev_idx;
                cand->action = -1;
                cand->step_cost = 0.0f;
                cand->step_lat_cost = 0.0f;
                cand->step_lon_cost = 0.0f;
                cand->depth = prev->depth;
                continue;
            }
            for (int action = 0; action < num_actions; action++) {
                int steering_index = action % num_steer;
                float steering = STEERING_VALUES[steering_index];
                int allow_large_correction = prev_lat_abs > 0.15f;
                if (!allow_large_correction && fabsf(gt_window_heading_delta) < 0.04f && fabsf(steering) > 0.167f) {
                    continue;
                }
                if (!allow_large_correction && fabsf(gt_window_heading_delta) < 0.08f && fabsf(steering) > 0.333f) {
                    continue;
                }
                if (!allow_large_correction && fabsf(gt_step_heading_delta) < 0.01f && fabsf(steering) > 0.333f) {
                    continue;
                }
                BCBeamNode *cand = &candidates[candidate_count++];
                cand->state = prev->state;
                bc_simulate_discrete_classic(env, &prev->state, action, &cand->state);
                cand->step_cost = bc_compute_step_cost(&cand->state, &gt_step, w_lat, w_lon, w_heading, w_speed,
                                                       &cand->step_lat_cost, &cand->step_lon_cost);
                cand->step_cost += bc_compute_action_transition_cost(env, prev->action, action, w_steer_change,
                                                                     w_accel_change, w_steer_flip);
                cand->step_cost +=
                    bc_compute_progress_cost(&prev->state, &cand->state, &gt_prev, &gt_step, w_progress, w_reverse);
                cand->step_cost += bc_compute_reference_control_cost(
                    env, &prev->state, action, &gt_prev, &gt_step, w_ref_accel, w_ref_steer);
                cand->cumulative_cost = prev->cumulative_cost + cand->step_cost;
                cand->cumulative_lat_cost = prev->cumulative_lat_cost + cand->step_lat_cost;
                cand->cumulative_lon_cost = prev->cumulative_lon_cost + cand->step_lon_cost;
                cand->parent = prev_idx;
                cand->action = action;
                cand->valid = 1;
                cand->terminal = bc_reached_goal(env, agent, &cand->state);
                cand->depth = prev->depth + 1;
            }
        }

        qsort(candidates, candidate_count, sizeof(BCBeamNode), compare_bc_beam_nodes);
        int next_count = candidate_count < max_beam ? candidate_count : max_beam;
        for (int i = 0; i < next_count; i++) {
            nodes[(step + 1) * max_beam + i] = candidates[i];
        }
        for (int i = next_count; i < max_beam; i++) {
            nodes[(step + 1) * max_beam + i].valid = 0;
        }
        prev_count = next_count;

        for (int i = 0; i < prev_count; i++) {
            if (nodes[(step + 1) * max_beam + i].valid && nodes[(step + 1) * max_beam + i].terminal) {
                final_step_count = step + 1;
                final_best_idx = i;
                step = num_steps; // break outer loop
                break;
            }
        }
    }

    int best_idx = final_best_idx >= 0 ? final_best_idx : 0;
    float best_cost = nodes[final_step_count * max_beam + best_idx].cumulative_cost;
    for (int i = (final_best_idx >= 0 ? 0 : 1); i < prev_count; i++) {
        float cost = nodes[final_step_count * max_beam + i].cumulative_cost;
        if (cost < best_cost) {
            best_cost = cost;
            best_idx = i;
        }
    }

    int current_idx = best_idx;
    for (int step = final_step_count; step > 0; step--) {
        BCBeamNode *node = &nodes[step * max_beam + current_idx];
        if (node->action < 0) {
            current_idx = node->parent;
            continue;
        }
        actions_out[step - 1] = node->action;
        step_costs_out[step - 1] = node->step_cost;
        if (step_lat_costs_out != NULL) {
            step_lat_costs_out[step - 1] = node->step_lat_cost;
        }
        if (step_lon_costs_out != NULL) {
            step_lon_costs_out[step - 1] = node->step_lon_cost;
        }
        current_idx = node->parent;
        if (current_idx < 0 && step > 1) {
            free(nodes);
            free(candidates);
            return -4;
        }
    }

    *total_cost_out = best_cost;
    if (total_lat_cost_out != NULL) {
        *total_lat_cost_out = nodes[final_step_count * max_beam + best_idx].cumulative_lat_cost;
    }
    if (total_lon_cost_out != NULL) {
        *total_lon_cost_out = nodes[final_step_count * max_beam + best_idx].cumulative_lon_cost;
    }
    free(nodes);
    free(candidates);
    return final_step_count;
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_EDGE) {
            count++;
            points += env->entities[i].array_size;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, int *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type == ROAD_EDGE) {
            lengths_out[poly_idx] = e->array_size;
            scenario_ids_out[poly_idx] = e->scenario_id;
            for (int j = 0; j < e->array_size; j++) {
                x_out[pt_idx] = e->traj_x[j] + env->world_mean_x;
                y_out[pt_idx] = e->traj_y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

void compute_observations(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3)
            break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float speed_magnitude = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        float v_dot_heading = ego_entity->vx * ego_entity->heading_x + ego_entity->vy * ego_entity->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;

        obs[0] = rel_goal_x * 0.005f;
        obs[1] = rel_goal_y * 0.005f;
        obs[2] = signed_speed / MAX_SPEED;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        if (env->dynamics_model == JERK) {
            obs[6] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[7] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[8] = ego_entity->a_lat / JERK_LAT[2];
            obs[9] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        } else {
            obs[6] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        }

        // Relative Pos of other cars
        int obs_idx = ego_dim;
        int cars_seen = 0;
        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1)
                continue;
            if (env->entities[index].type > 3)
                break;
            if (index == env->active_agent_indices[i])
                continue; // Skip self, but don't increment obs_idx
            if (has_valid_ego_trailer_pair(env) && env->active_agent_indices[i] == env->sdc_track_index &&
                index == env->ego_trailer_track_index) {
                continue; // Keep ego policy unaware of its own dependent trailer.
            }
            Entity *other_entity = &env->entities[index];
            if (ego_entity->respawn_timestep != -1)
                continue;
            if (other_entity->respawn_timestep != -1)
                continue;
            // Store original relative positions
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx * dx + dy * dy);
            if (dist > 2500.0f)
                continue;
            // Rotate to ego vehicle's frame
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            // Store observations with correct indexing
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            // relative heading
            float rel_heading_x =
                other_entity->heading_x * ego_entity->heading_x +
                other_entity->heading_y * ego_entity->heading_y; // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y =
                other_entity->heading_y * ego_entity->heading_x -
                other_entity->heading_x * ego_entity->heading_y; // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;

            // relative speed
            float other_speed_magnitude =
                sqrtf(other_entity->vx * other_entity->vx + other_entity->vy * other_entity->vy);
            float other_v_dot_heading =
                other_entity->vx * other_entity->heading_x + other_entity->vy * other_entity->heading_y;
            float other_signed_speed = copysignf(other_speed_magnitude, other_v_dot_heading);
            obs[obs_idx + 6] = other_signed_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 7; // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * 7;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_entities) {
                printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_entities - 1);
                continue;
            }

            Entity *entity = &env->entities[entity_idx];

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= entity->array_size) {
                printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n", geometry_idx, entity_idx,
                       entity->array_size - 1);
                continue;
            }
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx * dx + dy * dy);
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;
            obs[obs_idx] = x_obs * 0.02f;
            obs[obs_idx + 1] = y_obs * 0.02f;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - 4.0f;
            obs_idx += 7;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * 7;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void sample_new_goal(Drive *env, int agent_idx) {
    // Samples a new goal position based on the existing road lane points
    Entity *agent = &env->entities[agent_idx];
    float best_x = agent->x;
    float best_y = agent->y;
    float best_distance_error = 1e30f;

    // Sample points from all road lanes
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type != ROAD_LANE)
            continue;

        Entity *lane = &env->entities[i];

        // Check every point in the lane
        for (int j = 0; j < lane->array_size; j++) {
            float point_x = lane->traj_x[j];
            float point_y = lane->traj_y[j];

            // Calculate vector from agent to point
            float to_point_x = point_x - agent->x;
            float to_point_y = point_y - agent->y;

            // Check if point is ahead of agent
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot <= 0.0f)
                continue;

            // Calculate distance to point
            float distance = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);

            // Find point closest to target distance
            float distance_error = fabsf(distance - env->goal_target_distance);
            if (distance_error < best_distance_error) {
                best_distance_error = distance_error;
                best_x = point_x;
                best_y = point_y;
            }
        }
    }

    // If no valid goal found, use another agent's initial goal
    if (best_distance_error >= 1e30f && env->active_agent_count > 1) {
        int other_idx = env->active_agent_indices[(agent_idx + 1) % env->active_agent_count];
        best_x = env->entities[other_idx].init_goal_x;
        best_y = env->entities[other_idx].init_goal_y;
    }

    agent->goal_position_x = best_x;
    agent->goal_position_y = best_y;
    agent->goals_sampled_this_episode += 1;
}

void c_reset(Drive *env) {
    env->timestep = env->init_steps;
    set_start_position(env);
    env->invalid_initial_trailer_state = has_invalid_initial_sdc_trailer_collision(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].respawn_count = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].goals_reached_this_episode = 0.0f;
        // Initialize to 1 because there is one goal in the data file
        env->entities[agent_idx].goals_sampled_this_episode = 1.0f;
        env->entities[agent_idx].current_goal_reached = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].combo_collision_any = 0;
        env->entities[agent_idx].combo_collision_tractor_body = 0;
        env->entities[agent_idx].combo_collision_trailer_body = 0;
        env->entities[agent_idx].stopped = 0;
        env->entities[agent_idx].removed = 0;

        if (env->goal_behavior == GOAL_GENERATE_NEW) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
        }

        compute_agent_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void respawn_agent(Drive *env, int agent_idx) {
    env->entities[agent_idx].x = env->entities[agent_idx].traj_x[0];
    env->entities[agent_idx].y = env->entities[agent_idx].traj_y[0];
    env->entities[agent_idx].heading = env->entities[agent_idx].traj_heading[0];
    env->entities[agent_idx].heading_x = cosf(env->entities[agent_idx].heading);
    env->entities[agent_idx].heading_y = sinf(env->entities[agent_idx].heading);
    env->entities[agent_idx].vx = env->entities[agent_idx].traj_vx[0];
    env->entities[agent_idx].vy = env->entities[agent_idx].traj_vy[0];
    env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
    env->entities[agent_idx].combo_collision_any = 0;
    env->entities[agent_idx].combo_collision_tractor_body = 0;
    env->entities[agent_idx].combo_collision_trailer_body = 0;

    env->entities[agent_idx].respawn_timestep = env->timestep;
    env->entities[agent_idx].collided_before_goal = 0;
    env->entities[agent_idx].stopped = 0;
    env->entities[agent_idx].removed = 0;
    env->entities[agent_idx].a_long = 0.0f;
    env->entities[agent_idx].a_lat = 0.0f;
    env->entities[agent_idx].jerk_long = 0.0f;
    env->entities[agent_idx].jerk_lat = 0.0f;
    env->entities[agent_idx].steering_angle = 0.0f;
    env->entities[agent_idx].articulation_angle = 0.0f;

    if (has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index) {
        if (env->dynamics_model == ARTICULATED) {
            sync_ego_trailer_entity_to_trajectory_step(env, 0);
            initialize_articulated_ego_trailer_state(env, env->force_zero_trailer_articulation_at_init);
        } else if (env->force_zero_trailer_articulation_at_init) {
            force_zero_trailer_articulation_pose_from_params(env);
        } else {
            update_ego_trailer_pose(env);
        }
    }
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    env->timestep++;

    int sdc_trailer_snapshot_valid = 0;
    float sdc_trailer_prev_x = 0.0f;
    float sdc_trailer_prev_y = 0.0f;
    float sdc_trailer_prev_heading = 0.0f;
    float sdc_trailer_prev_heading_x = 1.0f;
    float sdc_trailer_prev_heading_y = 0.0f;
    float sdc_trailer_prev_vx = 0.0f;
    float sdc_trailer_prev_vy = 0.0f;
    float sdc_tractor_prev_articulation = 0.0f;
    float sdc_trailer_prev_articulation = 0.0f;

    int originals_remaining = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        // Keep flag true if there is at least one agent that has not been respawned yet
        if (env->entities[agent_idx].respawn_count == 0) {
            originals_remaining = 1;
            break;
        }
    }

    if (env->timestep == env->episode_length || (!originals_remaining && env->termination_mode == 1)) {
        memset(env->terminals, 1, env->active_agent_count * sizeof(unsigned char));
        add_log(env);
        c_reset(env);
        return;
    }

    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->entities[expert_idx].x == INVALID_POSITION)
            continue;
        move_expert(env, env->actions, expert_idx);
    }
    // Process actions for all active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        float prev_vx = env->entities[agent_idx].vx;
        float prev_vy = env->entities[agent_idx].vy;

        if (!sdc_trailer_snapshot_valid && has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index) {
            Entity *trailer = &env->entities[env->ego_trailer_track_index];
            sdc_trailer_snapshot_valid = 1;
            sdc_trailer_prev_x = trailer->x;
            sdc_trailer_prev_y = trailer->y;
            sdc_trailer_prev_heading = trailer->heading;
            sdc_trailer_prev_heading_x = trailer->heading_x;
            sdc_trailer_prev_heading_y = trailer->heading_y;
            sdc_trailer_prev_vx = trailer->vx;
            sdc_trailer_prev_vy = trailer->vy;
            sdc_tractor_prev_articulation = env->entities[agent_idx].articulation_angle;
            sdc_trailer_prev_articulation = trailer->articulation_angle;
        }

        move_dynamics(env, i, agent_idx);
        if (env->dynamics_model != ARTICULATED && has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index) {
            update_ego_trailer_pose(env);
        }

        // Tiny jerk penalty for smoothness
        if (env->dynamics_model == CLASSIC || env->dynamics_model == ARTICULATED) {
            float delta_vx = env->entities[agent_idx].vx - prev_vx;
            float delta_vy = env->entities[agent_idx].vy - prev_vy;
            float jerk_penalty = -0.0002f * sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
            env->rewards[i] += jerk_penalty;
            env->logs[i].episode_return += jerk_penalty;
        }
    }

    // Compute rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;

        compute_agent_metrics(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        if (collision_state > 0) {
            if (collision_state == VEHICLE_COLLISION) {
                env->rewards[i] += env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].collisions_per_agent += 1.0f;
            } else if (collision_state == OFFROAD) {
                env->rewards[i] += env->reward_offroad_collision;
                env->logs[i].episode_return += env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].offroad_per_agent += 1.0f;
            }

            env->entities[agent_idx].collided_before_goal = 1;
        }

        float distance_to_goal =
            relative_distance_2d(env->entities[agent_idx].x, env->entities[agent_idx].y,
                                 env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        float current_speed = sqrtf(env->entities[agent_idx].vx * env->entities[agent_idx].vx +
                                    env->entities[agent_idx].vy * env->entities[agent_idx].vy);

        // Reward agent if it is within X meters of goal and speed is below threshold
        bool within_distance = distance_to_goal < env->goal_radius;
        bool within_speed = current_speed <= env->goal_speed;

        if (within_distance && within_speed && !env->entities[agent_idx].current_goal_reached) {
            if (env->goal_behavior == GOAL_RESPAWN && env->entities[agent_idx].respawn_timestep != -1) {
                env->rewards[i] += env->reward_goal_post_respawn;
                env->logs[i].episode_return += env->reward_goal_post_respawn;
                env->entities[agent_idx].current_goal_reached = 1;
            } else if (env->goal_behavior == GOAL_GENERATE_NEW && (!env->entities[agent_idx].current_goal_reached)) {
                env->rewards[i] += env->reward_goal;
                env->logs[i].episode_return += env->reward_goal;
                sample_new_goal(env, agent_idx);
                env->entities[agent_idx].current_goal_reached = 0;
                env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            } else { // Zero out the velocity so that the agent stops at the goal
                env->rewards[i] = env->reward_goal;
                env->logs[i].episode_return = env->reward_goal;
                env->entities[agent_idx].current_goal_reached = 1;
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
                if (agent_idx == env->sdc_track_index)
                    freeze_ego_trailer_entity(env);
                env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            }
            env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
            env->logs[i].speed_at_goal = current_speed;
        }

        if (sdc_trailer_snapshot_valid && has_valid_ego_trailer_pair(env) && agent_idx == env->sdc_track_index &&
            env->entities[agent_idx].stopped) {
            Entity *trailer = &env->entities[env->ego_trailer_track_index];
            trailer->x = sdc_trailer_prev_x;
            trailer->y = sdc_trailer_prev_y;
            trailer->heading = sdc_trailer_prev_heading;
            trailer->heading_x = sdc_trailer_prev_heading_x;
            trailer->heading_y = sdc_trailer_prev_heading_y;
            trailer->vx = 0.0f;
            trailer->vy = 0.0f;
            env->entities[agent_idx].articulation_angle = sdc_tractor_prev_articulation;
            trailer->articulation_angle = sdc_trailer_prev_articulation;
        }

        int lane_aligned = env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
        env->logs[i].lane_alignment_rate = lane_aligned;
    }

    if (env->goal_behavior == GOAL_RESPAWN) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                respawn_agent(env, agent_idx);
                env->entities[agent_idx].respawn_count++;
            }
        }
    } else if (env->goal_behavior == GOAL_STOP) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
                if (agent_idx == env->sdc_track_index)
                    freeze_ego_trailer_entity(env);
            }
        }
    }

    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS]; // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client *make_client(Drive *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferDrive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0, // Y is up
        1  // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,      // Same X as target
        120.0f, // 20 units above target
        175.0f  // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                         -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {client->camera.position.x - client->camera.target.x,
                             client->camera.position.y - client->camera.target.y,
                             client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
        return;
    }

    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    if (mode == 0) {
        DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1) {
        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, 1}, (Vector3){partner_x, partner_y, 1}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), 1},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), 1},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, 0.0}, (Vector3){arrow_x, arrow_y, 0.0}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, 1}, (Vector3){arrow_x_world, arrow_y_world, 1}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end1, arrow_y_end1, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end2, arrow_y_end2, 0.0}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 0.0}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1); // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.0f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

static inline void draw_ego_trailer_linkage(Drive *env) {
    if (!has_valid_ego_trailer_pair(env))
        return;

    Entity *tractor = &env->entities[env->sdc_track_index];
    Entity *trailer = &env->entities[env->ego_trailer_track_index];

    // Match visibility rules used for entities in draw_scene.
    if (tractor->respawn_timestep != -1 || trailer->respawn_timestep != -1)
        return;
    if (tractor->x == INVALID_POSITION || trailer->x == INVALID_POSITION)
        return;

    // Use the same extension geometry used by update_ego_trailer_pose.
    float tractor2hitch = env->non_kinematic_vehicle_params[6];
    float trailer2hitch = env->non_kinematic_vehicle_params[7];

    float theta_tractor = tractor->heading;
    float theta_trailer = trailer->heading;

    float tractor_hitch_x = tractor->x + (-tractor->length * 0.5f + tractor2hitch) * cosf(theta_tractor);
    float tractor_hitch_y = tractor->y + (-tractor->length * 0.5f + tractor2hitch) * sinf(theta_tractor);

    float trailer_hitch_x = trailer->x + (trailer->length * 0.5f - trailer2hitch) * cosf(theta_trailer);
    float trailer_hitch_y = trailer->y + (trailer->length * 0.5f - trailer2hitch) * sinf(theta_trailer);

    Vector3 tractor_hitch = {tractor_hitch_x, tractor_hitch_y, 1.2f};
    Vector3 trailer_hitch = {trailer_hitch_x, trailer_hitch_y, 1.2f};

    rlSetLineWidth(4.0f);
    DrawLine3D(tractor_hitch, trailer_hitch, ORANGE);
    DrawSphere(tractor_hitch, 0.18f, ORANGE);
    DrawSphere(trailer_hitch, 0.18f, ORANGE);
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

    if (show_grid) {
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 0.0f}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, Fade(PUFF_BACKGROUND2, 0.3f));
            }
        }
    }

    // Draw a grid to help with orientation
    for (int i = 0; i < env->num_entities; i++) {
        // Draw objects
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_agent = false;
            bool is_ego_trailer_entity = has_valid_ego_trailer_pair(env) && i == env->ego_trailer_track_index;
            bool is_ego_tractor_entity = has_valid_ego_trailer_pair(env) && i == env->sdc_track_index;
            bool is_ego_combo_entity = is_ego_trailer_entity || is_ego_tractor_entity;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if ((!is_active_agent && !is_static_agent && !is_ego_trailer_entity) ||
                env->entities[i].respawn_timestep != -1) {
                continue;
            }
            Vector3 position;
            float heading;
            position = (Vector3){env->entities[i].x, env->entities[i].y, 1.1};
            heading = env->entities[i].heading;
            // Create size vector
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);
            bool combo_collision_any = false;
            bool combo_body_triggered = false;
            if (is_ego_combo_entity && has_valid_ego_trailer_pair(env)) {
                Entity *tractor_entity = &env->entities[env->sdc_track_index];
                combo_collision_any = (tractor_entity->combo_collision_any > 0);
                combo_body_triggered =
                    is_ego_trailer_entity ? (tractor_entity->combo_collision_trailer_body > 0)
                                          : (tractor_entity->combo_collision_tractor_body > 0);
            } else {
                combo_collision_any = (env->entities[i].collision_state > 0);
                combo_body_triggered = combo_collision_any;
            }

            // Save current transform
            if (mode == 1) {
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                              position.y + (half_len * sin_heading + half_width * cos_heading), position.z},
                    (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                              position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                              position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                              position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

                };

                if (agent_index == env->human_agent_idx &&
                    !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                    continue;
                }

                // --- Draw the car  ---
                Color car_color = GRAY; // default for static
                if (is_expert)
                    car_color = GOLD; // expert replay
                if (is_active_agent)
                    car_color = BLUE; // policy-controlled
                if ((is_active_agent && env->entities[i].collision_state > 0) || combo_collision_any)
                    car_color = RED;
                if (combo_body_triggered) {
                    // Draw a bright halo first so the triggering body is unambiguous.
                    rlSetLineWidth(10.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], YELLOW);
                    }
                    DrawLine3D(corners[0], corners[2], RED);
                    DrawLine3D(corners[1], corners[3], RED);
                }
                rlSetLineWidth(combo_body_triggered ? 6.0f : 3.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j + 1) % 4], car_color);
                }
                // --- Draw a heading arrow pointing forward ---
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                                    position.y + sin_heading * half_len * 1.5f, position.z};

                DrawLine3D(arrowStart, arrowEnd, car_color);
                DrawSphere(arrowEnd, 0.2f, car_color); // arrow tip

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model (skip index 0)
                Model car_model = client->cars[(i % 5) + 1]; // Cycles through indices 1-5

                if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0]; // Ego agent always uses red car
                } else if (is_active_agent) {

                    car_model = client->cars[(i % 5) + 1];

                    if ((env->entities[i].collision_state > 0) || combo_collision_any) {
                        car_model = client->cars[0]; // Collided agents use red
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->entities[agent_index].metrics_array[REACHED_GOAL_IDX] ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
                // if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                //     rlPopMatrix();
                //     continue;
                // }
                if (env->entities[i].type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (env->entities[i].type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY; // static
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1)
                        wire_color = GOLD; // expert replay
                    if (is_active_agent)
                        wire_color = BLUE; // policy
                    if ((is_active_agent && env->entities[i].collision_state > 0) || combo_collision_any)
                        wire_color = RED;
                    if (combo_body_triggered) {
                        rlSetLineWidth(7.0f);
                        for (int j = 0; j < 4; j++) {
                            DrawLine3D(corners[j], corners[(j + 1) % 4], YELLOW);
                        }
                        DrawLine3D(corners[0], corners[2], RED);
                        DrawLine3D(corners[1], corners[3], RED);
                    }
                    rlSetLineWidth(combo_body_triggered ? 5.0f : 2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // FPV Camera Control
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                Vector3 camera_position = (Vector3){position.x - (25.0f * cosf(heading)),
                                                    position.y - (25.0f * sinf(heading)), position.z + 15};

                Vector3 camera_target = (Vector3){position.x + 40.0f * cosf(heading),
                                                  position.y + 40.0f * sinf(heading), position.z - 5.0f};
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents
            if (!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 1}, 0.5f,
                           DARKGREEN);

                DrawCircle3D((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 0.1f},
                             env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.9f));
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 && env->entities[i].type >= 7) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], 1};
            Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], 1};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = WHITE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = WHITE;
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;

            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                if (env->entities[i].type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y);
                } else if (env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    draw_ego_trailer_linkage(env);
    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            // Ignore respawned agents
            if (env->entities[i].respawn_timestep != -1) {
                continue;
            }
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict_indices[i];

            float raw_x = -env->entities[agent_idx].x * pixels_per_world_unit;
            float raw_y = env->entities[agent_idx].y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width / 2 + 20;
            int screen_y = (int)raw_y + client->height / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

void c_render(Drive *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0);

    if (IsKeyPressed(KEY_TAB)) {
        env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
    }

    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", client->camera.position.x, client->camera.position.y,
                        client->camera.position.z),
             10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", client->camera.target.x, client->camera.target.y,
                        client->camera.target.z),
             10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);

    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);

    // Display current action values - yellow when controlling, white otherwise
    Color action_color = IsKeyDown(KEY_LEFT_SHIFT) ? YELLOW : PUFF_WHITE;

    if (env->action_type == 0) { // discrete
        int *action_array = (int *)env->actions;
        int action_val = action_array[env->human_agent_idx];

        if (env->dynamics_model == CLASSIC || env->dynamics_model == ARTICULATED) {
            int num_steer = 13;
            const float *acceleration_values = classic_acceleration_values(env);
            int accel_idx = action_val / num_steer;
            int steer_idx = action_val % num_steer;
            float accel_value = acceleration_values[accel_idx];
            float steer_value = STEERING_VALUES[steer_idx];

            DrawText(TextFormat("Acceleration: %.2f m/s^2", accel_value), 10, 110, 20, action_color);
            DrawText(TextFormat("Steering: %.3f", steer_value), 10, 130, 20, action_color);
        } else if (env->dynamics_model == JERK) {
            int num_lat = 3;
            int jerk_long_idx = action_val / num_lat;
            int jerk_lat_idx = action_val % num_lat;
            float jerk_long_value = JERK_LONG[jerk_long_idx];
            float jerk_lat_value = JERK_LAT[jerk_lat_idx];

            DrawText(TextFormat("Longitudinal Jerk: %.2f m/s^3", jerk_long_value), 10, 110, 20, action_color);
            DrawText(TextFormat("Lateral Jerk: %.2f m/s^3", jerk_lat_value), 10, 130, 20, action_color);
        }
    } else { // continuous
        float (*action_array_f)[2] = (float (*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, action_color);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, action_color);
    }

    // Show key press status
    int status_y = 150;
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        DrawText("[shift pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_SPACE)) {
        DrawText("[space pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        DrawText("[ctrl pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }

    // Controls help
    DrawText("Controls: SHIFT + W/S - Accelerate/Brake, SHIFT + A/D - Steer, TAB - Switch Agent", 10,
             client->height - 30, 20, PUFF_WHITE);

    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, status_y, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, status_y + 20, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

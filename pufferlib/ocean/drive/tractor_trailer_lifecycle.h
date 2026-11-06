#ifndef PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_LIFECYCLE_H
#define PUFFERLIB_OCEAN_DRIVE_TRACTOR_TRAILER_LIFECYCLE_H

static inline void clear_articulated_ego(Drive *env) {
    env->articulated_ego_valid = 0;
    env->tractor_idx = -1;
    env->trailer_idx = -1;
    env->articulated_geometry = (ArticulatedGeometry){0};
    env->articulated_state = (ArticulatedVehicleState){0};
}

static inline void refresh_articulated_state_from_entities(Drive *env) {
    if (!env->articulated_ego_valid)
        return;

    Entity *tractor = &env->entities[env->tractor_idx];
    Entity *trailer = &env->entities[env->trailer_idx];
    ArticulatedPose tractor_center = {
        .x = tractor->x,
        .y = tractor->y,
        .z = tractor->z,
        .heading = tractor->heading,
    };
    ArticulatedPose trailer_center = {
        .x = trailer->x,
        .y = trailer->y,
        .z = trailer->z,
        .heading = trailer->heading,
    };

    env->articulated_state.tractor_rear_axle =
        entity_center_pose_to_axle(tractor_center, env->articulated_geometry.tractor_axle_to_center);
    env->articulated_state.trailer_axle =
        entity_center_pose_to_axle(trailer_center, env->articulated_geometry.trailer_axle_to_center);
    float longitudinal_velocity = tractor->vx * tractor->heading_x + tractor->vy * tractor->heading_y;
    env->articulated_state.speed = longitudinal_velocity;
    env->articulated_state.steering_angle = tractor->steering_angle;
}

static inline void write_articulated_state_to_entities(Drive *env) {
    if (!env->articulated_ego_valid)
        return;

    Entity *tractor = &env->entities[env->tractor_idx];
    Entity *trailer = &env->entities[env->trailer_idx];
    ArticulatedPose tractor_center = axle_pose_to_entity_center(
        env->articulated_state.tractor_rear_axle,
        env->articulated_geometry.tractor_axle_to_center);
    ArticulatedPose trailer_center = axle_pose_to_entity_center(
        env->articulated_state.trailer_axle,
        env->articulated_geometry.trailer_axle_to_center);
    ArticulatedEntityVelocities velocities = articulated_entity_velocities(
        &env->articulated_state, &env->articulated_geometry);

    tractor->x = tractor_center.x;
    tractor->y = tractor_center.y;
    tractor->z = tractor_center.z;
    tractor->heading = tractor_center.heading;
    tractor->heading_x = cosf(tractor->heading);
    tractor->heading_y = sinf(tractor->heading);
    tractor->vx = velocities.tractor_vx;
    tractor->vy = velocities.tractor_vy;
    tractor->vz = 0.0f;
    tractor->steering_angle = env->articulated_state.steering_angle;

    trailer->x = trailer_center.x;
    trailer->y = trailer_center.y;
    trailer->z = trailer_center.z;
    trailer->heading = trailer_center.heading;
    trailer->heading_x = cosf(trailer->heading);
    trailer->heading_y = sinf(trailer->heading);
    trailer->vx = velocities.trailer_vx;
    trailer->vy = velocities.trailer_vy;
    trailer->vz = 0.0f;
}

static inline int initialize_articulated_ego(Drive *env) {
    clear_articulated_ego(env);
    if (env->ego_vehicle_type != EGO_VEHICLE_TRACTOR_TRAILER)
        return 0;

    int tractor_idx = env->sdc_track_index;
    if (tractor_idx < 0 || tractor_idx >= env->num_objects)
        return 0;

    Entity *tractor = &env->entities[tractor_idx];
    if (tractor->type != VEHICLE || tractor->traj_valid == NULL || tractor->traj_heading == NULL ||
        tractor->array_size <= 0)
        return 0;

    int init_step = env->init_steps;
    if (init_step < 0)
        init_step = 0;
    if (init_step >= tractor->array_size)
        init_step = tractor->array_size - 1;
    if (!tractor->traj_valid[init_step])
        return 0;

    ArticulatedGeometry geometry = {0};
    if (articulated_geometry_from_params(DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS, &geometry) !=
        ARTICULATED_GEOMETRY_OK)
        return 0;

    Entity trailer = {0};
    trailer.articulated_parent_idx = tractor_idx;
    trailer.array_size = tractor->array_size;
    size_t trajectory_bytes = (size_t)trailer.array_size * sizeof(float);
    size_t validity_bytes = (size_t)trailer.array_size * sizeof(int);
    trailer.traj_x = (float *)malloc(trajectory_bytes);
    trailer.traj_y = (float *)malloc(trajectory_bytes);
    trailer.traj_z = (float *)malloc(trajectory_bytes);
    trailer.traj_vx = (float *)malloc(trajectory_bytes);
    trailer.traj_vy = (float *)malloc(trajectory_bytes);
    trailer.traj_vz = (float *)malloc(trajectory_bytes);
    trailer.traj_heading = (float *)malloc(trajectory_bytes);
    trailer.traj_valid = (int *)malloc(validity_bytes);
    if (trailer.traj_x == NULL || trailer.traj_y == NULL || trailer.traj_z == NULL || trailer.traj_vx == NULL ||
        trailer.traj_vy == NULL || trailer.traj_vz == NULL || trailer.traj_heading == NULL ||
        trailer.traj_valid == NULL) {
        free_entity(&trailer);
        return 0;
    }

    for (int t = 0; t < tractor->array_size; t++) {
        trailer.traj_valid[t] = tractor->traj_valid[t];
        trailer.traj_heading[t] = tractor->traj_heading[t];
        trailer.traj_vx[t] = tractor->traj_vx[t];
        trailer.traj_vy[t] = tractor->traj_vy[t];
        trailer.traj_vz[t] = tractor->traj_vz[t];
        if (!tractor->traj_valid[t] || tractor->traj_x[t] == INVALID_POSITION) {
            trailer.traj_x[t] = INVALID_POSITION;
            trailer.traj_y[t] = INVALID_POSITION;
            trailer.traj_z[t] = 0.0f;
            continue;
        }

        ArticulatedPose tractor_center = {
            .x = tractor->traj_x[t],
            .y = tractor->traj_y[t],
            .z = tractor->traj_z[t],
            .heading = tractor->traj_heading[t],
        };
        ArticulatedPose tractor_axle =
            entity_center_pose_to_axle(tractor_center, geometry.tractor_axle_to_center);
        ArticulatedPoint connection = tractor_connection_point(tractor_axle, &geometry);
        ArticulatedPose trailer_axle =
            trailer_axle_pose_from_connection(connection, tractor_axle.heading, &geometry);
        ArticulatedPose trailer_center =
            axle_pose_to_entity_center(trailer_axle, geometry.trailer_axle_to_center);
        trailer.traj_x[t] = trailer_center.x;
        trailer.traj_y[t] = trailer_center.y;
        trailer.traj_z[t] = trailer_center.z;
    }

    trailer.scenario_id = tractor->scenario_id;
    trailer.type = VEHICLE;
    trailer.id = INT_MIN + tractor_idx;
    trailer.width = geometry.trailer_width;
    trailer.length = geometry.trailer_length;
    trailer.height = geometry.trailer_height;
    trailer.goal_position_x = tractor->goal_position_x;
    trailer.goal_position_y = tractor->goal_position_y;
    trailer.goal_position_z = tractor->goal_position_z;
    trailer.articulated_role = ARTICULATED_ROLE_TRAILER;

    int trailer_idx = env->num_objects;
    Entity *resized_entities = (Entity *)realloc(env->entities, (env->num_entities + 1) * sizeof(Entity));
    if (resized_entities == NULL) {
        free_entity(&trailer);
        return 0;
    }
    env->entities = resized_entities;
    memmove(&env->entities[trailer_idx + 1], &env->entities[trailer_idx], env->num_roads * sizeof(Entity));
    env->entities[trailer_idx] = trailer;
    env->num_objects++;
    env->num_entities++;

    tractor = &env->entities[tractor_idx];
    tractor->width = geometry.tractor_width;
    tractor->length = geometry.tractor_length;
    tractor->height = geometry.tractor_height;
    tractor->wheelbase = geometry.wheelbase;
    tractor->articulated_role = ARTICULATED_ROLE_TRACTOR;
    tractor->articulated_parent_idx = tractor_idx;

    env->articulated_ego_valid = 1;
    env->tractor_idx = tractor_idx;
    env->trailer_idx = trailer_idx;
    env->articulated_geometry = geometry;
    return 1;
}

static inline void reset_attached_trailer_runtime(Drive *env, int step, int respawn_timestep) {
    if (!env->articulated_ego_valid)
        return;

    Entity *trailer = &env->entities[env->trailer_idx];
    if (step < 0)
        step = 0;
    if (step >= trailer->array_size)
        step = trailer->array_size - 1;

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
    trailer->collision_state = 0;
    memset(trailer->metrics_array, 0, sizeof(trailer->metrics_array));
    trailer->respawn_timestep = respawn_timestep;
    trailer->collided_before_goal = 0;
    trailer->stopped = 0;
    trailer->removed = 0;
    trailer->a_long = 0.0f;
    trailer->a_lat = 0.0f;
    trailer->jerk_long = 0.0f;
    trailer->jerk_lat = 0.0f;
    trailer->steering_angle = 0.0f;
}

#endif

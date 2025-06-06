/* School: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#define MAX_SPEED 0.02f
#define MAX_FACTORY_SPEED 0.002f

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Camera3D camera;
    Model ship;
} Client;

typedef struct {
  float w, x, y, z;
} Quat;

typedef struct {
  float x, y, z;
} Vec3;

typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float speed;
    float pitch;
    float roll;
    float health;
    float max_turn;
    float max_speed;
    float attack_damage;
    float attack_range;
    Quat orientation;
    float yaw;
    int item;
    int target;
    int episode_length;
    float episode_return;
} Entity;

typedef struct {
    Log log;
    Client* client;
    Entity* agents;
    Entity* factories;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    float size_x;
    float size_y;
    float size_z;
    int num_agents;
    int num_factories;
    int num_resources;
} School;

void init(School* env) {
    env->agents = calloc(env->num_agents, sizeof(Entity));
    env->factories = calloc(env->num_factories, sizeof(Entity));
}

void update_abilities(Entity* agent) {
    agent->health = 1.0f;
    agent->attack_damage = 0.4f;
    agent->attack_range = 0.2f;

    if (agent->item == 0) {
        agent->max_turn = 0.75f;
        agent->max_speed = 0.75f;
    } else if (agent->item == 1) {
        agent->max_turn = 1.0f;
        agent->max_speed = 0.5f;
    } else if (agent->item == 2) {
        agent->max_turn = 0.5f;
        agent->max_speed = 1.0f;
    } else if (agent->item == 3) {
        agent->max_turn = 1.5f;
        agent->max_speed = 0.25f;
    }
}

void respawn(School* env, Entity* agent) {
    //agent->x = randf(-env->size_x, env->size_x);
    //agent->y = randf(-env->size_y, env->size_y);
    //agent->z = randf(-env->size_z, env->size_z);
    if (agent->item == 0) {
        agent->x = -env->size_x;
        agent->y = 0.0f;
        agent->z = 0.0f;
    } else if (agent->item == 1) {
        agent->x = env->size_x;
        agent->y = 0.0f;
        agent->z = 0.0f;
    } else if (agent->item == 2) {
        agent->x = 0.0f;
        agent->y = 0.0f;
        agent->z = -env->size_z;
    } else if (agent->item == 3) {
        agent->x = 0.0f;
        agent->y = 0.0f;
        agent->z = env->size_z;
    }
}


bool attack(Entity *agent, Entity *target) {
    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dy*dy + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    // Unit vec to target
    dy /= dd;
    dz /= dd;
    dx /= dd;

    // Unit forward vec
    float mag = sqrtf(agent->vx*agent->vx + agent->vy*agent->vy + agent->vz*agent->vz);
    float fx = agent->vx / mag;
    float fy = agent->vy / mag;
    float fz = agent->vz / mag;

    // Angle to target
    float angle = acosf(dx*fx + dy*fy + dz*fz);
    if (angle < PI/6) {
        return true;
    }
    return false;
}

static inline float clampf(float v, float min, float max) {
  if (v < min)
    return min;
  if (v > max)
    return max;
  return v;
}

static inline float rndf(float a, float b) {
  return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline int rndi(int a, int b) { return a + rand() % (b - a + 1); }

static inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

// In-place clamp of a vector
static inline void clamp3(Vec3 *vec, float min, float max) {
  vec->x = clampf(vec->x, min, max);
  vec->y = clampf(vec->y, min, max);
  vec->z = clampf(vec->z, min, max);
}

// In-place clamp of a vector
static inline void clamp4(float a[4], float min, float max) {
  a[0] = clampf(a[0], min, max);
  a[1] = clampf(a[1], min, max);
  a[2] = clampf(a[2], min, max);
  a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
  Quat out;
  out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return out;
}

static inline void quat_normalize(Quat *q) {
  float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
  if (n > 0.0f) {
    q->w /= n;
    q->x /= n;
    q->y /= n;
    q->z /= n;
  }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
  Quat qv = {0.0f, v.x, v.y, v.z};
  Quat tmp = quat_mul(q, qv);
  Quat q_conj = {q.w, -q.x, -q.y, -q.z};
  Quat res = quat_mul(tmp, q_conj);
  return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float norm = norm3(axis);
    if (norm < 0.0001f) { // Handle zero axis
        return (Quat){1.0f, 0.0f, 0.0f, 0.0f}; // Identity quaternion
    }
    Vec3 norm_axis = {axis.x / norm, axis.y / norm, axis.z / norm};
    float s = sinf(angle / 2.0f);
    float c = cosf(angle / 2.0f);
    Quat q = {c, s * norm_axis.x, s * norm_axis.y, s * norm_axis.z};
    quat_normalize(&q);
    return q;
}

int compare_floats(const void* a, const void* b) {
    return (*(float*)a - *(float*)b) > 0;
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

float clip_angle(float theta) {
    if (theta < 0.0f) {
        return theta + 2.0f*PI;
    } else if (theta > 2.0f*PI) {
        return theta - 2.0f*PI;
    }
    return theta;
}

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

float randi(int min, int max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

void move_basic(School* env, Entity* agent, int* actions) {
    float d_vx = ((float)actions[0] - 4.0f)/400.0f;
    float d_vy = ((float)actions[1] - 4.0f)/400.0f;
    float d_vz = ((float)actions[2] - 4.0f)/400.0f;

    agent->vx += d_vx;
    agent->vy += d_vy;
    agent->vz += d_vz;

    agent->vx = clip(agent->vx, -MAX_SPEED, MAX_SPEED);
    agent->vy = clip(agent->vy, -MAX_SPEED, MAX_SPEED);
    agent->vz = clip(agent->vz, -MAX_SPEED, MAX_SPEED);

    agent->x += agent->vx;
    agent->y += agent->vy;
    agent->z += agent->vz;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->y = clip(agent->y, -env->size_y, env->size_y);
    agent->z = clip(agent->z, -env->size_z, env->size_z);
}

void move_ship(School* env, Entity* agent, int* actions) {
    // Compute deltas from actions (same as original)
    float d_pitch = agent->max_turn * ((float)actions[0] - 4.0f) / 20.0f;
    float d_roll = agent->max_turn * ((float)actions[1] - 4.0f) / 20.0f;
    //float d_yaw = agent->max_turn * ((float)actions[2] - 4.0f) / 20.0f;

    // Update speed and clamp
    agent->speed = agent->max_speed * MAX_SPEED;
    //agent->speed = clampf(agent->speed, 0.0f, agent->max_speed * MAX_SPEED); // Assuming MAX_SPEED is defined

    // Get local axes in world coordinates for pitch and roll
    Vec3 x_axis = quat_rotate(agent->orientation, (Vec3){1.0f, 0.0f, 0.0f}); // Pitch axis
    Vec3 z_axis = quat_rotate(agent->orientation, (Vec3){0.0f, 0.0f, 1.0f}); // Roll axis
    //Vec3 y_axis = quat_rotate(agent->orientation, (Vec3){0.0f, 1.0f, 0.0f}); // Yaw axis

    // Create rotation quaternions
    Quat q_pitch = quat_from_axis_angle(x_axis, d_pitch);
    Quat q_roll = quat_from_axis_angle(z_axis, d_roll);
    //Quat q_yaw = quat_from_axis_angle(y_axis, d_yaw);

    // Update orientation: pitch, then roll (no yaw in original)
    agent->orientation = quat_mul(agent->orientation, q_pitch);
    agent->orientation = quat_mul(agent->orientation, q_roll);
    //agent->orientation = quat_mul(agent->orientation, q_yaw);
    quat_normalize(&agent->orientation);

    // Optional: Limit pitch to ±90° to match clip_angle behavior
    Vec3 forward = quat_rotate(agent->orientation, (Vec3){0.0f, 0.0f, 1.0f});
    if (fabsf(forward.y) > 0.999f) { // Near vertical, clamp pitch
        // Reset to max pitch (≈ ±89° to avoid singularity)
        float sign = forward.y > 0 ? 1.0f : -1.0f;
        Quat max_pitch = quat_from_axis_angle((Vec3){1.0f, 0.0f, 0.0f}, sign * 1.5533f); // ≈ 89°
        agent->orientation = max_pitch;
        quat_normalize(&agent->orientation);
    }

    // Update position (move along local z-axis)
    forward = quat_rotate(agent->orientation, (Vec3){0.0f, 0.0f, 1.0f});
    agent->x += agent->speed * forward.x;
    agent->y += agent->speed * forward.y;
    agent->z += agent->speed * forward.z;

    // Just for visualization
    agent->vx = agent->speed * forward.x;
    agent->vy = agent->speed * forward.y;
    agent->vz = agent->speed * forward.z;

    // Clamp position to environment bounds
    agent->x = clampf(agent->x, -env->size_x, env->size_x);
    agent->y = clampf(agent->y, -env->size_y, env->size_y);
    agent->z = clampf(agent->z, -env->size_z, env->size_z);
}

void compute_observations(School* env) {
    int obs_idx = 0;
    for (int a=0; a<env->num_agents; a++) {
        Entity* agent = &env->agents[a];
        float dists[env->num_resources];
        for (int i=0; i<env->num_resources; i++) {
            dists[i] = 999999;
        }
        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            float dx = factory->x - agent->x;
            float dy = factory->y - agent->y;
            float dz = factory->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            int type = f % env->num_resources;
            if (dd < dists[type]) {
                dists[type] = dd;
                env->observations[obs_idx + 3*type] = dx;
                env->observations[obs_idx + 3*type + 1] = dy;
                env->observations[obs_idx + 3*type + 2] = dz;
            }
        }
        obs_idx += 3*env->num_resources;
        env->observations[obs_idx++] = agent->vx/MAX_SPEED;
        env->observations[obs_idx++] = agent->vy/MAX_SPEED;
        env->observations[obs_idx++] = agent->vz/MAX_SPEED;
        env->observations[obs_idx++] = agent->speed;
        env->observations[obs_idx++] = agent->pitch;
        env->observations[obs_idx++] = agent->roll;
        env->observations[obs_idx++] = agent->x;
        env->observations[obs_idx++] = agent->y;
        env->observations[obs_idx++] = agent->z;
        env->observations[obs_idx++] = env->rewards[a];

        float min_dist = 999999;
        int min_idx = 0;
        for (int j=0; j<env->num_agents; j++) {
            if (j == a) {
                continue;
            }
            Entity* agent = &env->agents[j];
            float dx = agent->x - agent->x;
            float dy = agent->y - agent->y;
            float dz = agent->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            if (dd < min_dist) {
                min_dist = dd;
                min_idx = j;
            }
        }
        Entity* other = &env->agents[min_idx];
        env->observations[obs_idx++] = agent->x - other->x;
        env->observations[obs_idx++] = agent->y - other->y;
        env->observations[obs_idx++] = agent->z - other->z;
        env->observations[obs_idx++] = (float)(agent->item == other->item);
        memset(&env->observations[obs_idx], 0, env->num_resources*sizeof(float));
        env->observations[obs_idx + agent->item] = 1.0f;
        obs_idx += env->num_resources;
    }
}

// Required function
void c_reset(School* env) {
    for (int i=0; i<env->num_agents; i++) {
        respawn(env, &env->agents[i]);
        env->agents[i].orientation = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        env->agents[i].item = rand() % env->num_resources;
        update_abilities(&env->agents[i]);
        env->agents[i].episode_length = 0;
        env->agents[i].target = -1;
        env->agents[i].attack_range = 0.2f;
        env->agents[i].attack_damage = 0.4f;
    }
    for (int i=0; i<env->num_factories; i++) {
        env->factories[i].x = randf(-env->size_x, env->size_x);
        env->factories[i].y = randf(-env->size_y, env->size_y);
        env->factories[i].z = randf(-env->size_z, env->size_z);
        env->factories[i].vx = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
        env->factories[i].vy = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
        env->factories[i].vz = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
        env->factories[i].item = i % env->num_resources;
    }
    compute_observations(env);
}

void c_step(School* env) {
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));

    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        agent->episode_length += 1;
        agent->target = -1;

        if (agent->health <= 0) {
            agent->health = 1.0f;
            respawn(env, agent);

            env->log.episode_length += agent->episode_length;
            env->log.episode_return += agent->episode_return;
            env->log.n++;
            agent->episode_length = 0;
            agent->episode_return = 0;
        }

        //move_basic(env, agent, env->actions + 3*i);
        move_ship(env, agent, env->actions + 3*i);

        if (rand() % env->num_agents == 0) {
            respawn(env, &env->agents[i]);
        }

        // Collision penalty
        /*
        float penalty = 0.0f;
        float same_color_dist = 999999;
        float diff_color_dist = 999999;
        for (int j=0; j<env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            Entity* other = &env->agents[j];
            float dx = other->x - agent->x;
            float dy = other->y - agent->y;
            float dz = other->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            if (other->item == agent->item && dd < same_color_dist) {
                same_color_dist = dd;
            } else if (other->item != agent->item && dd < diff_color_dist) {
                diff_color_dist = dd;
            }
        }
        if (agent->item == 0 && diff_color_dist < same_color_dist) {
            env->rewards[i] -= 0.5f;
        } else if (agent->item == 1 && diff_color_dist < same_color_dist) {
            env->rewards[i] += 0.5f;
        }
        */


        // Distance penalty
        /*
        float dist = (agent->x*agent->x + agent->y*agent->y + agent->z*agent->z);
        if (dist > 0.25 && dist < 0.35) {
            env->rewards[i] += 1.0f;
            agent->episode_return += env->rewards[i];
        }
        */

        //env->rewards[i] -= penalty;
        //agent->episode_return += env->rewards[i];
        /*
        if (agent->episode_length > 256) {
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->episode_length;
            env->log.episode_return += agent->episode_return;
            env->log.n++;
            //env->rewards[i] += 1.0f;
            agent->episode_length = 0;
            agent->episode_return = 0;
        }
        */

        // Target seeking reward
        /*
        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            float dx = (factory->x - agent->x);
            float dy = (factory->y - agent->y);
            float dz = (factory->z - agent->z);
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.1) {
                continue;
            }
            if (factory->item == agent->item) {
                agent->item = (agent->item + 1) % env->num_resources;
                update_abilities(agent);
                env->log.perf += 1.0f;
                env->log.score += 1.0f;
                env->log.episode_length += agent->episode_length;
                env->log.n++;
                env->rewards[i] += 1.0f;
                agent->episode_length = 0;
            }
        }
        */
    }
    for (int f=0; f<env->num_factories; f++) {
        Entity* factory = &env->factories[f];
        factory->x += factory->vx;
        factory->y += factory->vy;
        factory->z += factory->vz;

        float factory_x = clip(factory->x, -env->size_x, env->size_x);
        float factory_y = clip(factory->y, -env->size_y, env->size_y);
        float factory_z = clip(factory->z, -env->size_z, env->size_z);

        if (factory_x != factory->x || factory_y != factory->y || factory_z != factory->z) {
            factory->vx = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->vy = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->vz = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->x = factory_x;
            factory->y = factory_y;
            factory->z = factory_z;
        }
    }
    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        for (int j=0; j<env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            Entity* target = &env->agents[j];
            if (agent->item == target->item) {
                continue;
            }
            if (!attack(agent, target)) {
                continue;
            }
            agent->target = j;
            env->rewards[i] += 0.25f;
            agent->episode_return += 0.25f;
            //env->rewards[j] -= 0.25f;
            target->health -= agent->attack_damage;
            break;
        }
    }


    compute_observations(env);
}

Color COLORS[8] = {
    (Color){0, 255, 255, 255},
    (Color){255, 0, 0, 255},
    (Color){0, 255, 0, 255},
    (Color){255, 255, 0, 255},
    (Color){255, 0, 255, 255},
    (Color){0, 0, 255, 255},
    (Color){128, 255, 0, 255},
    (Color){255, 128, 0, 255},
};

// Required function. Should handle creating the client on first call
void c_render(School* env) {
    if (env->client == NULL) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(env->width, env->height, "PufferLib School");
        SetTargetFPS(30);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->ship = LoadModel("glider.glb");
        //env->client->ship = LoadModel("resources/puffer.glb");

        Camera3D camera = { 0 };
                                                           //
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
        camera.fovy = 45.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
        camera.position = (Vector3){ 3*env->size_x, 2*env->size_y, 3*env->size_z};
        camera.target = (Vector3){ 0, 0, 0};
        env->client->camera = camera;
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    UpdateCamera(&env->client->camera, CAMERA_ORBITAL);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(env->client->camera);

        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            DrawSphere((Vector3){factory->x, factory->y, factory->z}, 0.01, COLORS[factory->item]);
        }

        for (int i=0; i<env->num_agents; i++) {
            Entity* agent = &env->agents[i];
            /*
            DrawLine3D(
                (Vector3){agent->x, agent->y, agent->z},
                (Vector3){agent->x + agent->vx, agent->y + agent->vy, agent->z + agent->vz},
                COLORS[agent->item]
            );
            DrawSphere((Vector3){agent->x, agent->y, agent->z}, 0.01, COLORS[agent->item]);
            */

            /*
            Matrix transform = {
                1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy), agent->x,
                2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx), agent->y,
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2), agent->z,
                0, 0, 0, 1
            };
            */
            Vector3 pos = {agent->x, agent->y, agent->z};
            Vector3 look = {agent->orientation.x, agent->orientation.y, agent->orientation.z};
            Vector3 up = {0.0f, 1.0f, 0.0f};
            env->client->ship.transform = MatrixLookAt(pos, look, up);

            float v_norm = sqrtf(agent->vx*agent->vx + agent->vy*agent->vy + agent->vz*agent->vz);
            float xx = agent->vx / v_norm;
            float yy = agent->vy / v_norm;
            float zz = agent->vz / v_norm;
            float pitch = asinf(-yy);
            float yaw = atan2f(xx, zz);

            //Vector3 angle = {agent->pitch, agent->yaw, agent->roll};
            Vector3 angle = {pitch, yaw, 0.0f};

            /*
            rlPushMatrix();
            rlTranslatef(0.0f, 0.0f, 0.0f);
            rlRotatef(-90.0f, 0, 0, 90.0f);
            DrawModel(env->client->ship, (Vector3){0, 0, 0}, 1.0f, WHITE);
            rlPopMatrix();
            */

            env->client->ship.transform = MatrixRotateXYZ(angle);
            DrawModel(env->client->ship, (Vector3){agent->x, agent->y, agent->z}, 0.01f, COLORS[agent->item]);

            if (agent->target >= 0) {
                Entity* target = &env->agents[agent->target];
                DrawLine3D(
                    (Vector3){agent->x, agent->y, agent->z},
                    (Vector3){target->x, target->y, target->z},
                    COLORS[agent->item]
                );
            }
        }

        DrawCubeWires(
            (Vector3){0, 0, 0},
            2*env->size_x, 2*env->size_y, 2*env->size_z,
            (Color){0, 255, 255, 128}
        );

    EndMode3D();
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(School* env) {
    free(env->agents);
    free(env->factories);
    if (env->client != NULL) {
        Client* client = env->client;
        //UnloadTexture(client->sprites);
        CloseWindow();
        free(client);
    }
}

// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Simulation properties
#define GRID_SIZE 10.0f
#define MARGIN (GRID_SIZE - 1)
#define V_TARGET 0.1f
#define RING_RAD 2.0f
#define RING_MARGIN 4.0f
#define DT 0.02f

// Corner to corner distance
#define MAX_DIST sqrtf(3*(2*GRID_SIZE)*(2*GRID_SIZE))

// Physical constants for the drone
#define MASS 1.0f       // kg
#define IXX 0.01f       // kgm^2
#define IYY 0.01f       // kgm^2
#define IZZ 0.02f       // kgm^2
#define ARM_LEN 0.1f    // m
#define K_THRUST 3e-5f  // thrust coefficient
#define K_ANG_DAMP 0.2f // angular damping coefficient
#define K_DRAG 1e-6f    // drag (torque) coefficient
#define B_DRAG 0.1f     // linear drag coefficient
#define GRAVITY 9.81f   // m/s^2
#define MAX_RPM 750.0f  // rad/s
#define MAX_VEL 50.0f   // m/s
#define MAX_OMEGA 50.0f // rad/s

#define TASK_IDLE 0
#define TASK_HOVER 1
#define TASK_ORBIT 2
#define TASK_FOLLOW 3
#define TASK_LINE 4
#define TASK_CONGO 5
#define TASK_PLANE 6
#define TASK_N 7

char* TASK_NAMES[TASK_N] = {"Idle", "Hover", "Orbit", "Follow", "Line", "Congo", "Plane"};

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float perf;
    float n;
};

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;
    Quat orientation;
    Vec3 normal;
    float radius;
} Ring;

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

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }

static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

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

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

Quat rndquat() {
    float u1 = rndf(0.0f, 1.0f);
    float u2 = rndf(0.0f, 1.0f);
    float u3 = rndf(0.0f, 1.0f);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * M_PI * u2;
    float pi_2_u3 = 2.0f * M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);

    return q;
}

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    // Trailing path buffer (for rendering only)
    Trail* trails;
};

typedef struct {
    Vec3 spawn_pos;
    Vec3 pos; // global position (x, y, z)
    Vec3 vel;   // linear velocity (u, v, w)
    Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega; // angular velocity (p, q, r)
    
    Vec3 target_pos;
    Vec3 target_vel;
   
    float last_abs_reward;
    float last_target_reward;
    float last_collision_reward;
    float episode_return;
    int episode_length;
    float score;
} Drone;

typedef struct DroneSwarm DroneSwarm;
struct DroneSwarm {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;

    int task;
    int num_agents;
    Drone* agents;

    Client *client;
};

void init(DroneSwarm *env) {
    env->agents = calloc(env->num_agents, sizeof(Drone));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneSwarm *env, int idx) {
    Drone *agent = &env->agents[idx];
    env->log.score += agent->score;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.perf += agent->score / (float)agent->episode_length;
    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
}

Drone* nearest_drone(DroneSwarm* env, Drone *agent) {
    float min_dist = 999999.0f;
    Drone *nearest = NULL;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *other = &env->agents[i];
        if (other == agent) {
            continue;
        }
        float dx = agent->pos.x - other->pos.x;
        float dy = agent->pos.y - other->pos.y;
        float dz = agent->pos.z - other->pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    return nearest;
}

void compute_observations(DroneSwarm *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->vel);
        Vec3 drone_up_world = quat_rotate(agent->quat, (Vec3){0.0f, 0.0f, 1.0f});

        env->observations[idx++] = linear_vel_body.x / MAX_VEL;
        env->observations[idx++] = linear_vel_body.y / MAX_VEL;
        env->observations[idx++] = linear_vel_body.z / MAX_VEL;

        env->observations[idx++] = agent->omega.x / MAX_OMEGA;
        env->observations[idx++] = agent->omega.y / MAX_OMEGA;
        env->observations[idx++] = agent->omega.z / MAX_OMEGA;

        env->observations[idx++] = drone_up_world.x;
        env->observations[idx++] = drone_up_world.y;
        env->observations[idx++] = drone_up_world.z;

        env->observations[idx++] = agent->quat.w;
        env->observations[idx++] = agent->quat.x;
        env->observations[idx++] = agent->quat.y;
        env->observations[idx++] = agent->quat.z;

        env->observations[idx++] = agent->pos.x / GRID_SIZE;
        env->observations[idx++] = agent->pos.y / GRID_SIZE;
        env->observations[idx++] = agent->pos.z / GRID_SIZE;

        env->observations[idx++] = agent->spawn_pos.x / GRID_SIZE;
        env->observations[idx++] = agent->spawn_pos.y / GRID_SIZE;
        env->observations[idx++] = agent->spawn_pos.z / GRID_SIZE;

        env->observations[idx++] = (agent->target_pos.x - agent->pos.x) / GRID_SIZE;
        env->observations[idx++] = (agent->target_pos.y - agent->pos.y) / GRID_SIZE;
        env->observations[idx++] = (agent->target_pos.z - agent->pos.z) / GRID_SIZE;

        env->observations[idx++] = agent->last_collision_reward;
        env->observations[idx++] = agent->last_target_reward;
        env->observations[idx++] = agent->last_abs_reward;

        Drone* nearest = nearest_drone(env, agent);
        env->observations[idx++] = (nearest->pos.x - agent->pos.x) / GRID_SIZE;
        env->observations[idx++] = (nearest->pos.y - agent->pos.y) / GRID_SIZE;
        env->observations[idx++] = (nearest->pos.z - agent->pos.z) / GRID_SIZE;
    }
}

void move_target(DroneSwarm* env, Drone *agent) {
    agent->target_pos.x += agent->target_vel.x;
    agent->target_pos.y += agent->target_vel.y;
    agent->target_pos.z += agent->target_vel.z;
    if (agent->target_pos.x < -GRID_SIZE || agent->target_pos.x > GRID_SIZE) {
        agent->target_vel.x = -agent->target_vel.x;
    }
    if (agent->target_pos.y < -GRID_SIZE || agent->target_pos.y > GRID_SIZE) {
        agent->target_vel.y = -agent->target_vel.y;
    }
    if (agent->target_pos.z < -GRID_SIZE || agent->target_pos.z > GRID_SIZE) {
        agent->target_vel.z = -agent->target_vel.z;
    }
}

void set_target_idle(DroneSwarm* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){rndf(-MARGIN, MARGIN), rndf(-MARGIN, MARGIN), rndf(-MARGIN, MARGIN)};
    agent->target_vel = (Vec3){rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET)};
}

void set_target_hover(DroneSwarm* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = agent->pos;
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_orbit(DroneSwarm* env, int idx) {
    // Fibbonacci sphere algorithm
    float R = 8.0f;
    float phi = PI * (sqrt(5.0f) - 1.0f);
    float y = 1.0f - 2*((float)idx / (float)env->num_agents);
    float radius = sqrtf(1.0f - y*y);

    float theta = phi * idx;

    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){R*x, R*z, R*y}; // convert to z up 
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_follow(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    if (idx == 0) {
        set_target_idle(env, idx);
    } else {
        agent->target_pos = env->agents[0].target_pos;
        agent->target_vel = env->agents[0].target_vel;
    }
}

void set_target_line(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    float d = ((float)idx / (float)env->num_agents)*2*(GRID_SIZE - 1) - (GRID_SIZE - 1);
    agent->target_pos = (Vec3){d, d, d};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_congo(DroneSwarm* env, int idx) {
    if (idx == 0) {
        set_target_idle(env, idx);
        return;
    }
    Drone* follow = &env->agents[idx - 1];
    Drone* lead = &env->agents[idx];
    lead->target_pos = follow->target_pos;
    lead->target_vel = follow->target_vel;

    for (int i = 0; i < 10; i++) {
        move_target(env, lead);
    }
}

void set_target_plane(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    float x = (float)(idx % 8);
    float y = (float)(idx / 8);
    x = 2.0f*x - 7;
    y = 2.0f*y - 7;
    agent->target_pos = (Vec3){x, y, 0.0f};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target(DroneSwarm* env, int idx) {
    if (env->task == TASK_IDLE) {
        set_target_idle(env, idx);
    } else if (env->task == TASK_HOVER) {
        set_target_hover(env, idx);
    } else if (env->task == TASK_ORBIT) {
        set_target_orbit(env, idx);
    } else if (env->task == TASK_FOLLOW) {
        set_target_follow(env, idx);
    } else if (env->task == TASK_LINE) {
        set_target_line(env, idx);
    } else if (env->task == TASK_CONGO) {
        set_target_congo(env, idx);
    } else if (env->task == TASK_PLANE) {
        set_target_plane(env, idx);
    }
}

float compute_reward(DroneSwarm* env, Drone *agent) {
    // Distance reward
    float dx = (agent->pos.x - agent->target_pos.x);
    float dy = (agent->pos.y - agent->target_pos.y);
    float dz = (agent->pos.z - agent->target_pos.z);
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float dist_reward = 1.0 - dist/MAX_DIST;

    // Density penalty
    Drone *nearest = nearest_drone(env, agent);
    dx = agent->pos.x - nearest->pos.x;
    dy = agent->pos.y - nearest->pos.y;
    dz = agent->pos.z - nearest->pos.z;
    float min_dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float density_reward = 1.0f;
    if (min_dist < 0.3f) {
        density_reward = 0.0f;
    }

    float last_abs_reward = agent->last_collision_reward * agent->last_target_reward;
    float abs_reward = dist_reward * density_reward;
    float delta_reward = abs_reward - last_abs_reward;

    agent->last_collision_reward = density_reward;
    agent->last_target_reward = dist_reward;
    agent->last_abs_reward = abs_reward;

    agent->episode_return += delta_reward;
    agent->episode_length++;
    agent->score += abs_reward;

    return delta_reward;
}

void reset_agent(DroneSwarm* env, Drone *agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->score = 0.0f;
    agent->pos = (Vec3){rndf(-9, 9), rndf(-9, 9), rndf(-9, 9)};
    agent->spawn_pos = agent->pos;
    agent->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->omega = (Vec3){0.0f, 0.0f, 0.0f};
    agent->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    compute_reward(env, agent);
}

void c_reset(DroneSwarm *env) {
    env->tick = 0;
    env->task = rand() % 4;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env, i);
    }

    compute_observations(env);
}

void c_step(DroneSwarm *env) {
    env->tick = (env->tick + 1) % 512;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        clamp4(&env->actions[4*i], -1.0f, 1.0f);

        // motor thrusts
        float T[4];
        for (int j = 0; j < 4; j++) {
            T[j] = K_THRUST * powf((env->actions[4*i + j] + 1.0f) * 0.5f * MAX_RPM, 2.0f);
        }

        // body frame net force
        Vec3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

        // body frame torques
        Vec3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
                  K_DRAG * (T[0] - T[1] + T[2] - T[3])};

        // applies angular damping to torques
        M.x -= K_ANG_DAMP * agent->omega.x;
        M.y -= K_ANG_DAMP * agent->omega.y;
        M.z -= K_ANG_DAMP * agent->omega.z;

        // body frame force -> world frame force
        Vec3 F_world = quat_rotate(agent->quat, F_body);

        // world frame linear drag
        F_world.x -= B_DRAG * agent->vel.x;
        F_world.y -= B_DRAG * agent->vel.y;
        F_world.z -= B_DRAG * agent->vel.z;

        // world frame gravity
        Vec3 accel = {F_world.x / MASS, F_world.y / MASS, (F_world.z / MASS) - GRAVITY};

        // from the definition of q dot
        Quat omega_q = {0.0f, agent->omega.x, agent->omega.y, agent->omega.z};
        Quat q_dot = quat_mul(agent->quat, omega_q);

        q_dot.w *= 0.5f;
        q_dot.x *= 0.5f;
        q_dot.y *= 0.5f;
        q_dot.z *= 0.5f;

        // integrations
        agent->pos.x += agent->vel.x * DT;
        agent->pos.y += agent->vel.y * DT;
        agent->pos.z += agent->vel.z * DT;

        agent->vel.x += accel.x * DT;
        agent->vel.y += accel.y * DT;
        agent->vel.z += accel.z * DT;

        agent->omega.x += (M.x / IXX) * DT;
        agent->omega.y += (M.y / IYY) * DT;
        agent->omega.z += (M.z / IZZ) * DT;

        clamp3(&agent->vel, -MAX_VEL, MAX_VEL);
        clamp3(&agent->omega, -MAX_OMEGA, MAX_OMEGA);

        agent->quat.w += q_dot.w * DT;
        agent->quat.x += q_dot.x * DT;
        agent->quat.y += q_dot.y * DT;
        agent->quat.z += q_dot.z * DT;

        quat_normalize(&agent->quat);

        // check out of bounds
        bool out_of_bounds = agent->pos.x < -GRID_SIZE || agent->pos.x > GRID_SIZE ||
                             agent->pos.y < -GRID_SIZE || agent->pos.y > GRID_SIZE ||
                             agent->pos.z < -GRID_SIZE || agent->pos.z > GRID_SIZE;

        move_target(env, agent);

        // Delta reward
        float reward = compute_reward(env, agent);
        env->rewards[i] += reward;

        if (out_of_bounds) {
            env->rewards[i] -= 1;
            env->terminals[i] = 1;
            add_log(env, i);
            reset_agent(env, agent, i);
        } else if (env->tick >= 511) {
            env->terminals[i] = 1;
            add_log(env, i);
        }
    }
    if (env->tick >= 511) {
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DroneSwarm *env) {
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client *make_client(DroneSwarm *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DroneSwarm");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 40.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 10.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    // Initialize trail buffer
    client->trails = (Trail*)calloc(env->num_agents, sizeof(Trail));
    for (int i = 0; i < env->num_agents; i++) {
        Trail* trail = &client->trails[i];
        trail->index = 0;
        trail->count = 0;
        for (int j = 0; j < TRAIL_LENGTH; j++) {
            trail->pos[j] = env->agents[i].pos;
        }
    }

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(DroneSwarm *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    if (IsKeyPressed(KEY_SPACE)) {
        env->task = (env->task + 1) % TASK_N;
        for (int i = 0; i < env->num_agents; i++) {
            set_target(env, i);
        }
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        Trail *trail = &client->trails[i];
        trail->pos[trail->index] = agent->pos;
        trail->index = (trail->index + 1) % TRAIL_LENGTH;
        if (trail->count < TRAIL_LENGTH) {
            trail->count++;
        }
        if (env->terminals[i]) {
            trail->index = 0;
            trail->count = 0;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_SIZE * 2.0f,
        GRID_SIZE * 2.0f, GRID_SIZE * 2.0f, WHITE);

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        // draws drone body
        DrawSphere((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, 0.3f, RED);

        // draws rotors according to thrust
        float T[4];
        for (int j = 0; j < 4; j++) {
            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * MAX_RPM;
            T[j] = K_THRUST * rpm * rpm;
        }

        const float rotor_radius = 0.15f;
        const float visual_arm_len = ARM_LEN * 4.0f;

        Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                      {-visual_arm_len, 0.0f, 0.0f},
                                      {0.0f, +visual_arm_len, 0.0f},
                                      {0.0f, -visual_arm_len, 0.0f}};

        Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};

        for (int j = 0; j < 4; j++) {
            Vec3 world_off = quat_rotate(agent->quat, rotor_offsets_body[j]);

            Vector3 rotor_pos = {agent->pos.x + world_off.x, agent->pos.y + world_off.y,
                                 agent->pos.z + world_off.z};

            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * MAX_RPM;
            float intensity = 0.75f + 0.25f * (rpm / MAX_RPM);

            Color rotor_color = (Color){(unsigned char)(base_colors[j].r * intensity),
                                        (unsigned char)(base_colors[j].g * intensity),
                                        (unsigned char)(base_colors[j].b * intensity), 255};

            DrawSphere(rotor_pos, rotor_radius, rotor_color);

            DrawCylinderEx((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                           BLACK);
        }

        // draws line with direction and magnitude of velocity / 10
        if (norm3(agent->vel) > 0.1f) {
            DrawLine3D((Vector3){agent->pos.x, agent->pos.y, agent->pos.z},
                       (Vector3){agent->pos.x + agent->vel.x * 0.1f, agent->pos.y + agent->vel.y * 0.1f,
                                 agent->pos.z + agent->vel.z * 0.1f},
                       MAGENTA);
        }

        // Draw trailing path
        Trail *trail = &client->trails[i];
        if (trail->count <= 2) {
            continue;
        }
        for (int j = 0; j < trail->count - 1; j++) {
            int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
            float alpha = (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
            Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
            DrawLine3D((Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
                       (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
                       trail_color);
        }

    }

    if (IsKeyDown(KEY_TAB)) {
        for (int i = 0; i < env->num_agents; i++) {
            Drone *agent = &env->agents[i];
            Vec3 target_pos = agent->target_pos;
            DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.25f, (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
    DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16, PUFF_WHITE);

    EndDrawing();
}

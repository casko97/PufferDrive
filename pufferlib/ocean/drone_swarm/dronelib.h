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
#define HORIZON 1024

// Simulation properties
#define GRID_SIZE 10.0f
#define MARGIN (GRID_SIZE - 1)
#define V_TARGET 0.05f
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

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float collision_rate;
    float oob;
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
    Vec3 pos;
    Quat orientation;
    Vec3 normal;
    float radius;
} Ring;

Ring rndring(void) {
    Ring ring;

    ring.pos.x = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);
    ring.pos.y = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);
    ring.pos.z = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);

    ring.orientation = rndquat();

    Vec3 base_normal = {0.0f, 0.0f, 1.0f};
    ring.normal = quat_rotate(ring.orientation, base_normal);

    ring.radius = RING_RAD;

    return ring;
}

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 spawn_pos;
    Vec3 pos; // global position (x, y, z)
    Vec3 prev_pos;
    Vec3 vel;   // linear velocity (u, v, w)
    Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega; // angular velocity (p, q, r)
    
    Vec3 target_pos;
    Vec3 target_vel;
   
    float last_abs_reward;
    float last_target_reward;
    float last_collision_reward;
    float episode_return;
    float collisions;
    int episode_length;
    float score;
} Drone;

void move_drone(Drone* drone, float* actions) {
    clamp4(actions, -1.0f, 1.0f);

    // motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        T[i] = K_THRUST * powf((actions[i] + 1.0f) * 0.5f * MAX_RPM, 2.0f);
    }


    // body frame net force
    Vec3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame torques
    Vec3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
              K_DRAG * (T[0] - T[1] + T[2] - T[3])};

    // applies angular damping to torques
    M.x -= K_ANG_DAMP * drone->omega.x;
    M.y -= K_ANG_DAMP * drone->omega.y;
    M.z -= K_ANG_DAMP * drone->omega.z;

    // body frame force -> world frame force
    Vec3 F_world = quat_rotate(drone->quat, F_body);

    // world frame linear drag
    F_world.x -= B_DRAG * drone->vel.x;
    F_world.y -= B_DRAG * drone->vel.y;
    F_world.z -= B_DRAG * drone->vel.z;

    // world frame gravity
    Vec3 accel = {F_world.x / MASS, F_world.y / MASS, (F_world.z / MASS) - GRAVITY};

    // from the definition of q dot
    Quat omega_q = {0.0f, drone->omega.x, drone->omega.y, drone->omega.z};
    Quat q_dot = quat_mul(drone->quat, omega_q);

    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // integrations
    drone->pos.x += drone->vel.x * DT;
    drone->pos.y += drone->vel.y * DT;
    drone->pos.z += drone->vel.z * DT;

    drone->vel.x += accel.x * DT;
    drone->vel.y += accel.y * DT;
    drone->vel.z += accel.z * DT;

    drone->omega.x += (M.x / IXX) * DT;
    drone->omega.y += (M.y / IYY) * DT;
    drone->omega.z += (M.z / IZZ) * DT;

    clamp3(&drone->vel, -MAX_VEL, MAX_VEL);
    clamp3(&drone->omega, -MAX_OMEGA, MAX_OMEGA);

    drone->quat.w += q_dot.w * DT;
    drone->quat.x += q_dot.x * DT;
    drone->quat.y += q_dot.y * DT;
    drone->quat.z += q_dot.z * DT;

    quat_normalize(&drone->quat);
}

float check_ring(Drone* drone, Ring* ring) {
    // previous dot product negative if on the 'entry' side of the ring's plane
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);

    // new dot product positive if on the 'exit' side of the ring's plane
    float new_dot = dot3(sub3(drone->pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    // if we have crossed the plane of the ring
    if (valid_dir || invalid_dir) {
        // find intesection with ring's plane
        Vec3 dir = sub3(drone->pos, drone->prev_pos);
        float t = -prev_dot / dot3(ring->normal, dir); // possible nan

        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float dist = norm3(sub3(intersection, ring->pos));

        // reward or terminate based on distance to ring center
        if (dist < (ring->radius - 0.5) && valid_dir) {
            return 1.0f;
        } else if (dist < ring->radius + 0.5) {
            return -1.0f;
        }
    }
    return 0.0f;
}


#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "raylib.h"

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define VELOCITY_CAP 3
#define MARGIN_TURN_FACTOR 0.2f
#define VISUAL_RANGE 20
#define VISUAL_RANGE_SQUARED (VISUAL_RANGE * VISUAL_RANGE)
#define PROTECTED_RANGE 2
#define PROTECTED_RANGE_SQUARED (PROTECTED_RANGE * PROTECTED_RANGE)
#define CENTERING_FACTOR 0.0005f
#define AVOID_FACTOR 0.05f
#define MATCHING_FACTOR 0.05f
#define MAX_AVOID_DISTANCE_SQUARED (PROTECTED_RANGE_SQUARED * AVOID_FACTOR)
#define MAX_AVG_POSITION_SQUARED  (VISUAL_RANGE_SQUARED * CENTERING_FACTOR)
#define MAX_AVG_VELOCITY_SQUARED  (VELOCITY_CAP * 4 * MATCHING_FACTOR)
#define MAX_MARGIN_TURN_FACTOR 2 * MARGIN_TURN_FACTOR
#define WIDTH 800
#define HEIGHT 600
#define BOID_WIDTH 32
#define BOID_HEIGHT 32
#define BOID_TEXTURE_PATH "./resources/puffers_128.png"
#define LOG_INTERVAL 128

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    float x;
    float y;
} Velocity;

typedef struct {
    float x;
    float y;
    Velocity velocity;
} Boid;

typedef struct Client Client;
typedef struct {
    // an array of shape (num_boids, 4) with the 4 values correspoinding to (x, y, velocity x, velocity y)
    float* observations;
    // an array of shape (num_boids, 2) with the 2 values correspoinding to (velocity x, velocity y)
    float* actions;
    // an array of shape (num_boids, 1) with the reward for each boid
    float* rewards;
    unsigned char* terminals; // Not being used but is required by env_binding.h
    Boid* boids;
    unsigned int num_boids;
    int max_reward;
    int min_reward;
    int max_steps;
    int tick;
    Log log;
    Log* boid_logs;
    Client* client;
} Boids;

static void add_log(Boids *env, unsigned boid_indx) {
    env->log.perf           += env->boid_logs[boid_indx].perf;
    env->log.score          += env->boid_logs[boid_indx].score;
    env->log.episode_return += env->boid_logs[boid_indx].episode_return;
    env->log.n              += 1.0f;

    /* clear per-boid log for next episode */
    env->boid_logs[boid_indx] = (Log){0};
}

static inline float flmax(float a, float b) { return a > b ? a : b; }
static inline float flmin(float a, float b) { return a > b ? b : a; }
static inline float flclip(float x,float lo,float hi) { return flmin(hi,flmax(lo,x)); }
static inline float rndf(float lo,float hi) { return lo + (float)rand()/(float)RAND_MAX*(hi-lo); }

static void respawn_boid(Boids *env, unsigned int i) {
    env->boids[i].x = rndf(LEFT_MARGIN, WIDTH  - RIGHT_MARGIN);
    env->boids[i].y = rndf(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN);
    env->boids[i].velocity.x = 0;
    env->boids[i].velocity.y = 0;
    env->boid_logs[i]       = (Log){0};
}

void init(Boids *env) {
    env->boids = (Boid*)calloc(env->num_boids, sizeof(Boid));
    env->boid_logs = (Log*)calloc(env->num_boids, sizeof(Log));
    env->log = (Log){0};
    env->tick = 0;
    env->max_steps = 1000;

    /* positions & velocities */
    for (unsigned i = 0; i < env->num_boids; i++) {
        env->boids[i].x = rndf(LEFT_MARGIN, WIDTH  - RIGHT_MARGIN);
        env->boids[i].y = rndf(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN);
        env->boids[i].velocity.x = 0;
        env->boids[i].velocity.y = 0;
    }

    /* reward bounds for min-max normalisation */
    env->max_reward = 0;
    env->min_reward = -flmax(MAX_AVOID_DISTANCE_SQUARED * env->num_boids,
                             MAX_AVG_POSITION_SQUARED) - MAX_MARGIN_TURN_FACTOR;
}

void free_allocated(Boids* env) {
    free(env->boids);
    free(env->boid_logs);
}

static void compute_observations(Boids *env) {
    unsigned base_indx;

    for (unsigned boids_indx = 0; boids_indx < env->num_boids; boids_indx++) {
        base_indx = boids_indx * 4;
        env->observations[base_indx + 0] = env->boids[boids_indx].x;
        env->observations[base_indx + 1] = env->boids[boids_indx].y;
        env->observations[base_indx + 2] = env->boids[boids_indx].velocity.x;
        env->observations[base_indx + 3] = env->boids[boids_indx].velocity.y;
    }
}

void c_reset(Boids *env) {
    env->log = (Log){0};
    env->tick = 0;
    for (unsigned boid_indx = 0; boid_indx < env->num_boids; boid_indx++) {
        respawn_boid(env, boid_indx);
    }
    compute_observations(env);
}

void c_step(Boids *env) {
    Boid* current_boid;
    Boid observed_boid;
    float vx_sum, vy_sum, x_sum, y_sum, reward;
    float diff_x, diff_y, dist2, x_avg, y_avg, vx_avg, vy_avg;
    unsigned visual_count;
    float current_boid_reward;
    float total_reward = 0.0f;

    env->tick++;

    for (unsigned current_indx = 0; current_indx < env->num_boids; current_indx++) {
        // apply action
        current_boid = &env->boids[current_indx];

        current_boid->velocity.x += flclip(env->actions[current_indx * 2 + 0], -VELOCITY_CAP, VELOCITY_CAP);
        current_boid->velocity.y += flclip(env->actions[current_indx * 2 + 1], -VELOCITY_CAP, VELOCITY_CAP);

        current_boid->x = flclip(current_boid->x + current_boid->velocity.x, 0, WIDTH  - BOID_WIDTH);
        current_boid->y = flclip(current_boid->y + current_boid->velocity.y, 0, HEIGHT - BOID_HEIGHT);

        // reward calculation
        reward = 0, visual_count = 0, vx_sum = 0, vy_sum = 0, x_sum = 0, y_sum = 0;

        for (unsigned observed_indx = 0; observed_indx < env->num_boids; observed_indx++) {
            if (current_indx == observed_indx) continue;
            observed_boid = env->boids[observed_indx];
            diff_x = current_boid->x - observed_boid.x;
            diff_y = current_boid->y - observed_boid.y;
            dist2 = diff_x*diff_x + diff_y*diff_y;

            if (dist2 < PROTECTED_RANGE_SQUARED) {
                reward -= (PROTECTED_RANGE_SQUARED - dist2) * AVOID_FACTOR;
            } else if (dist2 < VISUAL_RANGE_SQUARED) {
                x_sum += observed_boid.x;
                y_sum += observed_boid.y;
                vx_sum += observed_boid.velocity.x;
                vy_sum += observed_boid.velocity.y;
                visual_count++;
            }
        }

        if (visual_count) {
            x_avg  = x_sum  / visual_count;
            y_avg  = y_sum  / visual_count;
            vx_avg = vx_sum / visual_count;
            vy_avg = vy_sum / visual_count;

            reward -= fabsf(vx_avg - current_boid->velocity.x) * MATCHING_FACTOR;
            reward -= fabsf(vy_avg - current_boid->velocity.y) * MATCHING_FACTOR;
            reward -= fabsf(x_avg  - current_boid->x) * CENTERING_FACTOR;
            reward -= fabsf(y_avg  - current_boid->y) * CENTERING_FACTOR;
        }

        if (current_boid->y < TOP_MARGIN || current_boid->y > HEIGHT - BOTTOM_MARGIN) reward -= MARGIN_TURN_FACTOR;
        if (current_boid->x < LEFT_MARGIN || current_boid->x > WIDTH  - RIGHT_MARGIN) reward -= MARGIN_TURN_FACTOR;

        current_boid_reward = 2.0f * (reward - env->min_reward) / (env->max_reward - env->min_reward) - 1.0f;
        total_reward += current_boid_reward;

        // per-boid log update
        env->boid_logs[current_indx].episode_return += current_boid_reward;
        env->boid_logs[current_indx].episode_length += 1.0f;

        if (env->tick % LOG_INTERVAL == 0) {
            env->boid_logs[current_indx].score = env->boid_logs[current_indx].episode_return;
            env->boid_logs[current_indx].perf  = (env->boid_logs[current_indx].score/env->boid_logs[current_indx].episode_length + 1.0f)*0.5f;
            add_log(env, current_indx);
            env->tick = 0;
        }
    }

    // environment level updates
    env->rewards[0] = (env->num_boids > 0) ? total_reward / env->num_boids : 0.0f;
    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D boid_texture;
};

void c_close_client(Client* client) {
    UnloadTexture(client->boid_texture);
    CloseWindow();
    free(client);
}

Client* make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    
    client->width = WIDTH;
    client->height = HEIGHT;
    
    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);
    
    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }
    
    client->boid_texture = LoadTexture(BOID_TEXTURE_PATH);
    if (client->boid_texture.id == 0) {
        TraceLog(LOG_ERROR, "Failed to load texture: %s", BOID_TEXTURE_PATH);
        c_close_client(client);
        return NULL;
    }
    
    return client;
}

void c_render(Boids* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }
    
    if (!WindowShouldClose() && IsWindowReady()) {
        if (IsKeyDown(KEY_ESCAPE)) {
            exit(0);
        }

        BeginDrawing();
        ClearBackground((Color){6, 24, 24, 255});

        for (unsigned boid_indx = 0; boid_indx < env->num_boids; boid_indx++) {
            DrawTexturePro(
                env->client->boid_texture,
                (Rectangle){
                    (env->boids[boid_indx].velocity.x > 0) ? 0 : 128,
                    0,
                    128,
                    128,
                },
                (Rectangle){
                    env->boids[boid_indx].x,
                    env->boids[boid_indx].y,
                    BOID_WIDTH,
                    BOID_HEIGHT
                },
                (Vector2){0, 0},
                0,
                WHITE
            );
        }

        EndDrawing();
    } else {
        TraceLog(LOG_WARNING, "Window is not ready or should close");
    }
}

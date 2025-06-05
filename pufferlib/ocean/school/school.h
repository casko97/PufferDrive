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
} Client;

typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    int item;
    int episode_length;
} Agent;

typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    int item;
} Factory;

typedef struct {
    Log log;
    Client* client;
    Agent* agents;
    Factory* factories;
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
    env->agents = calloc(env->num_agents, sizeof(Agent));
    env->factories = calloc(env->num_factories, sizeof(Factory));
}

int compare_floats(const void* a, const void* b) {
    return (*(float*)a - *(float*)b) > 0;
}

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

float randi(int min, int max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

void compute_observations(School* env) {
    int obs_idx = 0;
    for (int a=0; a<env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        float dists[env->num_resources];
        for (int i=0; i<env->num_resources; i++) {
            dists[i] = 999999;
        }
        for (int f=0; f<env->num_factories; f++) {
            Factory* factory = &env->factories[f];
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
        env->observations[obs_idx++] = agent->x;
        env->observations[obs_idx++] = agent->y;
        env->observations[obs_idx++] = agent->z;
        env->observations[obs_idx++] = env->rewards[a];
        memset(&env->observations[obs_idx], 0, env->num_resources*sizeof(float));
        env->observations[obs_idx + agent->item] = 1.0f;
        obs_idx += env->num_resources;
    }
}

// Required function
void c_reset(School* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->agents[i].x = randf(-env->size_x, env->size_x);
        env->agents[i].y = randf(-env->size_y, env->size_y);
        env->agents[i].z = randf(-env->size_z, env->size_z);
        env->agents[i].item = rand() % env->num_resources;
        env->agents[i].episode_length = 0;
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

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

void c_step(School* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->terminals[i] = 0;
        env->rewards[i] = 0;
        Agent* agent = &env->agents[i];
        agent->episode_length += 1;

        float d_vx = ((float)env->actions[3*i] - 4.0f)/400.0f;
        float d_vy = ((float)env->actions[3*i + 1] - 4.0f)/400.0f;
        float d_vz = ((float)env->actions[3*i + 2] - 4.0f)/400.0f;

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

        if (rand() % env->num_agents == 0) {
            env->agents[i].x = randf(-env->size_x, env->size_x);
            env->agents[i].y = randf(-env->size_y, env->size_y);
            env->agents[i].z = randf(-env->size_z, env->size_z);
        }

        for (int f=0; f<env->num_factories; f++) {
            Factory* factory = &env->factories[f];
            float dx = (factory->x - agent->x);
            float dy = (factory->y - agent->y);
            float dz = (factory->z - agent->z);
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.1) {
                continue;
            }
            if (factory->item == agent->item) {
                agent->item = (agent->item + 1) % env->num_resources;
                env->log.perf += 1.0f;
                env->log.score += 1.0f;
                env->log.episode_length += agent->episode_length;
                env->log.n++;
                env->rewards[i] = 1.0f;
                agent->episode_length = 0;
            }
        }
    }
    for (int f=0; f<env->num_factories; f++) {
        Factory* factory = &env->factories[f];
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
    compute_observations(env);
}

Color COLORS[8] = {
    (Color){255, 0, 0, 255},
    (Color){0, 255, 0, 255},
    (Color){0, 0, 255, 255},
    (Color){255, 255, 0, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 0, 255, 255},
    (Color){128, 255, 0, 255},
    (Color){255, 128, 0, 255},
};

// Required function. Should handle creating the client on first call
void c_render(School* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib School");
        SetTargetFPS(30);
        env->client = (Client*)calloc(1, sizeof(Client));

        Camera3D camera = { 0 };
                                                           //
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
        camera.fovy = 45.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
        camera.position = (Vector3){ 2*env->size_x, env->size_y, 2*env->size_z};
        camera.target = (Vector3){ 0, 0, 0};
        env->client->camera = camera;
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(env->client->camera);

        for (int f=0; f<env->num_factories; f++) {
            Factory* factory = &env->factories[f];
            DrawCube((Vector3){factory->x, factory->y, factory->z}, 0.01, 0.01, 0.01, COLORS[factory->item]);
        }

        for (int i=0; i<env->num_agents; i++) {
            Agent* agent = &env->agents[i];
            DrawSphere((Vector3){agent->x, agent->y, agent->z}, 0.01, COLORS[agent->item]);
        }

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

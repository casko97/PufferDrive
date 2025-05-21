#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"

const float MAX_SPEED = 20.0f;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Texture2D puffer;
    Texture2D star;
} Client;

typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int ticks_since_reward;
} Agent;

typedef struct {
    float x;
    float y;
} Goal;

typedef struct {
    Log log;
    Client* client;
    Agent* agents;
    Goal* goals;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    int num_agents;
    int num_goals;
} Target;

void init(Target* env) {
    env->agents = calloc(env->num_agents, sizeof(Agent));
    env->goals = calloc(env->num_goals, sizeof(Goal));
}

void update_goals(Target* env) {
    for (int a=0; a<env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        for (int g=0; g<env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            float dx = (goal->x - agent->x);
            float dy = (goal->y - agent->y);
            float dist = sqrt(dx*dx + dy*dy);
            if (dist > 32) {
                continue;
            }
            goal->x = rand() % env->width;
            goal->y = rand() % env->height;
            env->rewards[a] = 1.0f;
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->ticks_since_reward;
            env->log.episode_return += 1.0f;
            env->log.n++;
            agent->ticks_since_reward = 0;
        }
    }
    int obs_idx = 0;
    for (int a=0; a<env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        for (int g=0; g<env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            env->observations[obs_idx++] = (goal->x - agent->x)/env->width;
            env->observations[obs_idx++] = (goal->y - agent->y)/env->height;
        }
        for (int a=0; a<env->num_agents; a++) {
            Agent* other = &env->agents[a];
            env->observations[obs_idx++] = (other->x - agent->x)/env->width;
            env->observations[obs_idx++] = (other->y - agent->y)/env->height;
        }
        env->observations[obs_idx++] = agent->heading/(2*PI);
        env->observations[obs_idx++] = env->rewards[a];
        env->observations[obs_idx++] = agent->x/env->width;
        env->observations[obs_idx++] = agent->y/env->height;
    }
}

void c_reset(Target* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->agents[i].x = rand() % env->width;
        env->agents[i].y = rand() % env->height;
        env->agents[i].ticks_since_reward = 0;
    }
    for (int i=0; i<env->num_goals; i++) {
        env->goals[i].x = rand() % env->width;
        env->goals[i].y = rand() % env->height;
    }
    update_goals(env);
}

void c_step(Target* env) {
    //memset(env->rewards, 0, env->num_agents*sizeof(float));

    for (int i=0; i<env->num_agents; i++) {
        env->rewards[i] = 0;
        Agent* agent = &env->agents[i];
        agent->ticks_since_reward += 1;

        agent->heading += ((float)env->actions[2*i] - 4.0f)/12.0f;
        if (agent->heading < 0) {
            agent->heading += 2*PI;
        } else if (agent->heading > 2*PI) {
            agent->heading -= 2*PI;
        }

        agent->speed += 1.0f*((float)env->actions[2*i + 1] - 2.0f);
        if (agent->speed > MAX_SPEED) {
            agent->speed = MAX_SPEED;
        } else if (agent->speed < -MAX_SPEED) {
            agent->speed = -MAX_SPEED;
        }

        agent->x += agent->speed*cosf(agent->heading);
        if (agent->x < 0) {
            agent->x = 0;
        } else if (agent->x > env->width) {
            agent->x = env->width;
        }

        agent->y += agent->speed*sinf(agent->heading);
        if (agent->y < 0) {
            agent->y = 0;
        } else if (agent->y > env->height) {
            agent->y = env->height;
        }
        if (agent->ticks_since_reward % 512 == 0) {
            env->agents[i].x = rand() % env->width;
            env->agents[i].y = rand() % env->height;
        }
    }
    update_goals(env);
}

void c_close(Target* env) {
    free(env->agents);
    free(env->goals);
}

Client* make_client(Target* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(env->width, env->height, "PufferLib Target");
    SetTargetFPS(60);

    client->puffer = LoadTexture("resources/puffers_128.png");
    client->star = LoadTexture("resources/star.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Target* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i=0; i<env->num_goals; i++) {
        Goal* goal = &env->goals[i];
        DrawTexture(
            env->client->star,
            goal->x - 32,
            goal->y - 32,
            WHITE
        );
    }

    for (int i=0; i<env->num_agents; i++) {
        Agent* agent = &env->agents[i];
        float heading = agent->heading;
        DrawTexturePro(
            env->client->puffer,
            (Rectangle){
                (heading < PI/2 || heading > 3*PI/2) ? 0 : 128,
                0, 128, 128,
            },
            (Rectangle){
                agent->x - 64,
                agent->y - 64,
                128,
                128
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    EndDrawing();
}

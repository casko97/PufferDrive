#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"

#define LEFT 0
#define NOOP 1
#define RIGHT 2

#define PI2 PI * 2

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    float width;   // 640
    float height;  // 480
    float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    float rrw_ang; // right right whisker angle
    float max_whisker_length;
    float turn_pi_frac; //  (pi / turn_pi_frac is the turn angle)
    float maxv;    // 5    
    Texture2D car;
} Client;

typedef struct WhiskerRacer {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;

    // Game State
    int width;
    int height;
    int score;
    int tick;
    int max_score;
    int half_max_score;
    int circuit;
    int frameskip;
    int continuous;

    // Car State
    float px;
    float py;
    float ang;
    float vx;
    float vy;
    float v;
    float vang;
    //float* brick_x;
    //float* brick_y;
    //float* brick_states;

    // Physics Constraints
    float maxv;
    float accel;
    float turn_pi_frac;

    // Track/Map
    float asdf;

    // Whiskers
    int num_whiskers;
    float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    float rrw_ang; // right right whisker angle
    float* whisker_angles;    // Array of whisker angles (radians)
    float max_whisker_length;
    float* whisker_lengths;   // Array of current whisker readings
} WhiskerRacer;

void init(WhiskerRacer* env) {
    env->tick = 0;
    /*
    env->num_bricks = env->brick_rows * env->brick_cols;
    assert(env->num_bricks > 0);

    env->brick_x = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_y = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_states = (float*)calloc(env->num_bricks, sizeof(float));
    env->num_balls = -1;
    generate_brick_positions(env);
    */

    // todo not sure what to do here yet maybe nothing else
}

void allocate(WhiskerRacer* env) {
    init(env);
    //env->observations = (float*)calloc(11 + env->num_bricks, sizeof(float));
    env->observations = (float*)calloc(9, sizeof(float)); // todo double check this later
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(WhiskerRacer* env) {
    // todo
    //free(env->brick_x);
    //free(env->brick_y);
    //free(env->brick_states);
}

void free_allocated(WhiskerRacer* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(WhiskerRacer* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
}

void compute_observations(WhiskerRacer* env) {
    /*
    env->observations[0] = env->paddle_x / env->width;
    env->observations[1] = env->paddle_y / env->height;
    env->observations[2] = env->ball_x / env->width;
    env->observations[3] = env->ball_y / env->height;
    env->observations[4] = env->ball_vx / 512.0f;
    env->observations[5] = env->ball_vy / 512.0f;
    env->observations[6] = env->balls_fired / 5.0f;
    env->observations[7] = env->score / 864.0f;
    env->observations[8] = env->num_balls / 5.0f;
    env->observations[9] = env->paddle_width / (2.0f * HALF_PADDLE_WIDTH);
    for (int i = 0; i < env->num_bricks; i++) {
        env->observations[10 + i] = env->brick_states[i];
    }
    */

    env->observations[0] = env->px / env->width;
    env->observations[1] = env->py / env->height;
    env->observations[2] = env->ang / (PI2);
    env->observations[3] = env->vx / env->maxv;
    env->observations[4] = env->vy / env->maxv;
    // other env->observations are probably based on the whiskers but idk how to represent that yet
    // I'm guessing I just send the length of the whisker, which is cut off at track boundary or sent as maxL for whisker
}

void calc_whisker_lengths(WhiskerRacer* env) {
    // Start from car, see how far down whisker length till it hits green (grass off track)
    // Do this for all 5 whiskers
    // Normalize them to max whisker length
    // if whisker touches no grass, just return the 1.0 for max whisker length after normalization
}

void get_random_start(WhiskerRacer* env) {
    if (env->circuit == 1) {
        // Each choice: {xmin, ymin, xmax, ymax, angle}
        const float choices[3][5] = {
            {330, 360, 475, 395, 0.0f},
            {490, 350, 520, 370, -M_PI/4.0f},
            {490, 125, 530, 345, -M_PI/2.0f}
        };
        int num_choices = 3;
        int ch = rand() % num_choices;

        float xmin = choices[ch][0];
        float ymin = choices[ch][1];
        float xmax = choices[ch][2];
        float ymax = choices[ch][3];
        float base_angle = choices[ch][4];

        // Random position within the rectangle
        env->px = (float)(rand() % ((int)(xmax - xmin + 1))) + xmin;
        env->py = (float)(rand() % ((int)(ymax - ymin + 1))) + ymin;

        // Random angle: base_angle + random offset in [-PI/6, PI/6]
        float angle_offset = ((float)rand() / RAND_MAX) * (M_PI/3.0f) - (M_PI/6.0f);
        env->ang = base_angle + angle_offset;
    }
}



void reset_round(WhiskerRacer* env) {
    /* old breakout logic
    env->balls_fired = 0;
    env->hit_brick = false;
    env->hits = 0;
    env->ball_speed = 256;
    env->paddle_width = 2 * HALF_PADDLE_WIDTH;

    env->paddle_x = env->width / 2.0 - env->paddle_width / 2;
    env->paddle_y = env->height - env->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2 - env->ball_width / 2);
    env->ball_y = env->height / 2 - 30;

    env->ball_vx = 0.0;
    env->ball_vy = 0.0;
    */

    get_random_start(env);
    env->vx = 0.0f;
    env->vy = 0.0f;
    env->v = env->maxv;
    env->vang = 0.0f;

}

void c_reset(WhiskerRacer* env) {
    env->score = 0;
    //env->num_balls = 5;
    //for (int i = 0; i < env->num_bricks; i++) {
    //    env->brick_states[i] = 0.0;
    //}
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
}

void step_frame(WhiskerRacer* env, float action) {
    // todo Still incomplete, still has some Breakout logic
    float act = 0.0;
    //if (env->balls_fired == 0) {
    //    env->balls_fired = 1;
    //    float direction = M_PI / 3.25f;

    //    env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
    //    env->ball_vx = sin(direction) * env->ball_speed * TICK_RATE;
    //    if (rand() % 2 == 0) {
    //        env->ball_vx = -env->ball_vx;
    //    }
    //}   
    if (action == LEFT) {
        act = -1.0;
    } else if (action == RIGHT) {
        act = 1.0;
    }
    if (env->continuous){
        act = action;
    }
    //env->paddle_x += act * 620 * TICK_RATE;
    //if (env->paddle_x <= 0){
    //    env->paddle_x = fmaxf(0, env->paddle_x);
    //} else {
    //    env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
    //}

    //Handle collisions. 
    //Regular timestepping is done only if there are no collisions.
    /*
    if(!handle_collisions(env)){
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if (env->ball_y >= env->paddle_y + env->paddle_height) {
        env->num_balls -= 1;
        reset_round(env);
    }
    if (env->num_balls < 0 || env->score == env->max_score) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    */
}

void c_step(WhiskerRacer* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }

    compute_observations(env);
}

Client* make_client(WhiskerRacer* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->llw_ang = env->llw_ang;
    client->flw_ang = env->flw_ang;
    client->frw_ang = env->frw_ang;
    client->rrw_ang = env->rrw_ang;
    client->max_whisker_length = env->max_whisker_length;
    client->turn_pi_frac = env->turn_pi_frac;
    client->maxv = env->maxv;

    InitWindow(env->width, env->height, "PufferLib Whisker Racer");
    SetTargetFPS(60 / env->frameskip);

    //client->ball = LoadTexture("resources/shared/puffers_128.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(WhiskerRacer* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();

    // Place Track Picture

    // Draw Car

    // Draw Whiskers Conditionally

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    //DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();
}

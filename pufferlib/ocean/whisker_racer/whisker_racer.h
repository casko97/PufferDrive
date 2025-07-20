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

    // Whiskers
    int num_whiskers;
    //float* whisker_angles;    // Array of whisker angles (radians)
    float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    float rrw_ang; // right right whisker angle
    //float* whisker_lengths;   // Array of current whisker readings
    float llw_length;
    float flw_length;
    float ffw_length;
    float frw_length;
    float rrw_length;
    float max_whisker_length;
    
    Texture2D track_texture;
} WhiskerRacer;

void load_track_texture(WhiskerRacer* env) {
    // Construct the filename: "img/circuits/circuit-<circuit>.jpg"
    char fname[128];
    snprintf(fname, sizeof(fname), "./img/circuits/circuit-%d.jpg", env->circuit);

    // Unload previous texture if already loaded
    if (env->track_texture.id != 0) {
        UnloadTexture(env->track_texture);
    }

    // Load the new texture
    env->track_texture = LoadTexture(fname);

    // Optional: error handling
    if (env->track_texture.id == 0) {
        printf("Failed to load track texture: %s\n", fname);
        // Handle error as needed (exit, fallback, etc.)
    }
}

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
    // I might need to run get_random_start()
    load_track_texture(env);
}

void allocate(WhiskerRacer* env) {
    init(env);
    env->observations = (float*)calloc(10, sizeof(float)); // todo double check this later
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(WhiskerRacer* env) {
    // todo not sure if this is even needed
    //free(env->brick_x);
    //free(env->brick_y);
    //free(env->brick_states);
    if (env->track_texture.id != 0) {
        UnloadTexture(env->track_texture);
        env->track_texture.id = 0;
    }
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
    env->observations[0] = env->px / env->width;
    env->observations[1] = env->py / env->height;
    env->observations[2] = env->ang / (PI2);
    env->observations[3] = env->vx / env->maxv;
    env->observations[4] = env->vy / env->maxv;
    env->observations[5] = env->llw_length;
    env->observations[6] = env->flw_length;
    env->observations[7] = env->ffw_length;
    env->observations[8] = env->frw_length;
    env->observations[9] = env->rrw_length;
}

static int is_green(Color color) {
    return (color.g > 150 && color.g > color.r + 40 && color.g > color.b + 40);
}

static inline Color GetTexturePixelColor(Texture2D texture, int x, int y) {
    Image img = LoadImageFromTexture(texture);
    Color color = {0};
    if (x >= 0 && x < img.width && y >= 0 && y < img.height) {
        Color* pixels = LoadImageColors(img);
        color = pixels[y * img.width + x];
        UnloadImageColors(pixels);
    }
    UnloadImage(img);
    return color;
}

void calc_whisker_lengths(WhiskerRacer* env) {
    // Start from car, see how far down whisker length till it hits green (grass off track)
    // Do this for all 5 whiskers
    // Normalize them to max whisker length
    // if whisker touches no grass, just return the 1.0 for max whisker length after normalization

    // Whisker angles relative to car's heading
    float angles[5] = {
        env->ang + env->llw_ang, // left-left
        env->ang + env->flw_ang, // front-left
        env->ang,                // front-forward
        env->ang + env->frw_ang, // front-right
        env->ang + env->rrw_ang  // right-right
    };

    float* lengths[5] = {
        &env->llw_length,
        &env->flw_length,
        &env->ffw_length,
        &env->frw_length,
        &env->rrw_length
    };

    for (int w = 0; w < 5; ++w) {
        float angle = angles[w];
        float max_len = env->max_whisker_length;
        float step = 1.0f; // pixel step size

        float hit_len = max_len;
        for (float l = 0.0f; l <= max_len; l += step) {
            float wx = env->px + l * cosf(angle);
            float wy = env->py + l * sinf(angle);

            int ix = (int)roundf(wx);
            int iy = (int)roundf(wy);

            // Bounds check
            if (ix < 0 || ix >= env->width || iy < 0 || iy >= env->height) {
                hit_len = l;
                break;
            }

            // Sample pixel from track texture
            Color color = GetTexturePixelColor(env->track_texture, ix, iy);

            if (is_green(color)) {
                hit_len = l;
                break;
            }
        }
        // Normalize
        float norm_len = hit_len / max_len;
        if (norm_len > 1.0f) norm_len = 1.0f;
        if (norm_len < 0.0f) norm_len = 0.0f;
        *lengths[w] = norm_len;
    }
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
        env->v = 5.0;
        env->llw_length = 0.25;
        env->flw_length = 0.50;
        env->ffw_length = 1.00;
        env->frw_length = 0.50;
        env->rrw_length = 0.25;

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
        env->ang = env->ang + PI * env->turn_pi_frac;
    } else if (action == RIGHT) {
        act = 1.0;
        env->ang = env->ang - PI * env->turn_pi_frac;
    }
    if (env->ang > PI2) {
        env->ang = env->ang - PI2;
    }
    else if (env->ang < 0) {
        env->ang = env->ang + PI2;
    }
    if (env->continuous){
        act = action;
    }
    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
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

    DrawTexture(env->track_texture, 0, 0, WHITE);

    // Draw Car

    // Draw Whiskers Conditionally

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    //DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();
}

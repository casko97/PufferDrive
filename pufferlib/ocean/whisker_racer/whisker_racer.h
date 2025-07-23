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

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define TRACK_WIDTH 50
#define MAX_CONTROL_POINTS 8
#define BEZIER_RESOLUTION 16  // Points per bezier segment

typedef struct {
    Vector2 position;
} ControlPoint;

typedef struct {
    ControlPoint controls[MAX_CONTROL_POINTS];
    int num_points;
    Vector2 centerline[MAX_CONTROL_POINTS * BEZIER_RESOLUTION];
    Vector2 inner_edge[MAX_CONTROL_POINTS * BEZIER_RESOLUTION];
    Vector2 outer_edge[MAX_CONTROL_POINTS * BEZIER_RESOLUTION];
    int total_points;
} Track;

typedef struct TrackManager {
    Image track_image;
    Color* track_pixels;
    Texture2D track_texture;
    int circuit;
    int ref_count;  // Track how many environments are using this
    int is_loaded;
} TrackManager;

static TrackManager g_track_manager = {0};

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
    int circuit;
    int render;
    int debug;
    Texture2D track_texture;
    Image track_image;
    Color* track_pixels;
} Client;

typedef struct WhiskerRacer {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int debug;

    float reward_yellow;
    float reward_green;
    float gamma;

    Track track;

    // Game State
    int width;
    int height;
    float score;
    int tick;
    int max_score;
    int half_max_score;
    int circuit;
    int frameskip;
    int render;
    int continuous;

    // Car State
    float px;
    float py;
    float ang;
    float vx;
    float vy;
    float v;
    float vang;

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

    float inv_width;
    float inv_height;
    float inv_maxv;
    float inv_pi2;
    
    Texture2D track_texture;
    Image track_image;
    Color* track_pixels;
} WhiskerRacer;

void unload_track() {
    if (!g_track_manager.is_loaded) return;

    g_track_manager.ref_count--;

    // Only unload when no environments are using it
    if (g_track_manager.ref_count <= 0) {
        if (g_track_manager.track_texture.id != 0) {
            UnloadTexture(g_track_manager.track_texture);
            g_track_manager.track_texture.id = 0;
        }
        if (g_track_manager.track_image.data != NULL) {
            UnloadImage(g_track_manager.track_image);
            g_track_manager.track_image.data = NULL;
        }
        if (g_track_manager.track_pixels != NULL) {
            UnloadImageColors(g_track_manager.track_pixels);
            g_track_manager.track_pixels = NULL;
        }
        g_track_manager.is_loaded = 0;
        g_track_manager.ref_count = 0;
    }
}

void load_track_once(int circuit, int need_texture) {
    if (g_track_manager.is_loaded && g_track_manager.circuit == circuit) {
        g_track_manager.ref_count++;
        return; // Already loaded for this circuit
    }

    // If we have a different circuit loaded, unload it first
    if (g_track_manager.is_loaded) {
        unload_track();
    }

    char fname[128];
    snprintf(fname, sizeof(fname), "./pufferlib/ocean/whisker_racer/img/circuits/circuit-1.jpg");

    g_track_manager.track_image = LoadImage(fname);
    g_track_manager.track_pixels = LoadImageColors(g_track_manager.track_image);

    if (need_texture) {
        g_track_manager.track_texture = LoadTextureFromImage(g_track_manager.track_image);
    } else {
        g_track_manager.track_texture.id = 0;
    }

    g_track_manager.circuit = circuit;
    g_track_manager.ref_count = 1;
    g_track_manager.is_loaded = 1;

    if (g_track_manager.track_image.data == NULL) {
        printf("Failed to load track image: %s\n", fname);
        g_track_manager.is_loaded = 0;
    }
}

void init(WhiskerRacer* env) {
    if (env->debug) printf("init\n");
    env->tick = 0;

    env->debug = 0;

    load_track_once(env->circuit, env->render);

    env->track_pixels = g_track_manager.track_pixels;
    env->track_texture = g_track_manager.track_texture;
    env->track_image = g_track_manager.track_image;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_maxv = 1.0f / env->maxv;
    env->inv_pi2 = 1.0f / PI2;

    if (env->debug) printf("end init\n");
}

void allocate(WhiskerRacer* env) {
    if (env->debug) printf("allocate");
    init(env);
    env->observations = (float*)calloc(11, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (env->debug) printf("end allocate");
}

void c_close(WhiskerRacer* env) {
    unload_track();
}

void free_allocated(WhiskerRacer* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(WhiskerRacer* env) {
    if (env->debug) printf("add_log\n");
    env->log.episode_length += env->tick;
    if (env->log.episode_length > 0.01f) {
    }
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
    if (env->debug) printf("end add_log\n");
}

void compute_observations(WhiskerRacer* env) {
    //if (env->debug) printf("compute_observations\n");
    env->observations[0] = env->px * env->inv_width;
    env->observations[1] = env->py * env->inv_height;
    env->observations[2] = env->ang * env->inv_pi2;
    env->observations[3] = env->vx * env->inv_maxv;
    env->observations[4] = env->vy * env->inv_maxv;
    env->observations[5] = env->llw_length;
    env->observations[6] = env->flw_length;
    env->observations[7] = env->ffw_length;
    env->observations[8] = env->frw_length;
    env->observations[9] = env->rrw_length;
    env->observations[10] = env->score / 100.0f;
    if (env->debug) printf("float0 %.3f \n", env->observations[0]);
    if (env->debug) printf("float1 %.3f \n", env->observations[1]);
    if (env->debug) printf("float2 %.3f \n", env->observations[2]);
    if (env->debug) printf("float3 %.3f \n", env->observations[3]);
    if (env->debug) printf("float4 %.3f \n", env->observations[4]);
    if (env->debug) printf("float5 %.3f \n", env->observations[5]);
    if (env->debug) printf("float6 %.3f \n", env->observations[6]);
    if (env->debug) printf("float7 %.3f \n", env->observations[7]);
    if (env->debug) printf("float8 %.3f \n", env->observations[8]);
    if (env->debug) printf("float9 %.3f \n", env->observations[9]);
    if (env->debug) printf("float10 %.3f \n", env->observations[10]);
    if (env->debug) printf("\n\n\n");
    //if (env->debug) printf("end compute_observations\n");
}

static inline int get_color_type(Color color) {
    if (color.g > 130 && color.g > color.r + 40 && color.g > color.b + 40)
        return 1; // Green
    if (color.r > 30 && color.g > 30 && color.b < 10)
        return 2; // Yellow
    if ((color.r & color.g & color.b) > 220)  // Bit trick for min
        return 3; // White
    return 0; // Other
}

void calc_whisker_lengths(WhiskerRacer* env) {
    float max_len = env->max_whisker_length;
    float inv_max_len = 1.0f / max_len;
    float step = 1.0f; // pixel step size
    int width = env->width;
    int height = env->height;

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

        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float hit_len = max_len;
        for (float l = 0.0f; l <= max_len; l += step) {
            float wx = env->px + l * cos_a;
            float wy = env->py + l * sin_a;

            int ix = (int)roundf(wx);
            int iy = (int)roundf(wy);

            // Bounds check
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                hit_len = l;
                break;
            }

            Color color = env->track_pixels[iy * width + ix];

            if (get_color_type(color) == 1) { // Car drove off track
                env->rewards[0] += env->reward_green;
                env->score -= 0.001f;
            }
            else if (get_color_type(color) == 2) {
                env->rewards[0] += env->reward_yellow;
                env->score += 1.0;
            }
            else if (get_color_type(color) == 3) {
                env->rewards[0] += 1.0;
                env->score += 10.0;
            }
        }
        // Normalize
        *lengths[w] = fminf(1.0f, fmaxf(0.0f, hit_len * inv_max_len));
    }
}

void get_random_start(WhiskerRacer* env) {
    //if (env->debug) printf("get_random_start\n");
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
    get_random_start(env);
    env->vx = 0.0f;
    env->vy = 0.0f;
    env->v = env->maxv;
    env->vang = 0.0f;

}

void c_reset(WhiskerRacer* env) {
    //if (env->debug) printf("c_reset\n");
    env->score = 0;
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
    //if (env->debug) printf("end c_reset\n");
}

void step_frame(WhiskerRacer* env, float action) {
    //if (env->debug) printf("step_frame\n");
    float act = 0.0;

    if (action == LEFT) {
        act = -1.0;
        env->ang += PI / env->turn_pi_frac;
    } else if (action == RIGHT) {
        act = 1.0;
        env->ang -= PI / env->turn_pi_frac;
    }
    if (env->ang > PI2) {
        env->ang -= PI2;
    }
    else if (env->ang < 0) {
        env->ang += PI2;
    }
    if (env->continuous){
        act = action;
    }
    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
    env->px = env->px + env->vx;
    env->py = env->py + env->vy;
    if (env->px < 0) env->px = 0;
    else if (env->px > env->width) env->px = env->width;
    if (env->py < 0) env->py = 0;
    else if (env->py > env->height) env->py = env->height;

    //if (env->debug) printf("calc_whisker_lengths\n");
    calc_whisker_lengths(env);

    // What color is the car touching
    int ix = (int)env->px;
    int iy = (int)env->py;
    Color color = env->track_pixels[iy * env->track_image.width + ix];
    if (env->debug) printf("Color at (%d, %d): r=%d g=%d b=%d\n", ix, iy, (int)color.r, (int)color.g, (int)color.b);
    if (get_color_type(color) == 1) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(WhiskerRacer* env) {
    //if (env->debug) printf("c_step\n");
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }

    //if (env->debug) printf("compute_observations\n");
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
    client->circuit = env->circuit;

    InitWindow(env->width, env->height, "PufferLib Whisker Racer");
    SetTargetFPS(60 / env->frameskip);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

// Cubic Bezier curve evaluation
Vector2 EvaluateCubicBezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;
    
    Vector2 result;
    result.x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x;
    result.y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y;
    return result;
}

// Get the derivative (tangent) of cubic Bezier curve
Vector2 GetBezierDerivative(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    
    Vector2 result;
    result.x = -3 * uu * p0.x + 3 * uu * p1.x - 6 * u * t * p1.x + 6 * u * t * p2.x - 3 * tt * p2.x + 3 * tt * p3.x;
    result.y = -3 * uu * p0.y + 3 * uu * p1.y - 6 * u * t * p1.y + 6 * u * t * p2.y - 3 * tt * p2.y + 3 * tt * p3.y;
    return result;
}

// Normalize a vector
Vector2 NormalizeVector(Vector2 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y);
    if (length == 0.0f) return (Vector2){0, 0};
    return (Vector2){v.x / length, v.y / length};
}

// Get perpendicular vector (rotated 90 degrees)
Vector2 GetPerpendicular(Vector2 v) {
    return (Vector2){-v.y, v.x};
}

/// Generate random control points for a closed circuit with F1-style characteristics
void GenerateRandomControlPoints(WhiskerRacer* env) {
    env->track.num_points = 6;
    
    float center_x = SCREEN_WIDTH * 0.5f;
    float center_y = SCREEN_HEIGHT * 0.5f;
    
    // Create a simple closed loop with varied corner radii
    for (int i = 0; i < env->track.num_points; i++) {
        float angle = (PI2 * i) / env->track.num_points;
        
        float corner_radius;
        if (i == 1 || i == 4) {
            corner_radius = 40.0f + (rand() % 30);
        } else if (i == 0 || i == 3) {
            corner_radius = 150.0f + (rand() % 40);
        } else {
            corner_radius = 220.0f + (rand() % 30);
        }
        
        float x_offset = (rand() % 20 - 10); // ±10px variation
        float y_offset = (rand() % 20 - 10); // ±10px variation
        
        env->track.controls[i].position.x = center_x + corner_radius * cosf(angle) + x_offset;
        env->track.controls[i].position.y = center_y + corner_radius * 0.8f * sinf(angle) + y_offset;
    }
}

void GenerateTrackCenterline(WhiskerRacer* env) {
    int point_index = 0;
    
    for (int i = 0; i < env->track.num_points; i++) {
        Vector2 p0 = env->track.controls[i].position;
        Vector2 p3 = env->track.controls[(i + 1) % env->track.num_points].position;
        
        // Create control points for varied turn sharpness
        Vector2 prev = env->track.controls[(i - 1 + env->track.num_points) % env->track.num_points].position;
        Vector2 next = env->track.controls[(i + 2) % env->track.num_points].position;
        
        // Calculate control points
        Vector2 dir1 = NormalizeVector((Vector2){p3.x - prev.x, p3.y - prev.y});
        Vector2 dir2 = NormalizeVector((Vector2){next.x - p0.x, next.y - p0.y});
        
        float dist = sqrtf((p3.x - p0.x) * (p3.x - p0.x) + (p3.y - p0.y) * (p3.y - p0.y));
        
        // Vary control length based on corner type - shorter = sharper turns
        float control_length;
        if (i == 1 || i == 3) {
            control_length = dist * 0.1f; // Sharp hairpins
        } else if (i == 0 || i == 4) {
            control_length = dist * 0.2f; // Medium corners
        } else {
            control_length = dist * 0.3f; // Sweeping turns
        }
        
        Vector2 p1 = (Vector2){p0.x + dir1.x * control_length, p0.y + dir1.y * control_length};
        Vector2 p2 = (Vector2){p3.x - dir2.x * control_length, p3.y - dir2.y * control_length};
        
        // Generate points along this Bezier segment
        for (int j = 0; j < BEZIER_RESOLUTION && point_index < MAX_CONTROL_POINTS * BEZIER_RESOLUTION - 1; j++) {
            float t = (float)j / (float)BEZIER_RESOLUTION;
            env->track.centerline[point_index] = EvaluateCubicBezier(p0, p1, p2, p3, t);
            point_index++;
        }
    }
    
    env->track.total_points = point_index;
}

// Generate inner and outer env->track edges
void GenerateTrackEdges(WhiskerRacer* env) {
    for (int i = 0; i < env->track.total_points; i++) {
        Vector2 current = env->track.centerline[i];
        Vector2 next = env->track.centerline[(i + 1) % env->track.total_points];
        
        // Calculate tangent direction
        Vector2 tangent = NormalizeVector((Vector2){next.x - current.x, next.y - current.y});
        Vector2 normal = GetPerpendicular(tangent);
        
        // Create inner and outer edges
        float half_width = TRACK_WIDTH * 0.5f;
        env->track.inner_edge[i] = (Vector2){current.x - normal.x * half_width, current.y - normal.y * half_width};
        env->track.outer_edge[i] = (Vector2){current.x + normal.x * half_width, current.y + normal.y * half_width};
    }
}

void GenerateRandomTrack(WhiskerRacer* env) {
    GenerateRandomControlPoints(env);
    GenerateTrackCenterline(env);
    GenerateTrackEdges(env);
}

void c_render(WhiskerRacer* env) {

    env->render = 1;
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    // If we need texture but don't have it, load it
    if (g_track_manager.track_texture.id == 0 && g_track_manager.is_loaded) {
        g_track_manager.track_texture = LoadTextureFromImage(g_track_manager.track_image);
        env->track_texture = g_track_manager.track_texture;
    }

    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

/*
    BeginDrawing();

    ClearBackground(BLACK);

    if (env->track_texture.id != 0) {
        DrawTexture(env->track_texture, 0, 0, WHITE);
    } else {
        DrawText("IMG FAIL", 10, 40, 20, RED);
    }

    float car_width = 24.0f;
    float car_height = 12.0f;
    Vector2 car_center = {env->px, env->py};
    Rectangle car_rect = {
        env->px - car_width / 2.0f,
        env->py - car_height / 2.0f,
        car_width,
        car_height
    };
    Vector2 origin = {car_width / 2.0f, car_height / 2.0f};
    DrawRectanglePro(
        (Rectangle){env->px, env->py, car_width, car_height},
        origin,
        env->ang * 180.0f / PI,
        (Color){255, 0, 255, 255}
    );

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    EndDrawing();
    */

    GenerateRandomTrack(env);

    Vector2* outer_points = malloc(sizeof(Vector2) * (env->track.total_points + 3));
    //outer_points[0] = (Vector2){SCREEN_WIDTH*0.5f, SCREEN_HEIGHT*0.5f};
    for (int i = 0; i < env->track.total_points; i++) {
        outer_points[i] = env->track.centerline[i];
    }

    // Without enough overlap it draws a C rather than an O
    outer_points[env->track.total_points] = env->track.centerline[0];
    outer_points[env->track.total_points + 1] = env->track.centerline[1];
    outer_points[env->track.total_points + 2] = env->track.centerline[2];

    BeginDrawing();
    SetWindowSize(640, 480);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground(GREEN);
    DrawSplineBasis(outer_points, env->track.total_points + 3, 50.0f, BLACK);
    free(outer_points);
    EndDrawing();
}

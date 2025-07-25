#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <time.h>

#define LEFT 0
#define NOOP 1
#define RIGHT 2

#define PI2 PI * 2

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define MAX_CONTROL_POINTS 8
#define NUM_RADIAL_SECTORS 16
#define MAX_BEZIER_RESOLUTION 16

typedef struct {
    Vector2 position;
} ControlPoint;

typedef struct {
    ControlPoint controls[MAX_CONTROL_POINTS];
    int num_points;
    Vector2 centerline[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 inner_edge[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 outer_edge[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    int total_points;
    Vector2 curbs[MAX_CONTROL_POINTS][4];
    int curb_count;
} Track;

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
    //float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    //float rrw_ang; // right right whisker angle
    float max_whisker_length;
    float turn_pi_frac; //  (pi / turn_pi_frac is the turn angle)
    float maxv;    // 5
    int circuit;
    int render;
    int debug;
} Client;

typedef struct WhiskerRacer {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int debug;
    unsigned int rng;

    float ftmp1;
    float ftmp2;
    float ftmp3;
    float ftmp4;

    int render_many;

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
    int current_sector;
    int sectors_completed[NUM_RADIAL_SECTORS];
    int total_sectors_crossed;
    int track_width;
    int num_radial_sectors;
    int num_points;
    int bezier_resolution;
    float inv_bezier_res;

    // Car State
    float px;
    float py;
    float ang;
    float vx;
    float vy;
    float v;
    int near_point_idx;

    // Physics Constraints
    float maxv;
    float turn_pi_frac;

    // Whiskers
    int num_whiskers;
    //float* whisker_angles;    // Array of whisker angles (radians)
    Vector2 whisker_dirs[2];
    float w_ang;
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
} WhiskerRacer;

void init(WhiskerRacer* env) {
    if (env->debug) printf("init\n");
    env->tick = 0;
    srand(env->rng);

    env->debug = 0;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_maxv = 1.0f / env->maxv;
    env->inv_pi2 = 1.0f / PI2;
    env->inv_bezier_res = 1.0f / env->bezier_resolution;

    env->flw_ang = -env->w_ang;
    env->frw_ang = env->w_ang;

    GenerateRandomTrack(env);

    if (env->debug) printf("end init\n");
}

void allocate(WhiskerRacer* env) {
    if (env->debug) printf("allocate");
    init(env);
    env->observations = (float*)calloc(3, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (env->debug) printf("end allocate");
}

void c_close(WhiskerRacer* env) {
    //unload_track();
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
    env->observations[0] = env->flw_length;
    env->observations[1] = env->frw_length;
    env->observations[2] = env->score / 100.0f;
    if (env->debug) printf("float0 %.3f \n", env->observations[0]);
    if (env->debug) printf("float1 %.3f \n", env->observations[1]);
    if (env->debug) printf("float2 %.3f \n", env->observations[2]);
    if (env->debug) printf("\n\n\n");
    //if (env->debug) printf("end compute_observations\n");
}

Client* make_client(WhiskerRacer* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    //client->llw_ang = env->llw_ang;
    client->flw_ang = env->flw_ang;
    client->frw_ang = env->frw_ang;
    //client->rrw_ang = env->rrw_ang;
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

void reset_round(WhiskerRacer* env) {
    get_random_start(env);
    reset_radial_progress(env);
    env->vx = 0.0f;
    env->vy = 0.0f;
    env->v = env->maxv;
}

void c_reset(WhiskerRacer* env) {
    env->score = 0;
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
}

void step_frame(WhiskerRacer* env, float action) {
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
    //env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->llw_ang), sinf(env->ang + env->llw_ang)}; // left-left
    //env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)}; // front-left
    //env->whisker_dirs[2] = (Vector2){cosf(env->ang), sinf(env->ang)};                               // front-forward
    //env->whisker_dirs[3] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)}; // front-right
    //env->whisker_dirs[4] = (Vector2){cosf(env->ang + env->rrw_ang), sinf(env->ang + env->rrw_ang)}; // right-
    env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)};

    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
    env->px = env->px + env->vx;
    env->py = env->py + env->vy;
    if (env->px < 0) env->px = 0;
    else if (env->px > env->width) env->px = env->width;
    if (env->py < 0) env->py = 0;
    else if (env->py > env->height) env->py = env->height;

    calc_whisker_lengths(env);

    update_radial_progress(env);
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

void get_random_start(WhiskerRacer* env) {
    int start_idx = rand() % env->track.total_points;
    env->near_point_idx = start_idx;

    env->px = env->track.centerline[start_idx].x;
    env->py = env->track.centerline[start_idx].y;

    int next_idx = (start_idx + 1) % env->track.total_points;
    float dx = env->track.centerline[next_idx].x - env->px;
    float dy = env->track.centerline[next_idx].y - env->py;
    env->ang = atan2f(dy, dx);

    //env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->llw_ang), sinf(env->ang + env->llw_ang)};
    //env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)};
    //env->whisker_dirs[2] = (Vector2){cosf(env->ang), sinf(env->ang)};
    //env->whisker_dirs[3] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)};
    //env->whisker_dirs[4] = (Vector2){cosf(env->ang + env->rrw_ang), sinf(env->ang + env->rrw_ang)};
    env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)};

    env->v = env->maxv;
    //env->llw_length = 0.25f;
    env->flw_length = 0.50f;
    //env->ffw_length = 1.00f;
    env->frw_length = 0.50f;
    //env->rrw_length = 0.25f;
}

// ============================================ Per Step Calculations =============================

// Line segment intersection helper function
// Returns 1 if intersection found, 0 otherwise
// If intersection found, stores the parameter t in *t_out (0 <= t <= 1 along the whisker ray)
static inline int line_segment_intersect(Vector2 ray_start, Vector2 ray_dir, float ray_length,
                                       Vector2 seg_start, Vector2 seg_end, float* t_out) {
    Vector2 seg_dir = {seg_end.x - seg_start.x, seg_end.y - seg_start.y};
    Vector2 diff = {seg_start.x - ray_start.x, seg_start.y - ray_start.y};

    float cross_rd_sd = ray_dir.x * seg_dir.y - ray_dir.y * seg_dir.x;

    // Lines are parallel
    if (fabsf(cross_rd_sd) < 1e-3f) {
        return 0;
    }

    float cross_diff_sd = diff.x * seg_dir.y - diff.y * seg_dir.x;
    float cross_diff_rd = diff.x * ray_dir.y - diff.y * ray_dir.x;

    float t = cross_diff_sd / cross_rd_sd;
    float u = cross_diff_rd / cross_rd_sd;

    // Check if intersection is within both line segments
    if (t >= 0.0f && t <= ray_length && u >= 0.0f && u <= 1.0f) {
        *t_out = t;
        return 1;
    }

    return 0;
}

void calc_whisker_lengths(WhiskerRacer* env) {
    float max_len = env->max_whisker_length;
    float inv_max_len = 1.0f / max_len;

    update_nearest_point(env);

    float* lengths[2] = {
        //&env->llw_length,
        &env->flw_length,
        //&env->ffw_length,
        &env->frw_length
        //&env->rrw_length
    };

    Vector2 car_pos = {env->px, env->py};

    for (int w = 0; w < 2; ++w) {
        Vector2 whisker_dir = env->whisker_dirs[w];
        float min_hit_distance = max_len;

        int window_size = 20;
        for (int offset = -window_size/2; offset <= window_size/2; offset++) {
            int i = (env->near_point_idx + offset + env->track.total_points) % env->track.total_points;
            int next_i = (i + 1) % env->track.total_points;

            float t;

            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                     env->track.inner_edge[i], env->track.inner_edge[next_i], &t)) {
                if (t < min_hit_distance) {
                    min_hit_distance = t;
                }
                if (t < 0.05) break;
            }

            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                     env->track.outer_edge[i], env->track.outer_edge[next_i], &t)) {
                if (t < min_hit_distance) {
                    min_hit_distance = t;
                }
                if (t < 0.05) break;
            }
        }

        *lengths[w] = fminf(1.0f, fmaxf(0.0f, min_hit_distance * inv_max_len));

        if (*lengths[w] < 0.05f) { // Car has crashed
            for (int j = 0; j < 2; j++) *lengths[j] = 0.0f;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
        }
    }
}

void update_nearest_point(WhiskerRacer* env) {
    float min_dist_sq = 100000;
    int closest_seg = env->near_point_idx; // Start with current
    Vector2 car_pos = {env->px, env->py};

    // Only search +/- 5 points around current nearest
    int search_range = 3;
    for (int offset = 0; offset <= search_range; offset++) {
        int i = (env->near_point_idx + offset + env->track.total_points) % env->track.total_points;

        Vector2 center = env->track.centerline[i];
        float dx = car_pos.x - center.x;
        float dy = car_pos.y - center.y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_seg = i;
        }
    }

    env->near_point_idx = closest_seg;
}

int find_closest_centerline_segment(WhiskerRacer* env) {
    float min_dist_sq = 100000;
    int closest_seg = 0;
    Vector2 car_pos = {env->px, env->py};

    for (int i = 0; i < env->track.total_points; i++) {
        Vector2 center = env->track.centerline[i];
        float dx = car_pos.x - center.x;
        float dy = car_pos.y - center.y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_seg = i;
        }
    }
    return closest_seg;
}

void update_radial_progress(WhiskerRacer* env) {
    float center_x = SCREEN_WIDTH * 0.5f;
    float center_y = SCREEN_HEIGHT * 0.5f;

    float angle = atan2f(env->py - center_y, env->px - center_x);

    if (angle < 0) angle += PI2;

    int sector = (int)(angle / (PI2 / 16.0f));
    sector = sector % env->num_radial_sectors;

    if (sector != env->current_sector) {
        int expected_next = (env->current_sector + 1) % 16;
        if (sector == expected_next) {
            if (!env->sectors_completed[sector]) {
                env->sectors_completed[sector] = 1;
                env->total_sectors_crossed++;
                env->rewards[0] += env->reward_yellow;
                env->score += env->reward_yellow;
            } else { // full lap
                env->rewards[0] += env->reward_yellow;
                env->score += env->reward_yellow;
            }
        }
        env->current_sector = sector;
    }
}

void reset_radial_progress(WhiskerRacer* env) {
    float center_x = SCREEN_WIDTH * 0.5f;
    float center_y = SCREEN_HEIGHT * 0.5f;

    float angle = atan2f(env->py - center_y, env->px - center_x);
    if (angle < 0) angle += PI2;

    env->current_sector = (int)(angle / (PI2 / 16.0f)) % 16;

    for (int i = 0; i < 16; i++) {
        env->sectors_completed[i] = 0;
    }
    env->total_sectors_crossed = 0;
}

// ============================================ End Per Step Calculations =============================

// ================================================================== BEZIER ==============================================

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

void GenerateRandomControlPoints(WhiskerRacer* env) {
    float center_x = SCREEN_WIDTH * 0.5f;
    float center_y = SCREEN_HEIGHT * 0.5f;

    int n = env->num_points;

    // Randomly choose distinct, non-adjacent indices for tight and medium corners
    int opt1 = rand() % n;
    int opt2;
    do {
        opt2 = rand() % n;
    } while (opt2 == opt1 || abs(opt2 - opt1) == 1 || abs(opt2 - opt1) == n - 1);

    int opt3, opt4;
    do {
        opt3 = rand() % n;
    } while (opt3 == opt1 || opt3 == opt2);

    do {
        opt4 = rand() % n;
    } while (opt4 == opt1 || opt4 == opt2 || opt4 == opt3 || abs(opt4 - opt3) == 1 || abs(opt4 - opt3) == n - 1);

    // Generate control points
    for (int i = 0; i < n; i++) {
        float angle = (PI2 * i) / n;

        float dist_from_center;
        if (i == opt1) {
            dist_from_center = 100.0f + (rand() % 30);
        } else if (i == opt2 || i == opt3) {
            dist_from_center = 150.0f + (rand() % 40);
        } else {
            dist_from_center = 220.0f + (rand() % 30);
        }

        env->track.controls[i].position.x = center_x + dist_from_center * cosf(angle);
        env->track.controls[i].position.y = center_y + dist_from_center * 0.8f * sinf(angle);
    }

    for (int i = 0; i < n; i++) {
        Vector2 prev = env->track.controls[(i - 1 + n) % n].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i + 1) % n].position;

        float vx1 = prev.x - curr.x;
        float vy1 = prev.y - curr.y;
        float vx2 = next.x - curr.x;
        float vy2 = next.y - curr.y;

        float dot = vx1 * vx2 + vy1 * vy2;
        float mag1 = sqrtf(vx1 * vx1 + vy1 * vy1);
        float mag2 = sqrtf(vx2 * vx2 + vy2 * vy2);

        if (mag1 < 1e-3f || mag2 < 1e-3f) continue;

        float angle_cos = dot / (mag1 * mag2);

        if (angle_cos > 0.0f) {
            float dx = curr.x - center_x;
            float dy = curr.y - center_y;
            float dist = sqrtf(dx * dx + dy * dy);

            float adjust_scale = env->ftmp1;
            if (dist < 150.0f) adjust_scale = env->ftmp2;
            else if (dist > 200) adjust_scale = env->ftmp3;

            env->track.controls[i].position.x = center_x + dx * adjust_scale;
            env->track.controls[i].position.y = center_y + dy * adjust_scale;
        }
    }
}


void GenerateTrackCenterline(WhiskerRacer* env) {
    int point_index = 0;

    for (int i = 0; i < env->num_points; i++) {
        Vector2 p0 = env->track.controls[i].position;
        Vector2 p3 = env->track.controls[(i + 1) % env->num_points].position;

        // Create control points for varied turn sharpness
        Vector2 prev = env->track.controls[(i - 1 + env->num_points) % env->num_points].position;
        Vector2 next = env->track.controls[(i + 2) % env->num_points].position;

        // Calculate control points
        Vector2 dir1 = NormalizeVector((Vector2){p3.x - prev.x, p3.y - prev.y});
        Vector2 dir2 = NormalizeVector((Vector2){next.x - p0.x, next.y - p0.y});

        float dist = sqrtf((p3.x - p0.x) * (p3.x - p0.x) + (p3.y - p0.y) * (p3.y - p0.y));

        // Vary control length based on corner type - shorter = sharper turns
        float control_length;
        if (i == 1 || i == 3) {
            control_length = dist * 0.2f; // Sharp hairpins
        } else if (i == 0 || i == 4) {
            control_length = dist * 0.3f; // Medium corners
        } else {
            control_length = dist * 0.4f; // Sweeping turns
        }

        Vector2 p1 = (Vector2){p0.x + dir1.x * control_length, p0.y + dir1.y * control_length};
        Vector2 p2 = (Vector2){p3.x - dir2.x * control_length, p3.y - dir2.y * control_length};

        // Generate points along this Bezier segment
        for (int j = 0; j < env->bezier_resolution && point_index < MAX_CONTROL_POINTS * env->bezier_resolution - 1; j++) {
            float t = (float)j * env->inv_bezier_res;
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
        float half_width = env->track_width * 0.5f;
        env->track.inner_edge[i] = (Vector2){current.x - normal.x * half_width, current.y - normal.y * half_width};
        env->track.outer_edge[i] = (Vector2){current.x + normal.x * half_width, current.y + normal.y * half_width};
    }
}

void GenerateCurbs(WhiskerRacer* env) {
    env->track.curb_count = 0;

    for (int i = 0; i < env->num_points; i++) {
        Vector2 prev = env->track.controls[(i - 1 + env->num_points) % env->num_points].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i + 1) % env->num_points].position;

        Vector2 to_prev = {prev.x - curr.x, prev.y - curr.y};
        Vector2 to_next = {next.x - curr.x, next.y - curr.y};

        float cross = to_prev.x * to_prev.y - to_prev.y * to_next.x;

        // Find the actual apex by looking for maximum curvature in this segment
        int start_idx = i * env->bezier_resolution;
        int end_idx = ((i + 1) % env->num_points) * env->bezier_resolution;
        int apex_idx = start_idx;
        float max_curvature = 0.0f;

        for (int k = start_idx + 1; k < end_idx - 1; k++) {
            Vector2 p1 = env->track.centerline[(k - 1 + env->track.total_points) % env->track.total_points];
            Vector2 p2 = env->track.centerline[k];
            Vector2 p3 = env->track.centerline[(k + 1) % env->track.total_points];

            Vector2 v1 = {p2.x - p1.x, p2.y - p1.y};
            Vector2 v2 = {p3.x - p2.x, p3.y - p2.y};
            float curvature = fabsf(v1.x * v2.y - v1.y * v2.x);

            if (curvature > max_curvature) {
                max_curvature = curvature;
                apex_idx = k;
            }
        }

        Vector2* edge_points = (cross > 0) ? env->track.inner_edge : env->track.outer_edge;

        for (int j = 0; j < 4; j++) {
            int idx = (apex_idx - 2 + j + env->track.total_points) % env->track.total_points;
            env->track.curbs[env->track.curb_count][j] = edge_points[idx];
        }

        env->track.curb_count++;
    }
}

// =========================================================================== END BEZIER ===========================================

void GenerateRandomTrack(WhiskerRacer* env) {
    GenerateRandomControlPoints(env);
    GenerateTrackCenterline(env);
    GenerateTrackEdges(env);
    GenerateCurbs(env);
}

void c_render(WhiskerRacer* env) {

    int height = env->height;

    env->render = 1;
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

    if (env->render_many)
    {
        GenerateRandomTrack(env);
    }

    Vector2* center_points = malloc(sizeof(Vector2) * (env->track.total_points + 3));
    //center_points[0] = (Vector2){SCREEN_WIDTH*0.5f, SCREEN_HEIGHT*0.5f};
    for (int i = 0; i < env->track.total_points; i++) {
        center_points[i] = env->track.centerline[i];
        center_points[i].y = height - center_points[i].y;
    }

    // Without enough overlap it draws a C rather than an O
    center_points[env->track.total_points] = center_points[0];
    center_points[env->track.total_points + 1] = center_points[1];
    center_points[env->track.total_points + 2] = center_points[2];

    BeginDrawing();
    SetWindowSize(640, 480);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground(GREEN);
    DrawSplineBasis(center_points, env->track.total_points + 3, env->track_width, BLACK);
    for (int i = 0; i < env->track.curb_count; i++) {
        Vector2 curb_points[4];
        for (int j = 0; j < 4; j++) {
            curb_points[j] = env->track.curbs[i][j];
            curb_points[j].y = height - curb_points[j].y; // Flip Y coordinate
        }
        DrawSplineBasis(curb_points, 4, 5.0f, RED); // 5 pixel wide red curbs
    }
    free(center_points);

    float car_width = 24.0f;
    float car_height = 12.0f;
    float car_x = env->px;
    float car_y = height - env->py;
    Vector2 car_center = {car_x, car_y};
    Rectangle car_rect = {
        car_x - car_width / 2.0f,
        car_y - car_height / 2.0f,
        car_width,
        car_height
    };
    Vector2 origin = {car_width / 2.0f, car_height / 2.0f};
    DrawRectanglePro(
        (Rectangle){car_x, car_y, car_width, car_height},
        origin,
        -env->ang * 180.0f / PI,
        (Color){255, 0, 255, 255}
    );

    EndDrawing();
}

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <lammps/library.h>
#include "raylib.h"

#define WIDTH 1080
#define HEIGHT 720

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

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


// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field 
} Log;

typedef struct {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
} Client;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    float* actions;              // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    Vec3 pos;
    Vec3 vel;
    Vec3 goal;
    int tick;
    Client* client;
    void* handle;
} Matsci;

void init(Matsci* env) {
  void *handle;
  const char *lmpargv[] = { "liblammps", "-log", "none", "-screen", "none"};
  int lmpargc = sizeof(lmpargv)/sizeof(const char *);

  /* create LAMMPS instance */
  handle = lammps_open_no_mpi(lmpargc, (char **)lmpargv, NULL);
  if (handle == NULL) {
    printf("LAMMPS initialization failed\n");
    lammps_mpi_finalize();
  }

  // Setup the basic simulation parameters via string commands
  lammps_command(handle, "units lj");
  lammps_command(handle, "dimension 3");
  lammps_command(handle, "boundary p p p");
  lammps_command(handle, "atom_style atomic");
  lammps_command(handle, "pair_style zero 1.0 nocoeff");  // Dummy pair style for no interactions
  lammps_command(handle, "region box block -10 10 -10 10 -10 10");
  lammps_command(handle, "create_box 1 box");
  lammps_command(handle, "mass 1 1.0");
  lammps_command(handle, "pair_coeff * *");

  // Load a single atom at a specific position (0.0, 0.0, 0.0)
  lammps_command(handle, "create_atoms 1 single 0.0 0.0 0.0");

  // Setup for running simulations (timestep and integrator)
  lammps_command(handle, "timestep 0.05");
  lammps_command(handle, "fix 1 all nve");

  // Initialize
  lammps_command(handle, "run 0");
  env->handle = handle;
}

void compute_observations(Matsci* env) {
    env->observations[0] = env->pos.x - env->goal.x;
    env->observations[1] = env->pos.y - env->goal.y;
    env->observations[2] = env->pos.z - env->goal.z;
}

void c_reset(Matsci* env) {
    void* handle = env->handle;
    double** x = (double **) lammps_extract_atom(handle, "x");
    x[0][0] = rndf(-1.0f, 1.0f);
    x[0][1] = rndf(-1.0f, 1.0f);
    x[0][2] = rndf(-1.0f, 1.0f);
    env->goal.x = rndf(-1.0f, 1.0f);
    env->goal.y = rndf(-1.0f, 1.0f);
    env->goal.z = rndf(-1.0f, 1.0f);
    env->tick = 0;
}

void c_step(Matsci* env) {
    void* handle = env->handle;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick++;

    if (env->tick >= 128) {
        c_reset(env);
        env->rewards[0] = -1;
        env->terminals[0] = 1;
        env->log.n += 1;
	return;
    }

    double **v = (double **) lammps_extract_atom(handle, "v");
    v[0][0] = 1.0f*env->actions[0];
    v[0][1] = 1.0f*env->actions[1];
    v[0][2] = 1.0f*env->actions[2];

    lammps_command(handle, "run 1");

    double** x = (double **) lammps_extract_atom(handle, "x");

    // Example of moving again and rerunning: reset position and set new velocity
    x = (double **) lammps_extract_atom(handle, "x");
    env->pos.x = x[0][0];
    env->pos.y = x[0][1];
    env->pos.z = x[0][2];

    float dist = norm3(sub3(env->pos, env->goal));

    if (dist > 2.0f) {
        c_reset(env);
        env->rewards[0] = -1;
        env->terminals[0] = 1;
        env->log.n += 1;
	return;
    }

    if (dist < 0.1) {
        c_reset(env);
        env->rewards[0] = 1;
        env->terminals[0] = 1;
        env->log.score += 1;
        env->log.n += 1;
    }

    compute_observations(env);
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

void c_close(Matsci* env) {
    /*
    if (IsWindowReady()) {
        CloseWindow();
    }
    */
}

void c_render(Matsci* env) {
    if (!IsWindowReady()) {
        Client *client = (Client *)calloc(1, sizeof(Client));

        client->width = WIDTH;
        client->height = HEIGHT;

        SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
        InitWindow(WIDTH, HEIGHT, "PufferLib DroneSwarm");

        #ifndef __EMSCRIPTEN__
            SetTargetFPS(60);
        #endif

        client->camera_distance = 40.0f;
        client->camera_azimuth = 0.0f;
        client->camera_elevation = PI / 10.0f;
        client->is_dragging = false;
        client->last_mouse_pos = (Vector2){0.0f, 0.0f};

        client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;
	env->client = client;

        update_camera_position(client);
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {

        c_close(env);
        exit(0);
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, 2.0f, 2.0f, 2.0f, WHITE);

    DrawSphere((Vector3){env->pos.x, env->pos.y, env->pos.z}, 0.1f, PUFF_CYAN);
    DrawSphere((Vector3){env->goal.x, env->goal.y, env->goal.z}, 0.1f, PUFF_RED);
    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);

    EndDrawing();
}



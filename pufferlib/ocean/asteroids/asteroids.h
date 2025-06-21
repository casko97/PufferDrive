#pragma once

#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const unsigned char FORWARD = 0;
const unsigned char TURN_LEFT = 1;
const unsigned char TURN_RIGHT = 2;

// Only use floats!
typedef struct {
  float score;
  float n; // Required as the last field
} Log;

typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  Vector2 player_position;
  float player_angle;
  int thruster_on;
} Asteroids;

Vector2 rotate_vector(Vector2 point, Vector2 center, float angle) {
  float s = sinf(angle);
  float c = cosf(angle);

  // Translate point back to origin:
  point.x -= center.x;
  point.y -= center.y;

  // Rotate point
  float xnew = point.x * c - point.y * s;
  float ynew = point.x * s + point.y * c;

  // Translate point back:
  point.x = xnew + center.x;
  point.y = ynew + center.y;
  return point;
}

Vector2 get_direction_vector(Asteroids *env) {
  float px = env->player_position.x;
  float py = env->player_position.y;
  Vector2 dir = (Vector2){px, py - 1};
  dir = rotate_vector(dir, env->player_position, env->player_angle);
  return (Vector2){dir.x - px, dir.y - py};
}

void c_reset(Asteroids *env) {
  env->player_position = (Vector2){env->size / 2.0f, env->size / 2.0f};
  env->player_angle = 0.5f;
  env->thruster_on = 0;
}

void c_step(Asteroids *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
  env->thruster_on = 0;
  int action = env->actions[0];
  if (action == TURN_LEFT)
    env->player_angle -= 0.1f;
  if (action == TURN_RIGHT)
    env->player_angle += 0.1f;
  if (action == FORWARD) {
    Vector2 dir = get_direction_vector(env);
    env->player_position.x += dir.x * 2.0f;
    env->player_position.y += dir.y * 2.0f;
    env->thruster_on = 1;
  }
}

void draw_player(Asteroids *env) {
  float px = env->player_position.x;
  float py = env->player_position.y;

  // debug
  DrawRectangle(px, py, 1, 1, RED);
  Vector2 dir = get_direction_vector(env);
  dir = (Vector2){dir.x * 10.0f, dir.y * 10.f};
  Vector2 t = (Vector2){dir.x + px, dir.y + py};
  DrawLineV(env->player_position, t, RED);

  Vector2 ps[8];

  // ship
  ps[0] = (Vector2){px - 10, py + 10};
  ps[1] = (Vector2){px + 10, py + 10};
  ps[2] = (Vector2){px, py - 20};
  ps[3] = (Vector2){px - 9, py + 6};
  ps[4] = (Vector2){px + 9, py + 6};
  ps[5] = (Vector2){px - 5, py + 6};
  ps[6] = (Vector2){px + 5, py + 6};
  ps[7] = (Vector2){px, py + 14};

  for (int i = 0; i < 8; i++)
    ps[i] = rotate_vector(ps[i], env->player_position, env->player_angle);

  DrawLineV(ps[0], ps[2], RAYWHITE);
  DrawLineV(ps[1], ps[2], RAYWHITE);

  DrawLineV(ps[3], ps[4], RAYWHITE);

  if (env->thruster_on) {
    DrawLineV(ps[5], ps[7], RAYWHITE);
    DrawLineV(ps[6], ps[7], RAYWHITE);
  }
}

void c_render(Asteroids *env) {
  if (!IsWindowReady()) {
    InitWindow(env->size, env->size, "PufferLib Asteroids");
    SetTargetFPS(60);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(BLACK);
  draw_player(env);
  EndDrawing();
}

void c_close(Asteroids *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}

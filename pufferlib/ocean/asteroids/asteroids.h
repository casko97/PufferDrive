#pragma once

#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
} Asteroids;

void c_reset(Asteroids *env) {
  env->player_position = (Vector2){env->size / 2.0f, env->size / 2.0f};
  env->player_angle = 0.0f;
}

void c_step(Asteroids *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
}

Vector2 RotatePoint(Vector2 point, Vector2 center, float angle) {
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

void draw_player(Asteroids *env) {
  DrawRectangle(env->player_position.x, env->player_position.y, 1, 1, RED);
  Vector2 ps[8];

  // ship
  ps[0] = (Vector2){env->player_position.x - 10, env->player_position.y + 10};
  ps[1] = (Vector2){env->player_position.x + 10, env->player_position.y + 10};
  ps[2] = (Vector2){env->player_position.x, env->player_position.y - 20};
  DrawLineV(ps[0], ps[2], RAYWHITE);
  DrawLineV(ps[1], ps[2], RAYWHITE);

  ps[3] = (Vector2){env->player_position.x - 9, env->player_position.y + 6};
  ps[4] = (Vector2){env->player_position.x + 9, env->player_position.y + 6};
  DrawLineV(ps[3], ps[4], RAYWHITE);

  ps[5] = (Vector2){env->player_position.x - 5, env->player_position.y + 6};
  ps[6] = (Vector2){env->player_position.x + 5, env->player_position.y + 6};
  ps[7] = (Vector2){env->player_position.x, env->player_position.y + 14};
  DrawLineV(ps[5], ps[7], RAYWHITE);
  DrawLineV(ps[6], ps[7], RAYWHITE);
}

void c_render(Asteroids *env) {
  if (!IsWindowReady()) {
    InitWindow(env->size, env->size, "PufferLib Asteroids");
    SetTargetFPS(5);
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

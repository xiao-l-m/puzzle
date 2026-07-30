#ifndef GANTRY_CONTROLLER_H
#define GANTRY_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

void GantryController_Init(uint32_t nowMs);
void GantryController_Update(uint32_t nowMs);

/* mode 1 = move pieces across divider; mode 2 = solve the puzzle. */
bool GantryController_RequestMode(uint8_t mode, uint32_t nowMs);

/* The buffer is tokenized in place. */
void GantryController_ProcessLine(char *line, uint32_t nowMs);

/* Any emergency stop invalidates the open-loop position estimate. */
void GantryController_EmergencyStop(const char *reason);

bool GantryController_IsIdle(void);

#endif

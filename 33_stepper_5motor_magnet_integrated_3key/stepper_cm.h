#ifndef STEPPER_CM_H
#define STEPPER_CM_H

#include <stdbool.h>
#include <stdint.h>

#define STEPPER_CM_MOTOR_COUNT 5U

typedef enum {
    STEPPER_MOTOR_1 = 1,
    STEPPER_MOTOR_2 = 2,
    STEPPER_MOTOR_3 = 3,
    STEPPER_MOTOR_4 = 4,
    STEPPER_MOTOR_5 = 5
} StepperMotorId;

typedef struct {
    StepperMotorId motor;
    float centimeters; /* Positive = forward, negative = reverse. */
} StepperCmMove;

typedef struct {
    StepperMotorId motor;
    float centimeters;
    uint16_t speedRpm;
    uint8_t acceleration;
} StepperCmProfiledMove;

extern volatile uint8_t g_stepperCmLastFrame[13];
extern volatile uint8_t g_stepperCmLastFrameLength;
extern volatile uint32_t g_stepperCmFrameCount;
extern volatile uint32_t g_stepperCmErrorCount;

/* Call once after SYSCFG_DL_init(). */
void StepperCm_Init(void);

bool StepperCm_Enable(StepperMotorId motor, bool enable);
bool StepperCm_Stop(StepperMotorId motor);
bool StepperCm_StopAll(void);

/* Uses the default speed and acceleration from stepper_cm_config.h. */
bool StepperCm_Move(StepperMotorId motor, float centimeters);

bool StepperCm_MoveWithParams(StepperMotorId motor, float centimeters,
                              uint16_t speedRpm, uint8_t acceleration);

/* All queued motors start together after one broadcast synchronization frame. */
bool StepperCm_MoveSynchronized(const StepperCmMove *moves, uint8_t count);
bool StepperCm_MoveSynchronizedWithParams(const StepperCmMove *moves,
                                          uint8_t count,
                                          uint16_t speedRpm,
                                          uint8_t acceleration);

/* Starts all queued motors together while allowing a different speed for
 * each axis. This is useful for straight XY moves with unequal distances.
 */
bool StepperCm_MoveProfiledSynchronized(
    const StepperCmProfiledMove *moves, uint8_t count);

/* Convenient per-motor wrappers requested by the application. */
bool Motor1_MoveCm(float centimeters);
bool Motor2_MoveCm(float centimeters);
bool Motor3_MoveCm(float centimeters);
bool Motor4_MoveCm(float centimeters);
bool Motor5_MoveCm(float centimeters);

/* Motor 5 rotary-axis API. Positive/negative select opposite directions. */
bool Motor5_RotateDeg(float degrees);
bool Motor5_RotateDegWithParams(float degrees,
                                uint16_t speedRpm,
                                uint8_t acceleration);

#endif

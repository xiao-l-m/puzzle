#include "stepper_cm.h"

#include "stepper_cm_config.h"
#include "ti_msp_dl_config.h"

#include <stddef.h>

#define CPU_CLOCK_HZ             32000000U
#define UART_TX_TIMEOUT          100000U
#define EMM_FRAME_END            0x6BU
#define EMM_SYNC_FLAG            0x01U
#define EMM_RELATIVE_MODE        0x00U

static const uint8_t g_motorAddresses[STEPPER_CM_MOTOR_COUNT] = {
    STEPPER_MOTOR_1_ADDRESS,
    STEPPER_MOTOR_2_ADDRESS,
    STEPPER_MOTOR_3_ADDRESS,
    STEPPER_MOTOR_4_ADDRESS,
    STEPPER_MOTOR_5_ADDRESS
};

static const uint8_t g_directionXor[STEPPER_CM_MOTOR_COUNT] = {
    STEPPER_MOTOR_1_DIRECTION_XOR,
    STEPPER_MOTOR_2_DIRECTION_XOR,
    STEPPER_MOTOR_3_DIRECTION_XOR,
    STEPPER_MOTOR_4_DIRECTION_XOR,
    STEPPER_MOTOR_5_DIRECTION_XOR
};

static const float g_distanceScale[STEPPER_CM_MOTOR_COUNT] = {
    STEPPER_MOTOR_1_DISTANCE_SCALE,
    STEPPER_MOTOR_2_DISTANCE_SCALE,
    STEPPER_MOTOR_3_DISTANCE_SCALE,
    STEPPER_MOTOR_4_DISTANCE_SCALE,
    STEPPER_MOTOR_5_DISTANCE_SCALE
};

volatile uint8_t g_stepperCmLastFrame[13];
volatile uint8_t g_stepperCmLastFrameLength;
volatile uint32_t g_stepperCmFrameCount;
volatile uint32_t g_stepperCmErrorCount;

static void delay_ms(uint32_t milliseconds)
{
    while (milliseconds-- > 0U) {
        delay_cycles(CPU_CLOCK_HZ / 1000U);
    }
}

static bool motor_is_valid(StepperMotorId motor)
{
    return ((uint8_t)motor >= (uint8_t)STEPPER_MOTOR_1) &&
           ((uint8_t)motor <= (uint8_t)STEPPER_MOTOR_5);
}

static uint8_t motor_index(StepperMotorId motor)
{
    return (uint8_t)((uint8_t)motor - 1U);
}

static bool send_byte(uint8_t data)
{
    uint32_t timeout = UART_TX_TIMEOUT;

    while (DL_UART_isBusy(UART_BUS_INST)) {
        if (timeout-- == 0U) {
            return false;
        }
    }

    DL_UART_Main_transmitData(UART_BUS_INST, data);
    return true;
}

static bool send_frame(const uint8_t *data, uint8_t length)
{
    uint8_t i;
    uint32_t timeout = UART_TX_TIMEOUT;

    if ((data == NULL) || (length == 0U) ||
        (length > (uint8_t)sizeof(g_stepperCmLastFrame))) {
        g_stepperCmErrorCount++;
        return false;
    }

    g_stepperCmLastFrameLength = length;
    for (i = 0U; i < length; i++) {
        g_stepperCmLastFrame[i] = data[i];
        if (!send_byte(data[i])) {
            g_stepperCmErrorCount++;
            return false;
        }
    }

    while (DL_UART_isBusy(UART_BUS_INST)) {
        if (timeout-- == 0U) {
            g_stepperCmErrorCount++;
            return false;
        }
    }

    g_stepperCmFrameCount++;
    return true;
}

static bool trigger_synchronous(void)
{
    const uint8_t command[4] = {
        0x00U, 0xFFU, 0x66U, EMM_FRAME_END
    };

    return send_frame(command, (uint8_t)sizeof(command));
}

static bool pulses_to_command(StepperMotorId motor,
                              bool reverse,
                              uint32_t pulses,
                              uint16_t speedRpm,
                              uint8_t acceleration,
                              uint8_t command[13])
{
    uint8_t index;
    uint8_t direction;

    if (!motor_is_valid(motor) || (pulses == 0U) ||
        (speedRpm == 0U) || (speedRpm > 5000U) ||
        (command == NULL)) {
        return false;
    }

    index = motor_index(motor);
    direction = reverse ? 1U : 0U;
    direction ^= g_directionXor[index];

    command[0]  = g_motorAddresses[index];
    command[1]  = 0xFDU;
    command[2]  = direction;
    command[3]  = (uint8_t)(speedRpm >> 8);
    command[4]  = (uint8_t)speedRpm;
    command[5]  = acceleration;
    command[6]  = (uint8_t)(pulses >> 24);
    command[7]  = (uint8_t)(pulses >> 16);
    command[8]  = (uint8_t)(pulses >> 8);
    command[9]  = (uint8_t)pulses;
    command[10] = EMM_RELATIVE_MODE;
    command[11] = EMM_SYNC_FLAG;
    command[12] = EMM_FRAME_END;
    return true;
}

static bool centimeters_to_command(StepperMotorId motor, float centimeters,
                                   uint16_t speedRpm,
                                   uint8_t acceleration,
                                   uint8_t command[13])
{
    float magnitudeCm;
    float pulseValue;
    uint32_t pulses;
    bool reverse;

    if (!motor_is_valid(motor) || (centimeters == 0.0F) ||
        (centimeters > STEPPER_MAX_ABS_DISTANCE_CM) ||
        (centimeters < -STEPPER_MAX_ABS_DISTANCE_CM) ||
        (speedRpm == 0U) || (speedRpm > 5000U)) {
        return false;
    }

    reverse = centimeters < 0.0F;
    magnitudeCm = (centimeters < 0.0F) ? -centimeters : centimeters;

    if (g_distanceScale[motor_index(motor)] <= 0.0F) {
        return false;
    }

    pulseValue = (magnitudeCm * 10.0F * (float)STEPPER_PULSES_PER_REV *
                  g_distanceScale[motor_index(motor)]) /
                 (float)STEPPER_SLIDE_LEAD_MM;
    pulses = (uint32_t)(pulseValue + 0.5F);
    if (pulses == 0U) {
        return false;
    }

    return pulses_to_command(motor, reverse, pulses, speedRpm,
                             acceleration, command);
}

static bool degrees_to_command(float degrees,
                               uint16_t speedRpm,
                               uint8_t acceleration,
                               uint8_t command[13])
{
    float magnitudeDegrees;
    float pulseValue;
    uint32_t pulses;

    if ((degrees == 0.0F) ||
        (degrees > STEPPER_MOTOR_5_MAX_ABS_DEGREES) ||
        (degrees < -STEPPER_MOTOR_5_MAX_ABS_DEGREES)) {
        return false;
    }

    magnitudeDegrees = (degrees < 0.0F) ? -degrees : degrees;
    pulseValue = magnitudeDegrees *
                 (float)STEPPER_MOTOR_5_PULSES_PER_OUTPUT_REV / 360.0F;
    pulses = (uint32_t)(pulseValue + 0.5F);

    return pulses_to_command(STEPPER_MOTOR_5, degrees < 0.0F, pulses,
                             speedRpm, acceleration, command);
}

void StepperCm_Init(void)
{
    uint8_t i;

    g_stepperCmLastFrameLength = 0U;
    g_stepperCmFrameCount = 0U;
    g_stepperCmErrorCount = 0U;
    for (i = 0U; i < (uint8_t)sizeof(g_stepperCmLastFrame); i++) {
        g_stepperCmLastFrame[i] = 0U;
    }
}

bool StepperCm_Enable(StepperMotorId motor, bool enable)
{
    uint8_t command[6];

    if (!motor_is_valid(motor)) {
        g_stepperCmErrorCount++;
        return false;
    }

    command[0] = g_motorAddresses[motor_index(motor)];
    command[1] = 0xF3U;
    command[2] = 0xABU;
    command[3] = enable ? 0x01U : 0x00U;
    command[4] = 0x00U;
    command[5] = EMM_FRAME_END;
    return send_frame(command, (uint8_t)sizeof(command));
}

bool StepperCm_Stop(StepperMotorId motor)
{
    uint8_t command[5];

    if (!motor_is_valid(motor)) {
        g_stepperCmErrorCount++;
        return false;
    }

    command[0] = g_motorAddresses[motor_index(motor)];
    command[1] = 0xFEU;
    command[2] = 0x98U;
    command[3] = EMM_SYNC_FLAG;
    command[4] = EMM_FRAME_END;

    if (!send_frame(command, (uint8_t)sizeof(command))) {
        return false;
    }
    delay_ms(10U);
    return trigger_synchronous();
}

bool StepperCm_StopAll(void)
{
    StepperMotorId motor;
    bool ok = true;

    for (motor = STEPPER_MOTOR_1;
         motor <= STEPPER_MOTOR_5;
         motor = (StepperMotorId)((uint8_t)motor + 1U)) {
        const uint8_t command[5] = {
            g_motorAddresses[motor_index(motor)],
            0xFEU, 0x98U, EMM_SYNC_FLAG, EMM_FRAME_END
        };
        ok = send_frame(command, (uint8_t)sizeof(command)) && ok;
        delay_ms(10U);
    }
    return trigger_synchronous() && ok;
}

bool StepperCm_MoveWithParams(StepperMotorId motor, float centimeters,
                              uint16_t speedRpm, uint8_t acceleration)
{
    uint8_t command[13];

    if (!centimeters_to_command(motor, centimeters, speedRpm,
                                acceleration, command)) {
        g_stepperCmErrorCount++;
        return false;
    }

    if (!StepperCm_Enable(motor, true)) {
        return false;
    }
    delay_ms(20U);
    if (!send_frame(command, (uint8_t)sizeof(command))) {
        return false;
    }
    delay_ms(10U);
    return trigger_synchronous();
}

bool StepperCm_Move(StepperMotorId motor, float centimeters)
{
    return StepperCm_MoveWithParams(motor, centimeters,
                                    STEPPER_DEFAULT_SPEED_RPM,
                                    STEPPER_DEFAULT_ACCELERATION);
}

bool StepperCm_MoveSynchronizedWithParams(const StepperCmMove *moves,
                                          uint8_t count,
                                          uint16_t speedRpm,
                                          uint8_t acceleration)
{
    StepperCmProfiledMove profiledMoves[STEPPER_CM_MOTOR_COUNT];
    uint8_t i;

    if ((moves == NULL) || (count == 0U) ||
        (count > STEPPER_CM_MOTOR_COUNT)) {
        g_stepperCmErrorCount++;
        return false;
    }

    for (i = 0U; i < count; i++) {
        profiledMoves[i].motor = moves[i].motor;
        profiledMoves[i].centimeters = moves[i].centimeters;
        profiledMoves[i].speedRpm = speedRpm;
        profiledMoves[i].acceleration = acceleration;
    }

    return StepperCm_MoveProfiledSynchronized(profiledMoves, count);
}

bool StepperCm_MoveProfiledSynchronized(
    const StepperCmProfiledMove *moves, uint8_t count)
{
    uint8_t commands[STEPPER_CM_MOTOR_COUNT][13];
    uint8_t usedMotors = 0U;
    uint8_t i;

    if ((moves == NULL) || (count == 0U) ||
        (count > STEPPER_CM_MOTOR_COUNT)) {
        g_stepperCmErrorCount++;
        return false;
    }

    /* Validate every request before transmitting any part of the movement. */
    for (i = 0U; i < count; i++) {
        uint8_t bit;

        if (!motor_is_valid(moves[i].motor)) {
            g_stepperCmErrorCount++;
            return false;
        }
        bit = (uint8_t)(1U << motor_index(moves[i].motor));
        if ((usedMotors & bit) != 0U) {
            g_stepperCmErrorCount++;
            return false;
        }
        usedMotors |= bit;

        if (!centimeters_to_command(moves[i].motor,
                                    moves[i].centimeters,
                                    moves[i].speedRpm,
                                    moves[i].acceleration,
                                    commands[i])) {
            g_stepperCmErrorCount++;
            return false;
        }
    }

    for (i = 0U; i < count; i++) {
        if (!StepperCm_Enable(moves[i].motor, true)) {
            return false;
        }
        delay_ms(20U);
    }

    for (i = 0U; i < count; i++) {
        if (!send_frame(commands[i], (uint8_t)sizeof(commands[i]))) {
            return false;
        }
        delay_ms(10U);
    }

    return trigger_synchronous();
}

bool StepperCm_MoveSynchronized(const StepperCmMove *moves, uint8_t count)
{
    return StepperCm_MoveSynchronizedWithParams(
        moves, count,
        STEPPER_DEFAULT_SPEED_RPM,
        STEPPER_DEFAULT_ACCELERATION);
}

bool Motor1_MoveCm(float centimeters)
{
    return StepperCm_Move(STEPPER_MOTOR_1, centimeters);
}

bool Motor2_MoveCm(float centimeters)
{
    return StepperCm_Move(STEPPER_MOTOR_2, centimeters);
}

bool Motor3_MoveCm(float centimeters)
{
    return StepperCm_Move(STEPPER_MOTOR_3, centimeters);
}

bool Motor4_MoveCm(float centimeters)
{
    return StepperCm_Move(STEPPER_MOTOR_4, centimeters);
}

bool Motor5_MoveCm(float centimeters)
{
    return StepperCm_Move(STEPPER_MOTOR_5, centimeters);
}

bool Motor5_RotateDegWithParams(float degrees,
                                uint16_t speedRpm,
                                uint8_t acceleration)
{
    uint8_t command[13];

    if (!degrees_to_command(degrees, speedRpm, acceleration, command)) {
        g_stepperCmErrorCount++;
        return false;
    }

    if (!StepperCm_Enable(STEPPER_MOTOR_5, true)) {
        return false;
    }
    delay_ms(20U);
    if (!send_frame(command, (uint8_t)sizeof(command))) {
        return false;
    }
    delay_ms(10U);
    return trigger_synchronous();
}

bool Motor5_RotateDeg(float degrees)
{
    return Motor5_RotateDegWithParams(
        degrees,
        STEPPER_MOTOR_5_SPEED_RPM,
        STEPPER_MOTOR_5_ACCELERATION);
}

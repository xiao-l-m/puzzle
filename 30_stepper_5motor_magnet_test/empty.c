#include "ti_msp_dl_config.h"

#include "gantry_controller.h"
#include "mos_switch.h"
#include "rpi_link.h"
#include "stepper_cm.h"

#include <stdbool.h>
#include <stdint.h>

#define KEY_POLL_MS          10U
#define KEY_DEBOUNCE_SAMPLES 3U
#define TASK_KEY_COUNT       2U

typedef struct {
    uint8_t raw;
    uint8_t stable;
    uint8_t samples;
} KeyState;

volatile uint32_t g_milliseconds;
static volatile bool g_emergencyRequested;

static KeyState g_taskKeys[TASK_KEY_COUNT];
static const uint32_t g_taskKeyPins[TASK_KEY_COUNT] = {
    KEYS_KEY1_PIN,
    KEYS_KEY2_PIN
};
static uint32_t g_lastKeyPollMs;
static bool g_emergencyHandled;

void SysTick_Handler(void)
{
    g_milliseconds++;
}

static uint8_t key_is_pressed(uint32_t pin)
{
    return (DL_GPIO_readPins(KEYS_PORT, pin) == 0U) ? 1U : 0U;
}

static bool emergency_is_pressed(void)
{
    return DL_GPIO_readPins(KEY_PORT, KEY_PIN_21_PIN) == 0U;
}

static void task_keys_init(void)
{
    uint8_t index;

    for (index = 0U; index < TASK_KEY_COUNT; index++) {
        uint8_t pressed = key_is_pressed(g_taskKeyPins[index]);

        g_taskKeys[index].raw = pressed;
        g_taskKeys[index].stable = pressed;
        g_taskKeys[index].samples = KEY_DEBOUNCE_SAMPLES;
    }
}

static void task_keys_poll(uint32_t nowMs)
{
    uint8_t index;

    for (index = 0U; index < TASK_KEY_COUNT; index++) {
        uint8_t pressed = key_is_pressed(g_taskKeyPins[index]);
        KeyState *state = &g_taskKeys[index];

        if (pressed != state->raw) {
            state->raw = pressed;
            state->samples = 1U;
            continue;
        }
        if (state->samples < KEY_DEBOUNCE_SAMPLES) {
            state->samples++;
        }
        if ((state->samples == KEY_DEBOUNCE_SAMPLES) &&
            (state->stable != pressed)) {
            state->stable = pressed;
            if (pressed != 0U) {
                /* KEY1 starts mode 1, KEY2 starts mode 2. */
                (void)GantryController_RequestMode((uint8_t)(index + 1U),
                                                   nowMs);
            }
        }
    }
}

static void emergency_service(void)
{
    bool pressed = emergency_is_pressed();

    if ((g_emergencyRequested || pressed) && !g_emergencyHandled) {
        g_emergencyRequested = false;
        g_emergencyHandled = true;
        GantryController_EmergencyStop("PB21");
    }
    if (!pressed) {
        g_emergencyHandled = false;
        g_emergencyRequested = false;
    }
}

int main(void)
{
    char line[RPI_LINK_LINE_MAX];

    SYSCFG_DL_init();
    (void)SysTick_Config(CPUCLK_FREQ / 1000U);

    StepperCm_Init();
    MOS_Switch_Init();
    RpiLink_Init();
    task_keys_init();

    g_lastKeyPollMs = g_milliseconds;
    g_emergencyRequested = emergency_is_pressed();
    g_emergencyHandled = false;

    GantryController_Init(g_milliseconds);

    NVIC_ClearPendingIRQ(KEY_INT_IRQN);
    NVIC_EnableIRQ(KEY_INT_IRQN);

    while (1) {
        uint32_t nowMs = g_milliseconds;

        /* This service always precedes protocol parsing and state transitions. */
        emergency_service();

        if (RpiLink_TakeRxError()) {
            (void)RpiLink_SendLine("ERR 0 RX_OVERFLOW_OR_LINE_TOO_LONG");
        }
        while (RpiLink_ReadLine(line, sizeof(line))) {
            GantryController_ProcessLine(line, nowMs);
            emergency_service();
        }

        if ((nowMs - g_lastKeyPollMs) >= KEY_POLL_MS) {
            g_lastKeyPollMs = nowMs;
            task_keys_poll(nowMs);
        }

        GantryController_Update(nowMs);

        /* Solid LED means a request is active or the controller is stopped. */
        if (GantryController_IsIdle()) {
            DL_GPIO_clearPins(LED1_PORT, LED1_PIN_22_PIN);
        } else {
            DL_GPIO_setPins(LED1_PORT, LED1_PIN_22_PIN);
        }
    }
}

void GROUP1_IRQHandler(void)
{
    switch (DL_Interrupt_getPendingGroup(DL_INTERRUPT_GROUP_1)) {
        case KEY_INT_IIDX:
            if (emergency_is_pressed()) {
                /* Hardware action first: de-energize the magnet immediately.
                 * Motor stop frames are sent from the main loop.
                 */
                MOS_Switch_EmergencyOff();
                g_emergencyRequested = true;
            }
            break;
        default:
            break;
    }
}

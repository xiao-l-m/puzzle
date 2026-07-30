#include "mos_switch.h"

#include "ti_msp_dl_config.h"

static volatile bool s_isOn;
static volatile bool s_emergencyInhibit;

void MOS_Switch_Init(void)
{
    s_emergencyInhibit = false;
    MOS_Switch_Off();
}

void MOS_Switch_On(void)
{
    if (s_emergencyInhibit) {
        DL_GPIO_clearPins(MOS_CTRL_PORT, MOS_CTRL_CONTROL_PIN);
        s_isOn = false;
        return;
    }
    DL_GPIO_setPins(MOS_CTRL_PORT, MOS_CTRL_CONTROL_PIN);
    s_isOn = true;
}

void MOS_Switch_Off(void)
{
    DL_GPIO_clearPins(MOS_CTRL_PORT, MOS_CTRL_CONTROL_PIN);
    s_isOn = false;
}

void MOS_Switch_EmergencyOff(void)
{
    DL_GPIO_clearPins(MOS_CTRL_PORT, MOS_CTRL_CONTROL_PIN);
    s_isOn = false;
    s_emergencyInhibit = true;
}

void MOS_Switch_Set(bool enabled)
{
    if (enabled) {
        MOS_Switch_On();
    } else {
        MOS_Switch_Off();
    }
}

bool MOS_Switch_IsOn(void)
{
    return s_isOn;
}

bool MOS_Switch_IsInhibited(void)
{
    return s_emergencyInhibit;
}

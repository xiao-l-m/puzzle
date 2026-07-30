#include "rpi_link.h"

#include "ti_msp_dl_config.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define RPI_RX_BUFFER_SIZE 256U
#define RPI_TX_TIMEOUT     100000U

static volatile uint8_t g_rxBuffer[RPI_RX_BUFFER_SIZE];
static volatile uint16_t g_rxHead;
static volatile uint16_t g_rxTail;
static volatile bool g_rxError;

static char g_lineBuffer[RPI_LINK_LINE_MAX];
static uint16_t g_lineLength;
static bool g_discardLine;

static bool receive_byte(uint8_t *value)
{
    uint16_t tail = g_rxTail;

    if ((value == NULL) || (tail == g_rxHead)) {
        return false;
    }
    *value = g_rxBuffer[tail];
    g_rxTail = (uint16_t)((tail + 1U) % RPI_RX_BUFFER_SIZE);
    return true;
}

static bool send_byte(uint8_t value)
{
    uint32_t timeout = RPI_TX_TIMEOUT;

    while (DL_UART_isBusy(UART_PI_INST)) {
        if (timeout-- == 0U) {
            return false;
        }
    }
    DL_UART_Main_transmitData(UART_PI_INST, value);
    return true;
}

void RpiLink_Init(void)
{
    g_rxHead = 0U;
    g_rxTail = 0U;
    g_rxError = false;
    g_lineLength = 0U;
    g_discardLine = false;

    NVIC_ClearPendingIRQ(UART_PI_INST_INT_IRQN);
    NVIC_EnableIRQ(UART_PI_INST_INT_IRQN);
}

bool RpiLink_ReadLine(char *line, size_t capacity)
{
    uint8_t byte;

    if ((line == NULL) || (capacity == 0U)) {
        return false;
    }

    while (receive_byte(&byte)) {
        if (byte == (uint8_t)'\r') {
            continue;
        }
        if (byte == (uint8_t)'\n') {
            if (g_discardLine) {
                g_discardLine = false;
                g_lineLength = 0U;
                continue;
            }
            if ((size_t)g_lineLength >= capacity) {
                g_rxError = true;
                g_lineLength = 0U;
                continue;
            }
            memcpy(line, g_lineBuffer, g_lineLength);
            line[g_lineLength] = '\0';
            g_lineLength = 0U;
            return true;
        }

        if (g_discardLine) {
            continue;
        }
        if (g_lineLength < (RPI_LINK_LINE_MAX - 1U)) {
            g_lineBuffer[g_lineLength++] = (char)byte;
        } else {
            g_rxError = true;
            g_lineLength = 0U;
            g_discardLine = true;
        }
    }
    return false;
}

bool RpiLink_TakeRxError(void)
{
    bool hadError = g_rxError;

    g_rxError = false;
    if (hadError) {
        g_lineLength = 0U;
        g_discardLine = true;
    }
    return hadError;
}

bool RpiLink_SendLine(const char *line)
{
    uint32_t timeout = RPI_TX_TIMEOUT;

    if (line == NULL) {
        return false;
    }
    while (*line != '\0') {
        if (!send_byte((uint8_t)*line++)) {
            return false;
        }
    }
    if (!send_byte((uint8_t)'\r') || !send_byte((uint8_t)'\n')) {
        return false;
    }
    while (DL_UART_isBusy(UART_PI_INST)) {
        if (timeout-- == 0U) {
            return false;
        }
    }
    return true;
}

bool RpiLink_SendFormat(const char *format, ...)
{
    char buffer[192];
    int length;
    va_list arguments;

    if (format == NULL) {
        return false;
    }
    va_start(arguments, format);
    length = vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    if ((length < 0) || ((size_t)length >= sizeof(buffer))) {
        return false;
    }
    return RpiLink_SendLine(buffer);
}

void UART_PI_INST_IRQHandler(void)
{
    switch (DL_UART_getPendingInterrupt(UART_PI_INST)) {
        case DL_UART_IIDX_RX:
        {
            uint8_t byte = DL_UART_Main_receiveData(UART_PI_INST);
            uint16_t next = (uint16_t)((g_rxHead + 1U) % RPI_RX_BUFFER_SIZE);

            if (next == g_rxTail) {
                g_rxError = true;
            } else {
                g_rxBuffer[g_rxHead] = byte;
                g_rxHead = next;
            }
            break;
        }
        default:
            break;
    }
}

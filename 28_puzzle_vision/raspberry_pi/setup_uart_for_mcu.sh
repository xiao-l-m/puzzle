#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用: sudo sh setup_uart_for_mcu.sh" >&2
    exit 2
fi

cmdline=/boot/firmware/cmdline.txt
backup=/boot/firmware/cmdline.txt.codex-uart-backup
if [ ! -f "$cmdline" ]; then
    echo "找不到 $cmdline" >&2
    exit 3
fi

cp -n "$cmdline" "$backup"

uart_path=$(readlink -f /dev/serial0 2>/dev/null || true)
uart_name=$(basename "${uart_path:-ttyAMA10}")

tmp_file=$(mktemp /tmp/cmdline-uart.XXXXXX)
trap 'rm -f "$tmp_file"' EXIT INT TERM

# GPIO14/15 must be an application UART, not a Linux boot/login console.
sed -E \
    -e "s/(^|[[:space:]])console=(serial0|${uart_name}),[^[:space:]]+([[:space:]]|$)/ /g" \
    -e 's/[[:space:]]+/ /g' \
    -e 's/^ //; s/ $//' \
    "$cmdline" > "$tmp_file"
install -m 0644 "$tmp_file" "$cmdline"

systemctl disable --now "serial-getty@${uart_name}.service" >/dev/null 2>&1 || true
systemctl disable --now serial-getty@serial0.service >/dev/null 2>&1 || true

echo "UART应用串口配置完成：/dev/serial0 -> ${uart_path:-unknown}"
echo "原启动参数已备份到: $backup"
echo "请稍后重启树莓派使内核console修改完全生效。"

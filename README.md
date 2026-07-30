# A4 镀锌铁片拼图系统

本仓库包含 E 题第一大问的两种工作模式，以及完整的树莓派视觉端和天猛星 MSPM0G3507 执行端代码。

- 小问 1：识别下半区 4 片灰色镀锌铁片，将其搬运至分界线上半区。
- 小问 2：识别、匹配并旋转 4 片铁片，在上半区按题图 2 排列。

## 目录

- `28_puzzle_vision/`：树莓派相机标定、A4 透视展开、铁片识别、质心/角度计算、目标排布、网页控制、串口协议和测试。
- `30_stepper_5motor_magnet_test/`：天猛星 CCS 工程，控制 X、双 Y、Z、旋转轴和电磁铁，并实现按键、急停及树莓派协议。

## 坐标和机械标定

- A4 左上角为 `(0,0)`，X 向右、Y 向下，单位 mm。
- 固定 HOME 电磁铁中心：A4 `(5,61) mm`。
- 当前可达质心范围：`X=5..205 mm`、`Y=61..286 mm`。
- 电机 1：X；电机 2、3：同步 Y；电机 4：Z；电机 5：铁片旋转。
- 每次抓取或放置，Z 轴下降 8 mm 后返回安全高度。

## 通信接线

| Raspberry Pi 5 | 天猛星 MSPM0G3507 |
|---|---|
| GPIO14 / TXD，物理 8 脚 | PB13 / UART3 RX |
| GPIO15 / RXD，物理 10 脚 | PB12 / UART3 TX |
| GND | GND |

只交叉连接 TX/RX 并共地，不连接两端的 3.3 V 或 5 V。串口参数为 115200 8N1。

天猛星 PA10 / UART0 TX 连接五台 Emm 驱动器 RX，地址为 1 至 5。驱动器 TX 当前不并联回天猛星。PA14 连接电磁铁 MOS 模块控制输入，PB21 为急停。

## 启动

树莓派首次配置串口：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sudo sh setup_uart_for_mcu.sh
sudo reboot
```

启动服务：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sh stop_puzzle_vision.sh
sh start_puzzle_vision.sh
```

网页地址：`http://<树莓派IP>:8081`

运行前必须将滑台人工放回固定 HOME，确认 Z 在安全高度、旋转轴为 0°、电磁铁关闭，再复位天猛星。等待网页显示 A4 方向已锁定、4 片稳定、方案可执行且 MCU 串口已连接后，才能启动小问 1 或小问 2。

## 运动时序

每片严格执行：

1. 安全高度移动至拾取质心；
2. Z 下降 8 mm；
3. 开启电磁铁并等待；
4. Z 返回安全高度；
5. 按规划角度旋转；
6. XY 移动至目标质心；
7. Z 下降、关闭电磁铁；
8. Z 返回安全高度，旋转轴回零。

XY 使用 600 RPM 上限。固件根据 Emm V5 加速度档位自动计算三角形或梯形速度曲线并增加停稳余量，等待结束后才允许 Z 动作。由于驱动器没有回传到位信号，运行时仍必须有人看护并准备急停。

## 构建和测试

- 在 Code Composer Studio 中导入并构建 `30_stepper_5motor_magnet_test`。
- 树莓派测试：

```bash
cd 28_puzzle_vision/raspberry_pi
python3 -m unittest test_puzzle_motion.py
python3 puzzle_vision.py --self-test
```

更详细的标定、协议和安全说明见两个子工程中的 README 与 `RASPBERRY_PI_PROTOCOL.md`。

## 第二大问第（1）小问

新增模式3：识别A4上半区1～4片白色碎片，不依赖固定模板自动重建规定尺寸的矩形，计算安全吸取点、刚体旋转后的放置点与最短抓放顺序，再搬运到下半区。详见 [28_puzzle_vision/MODE3_QUESTION2_1.md](28_puzzle_vision/MODE3_QUESTION2_1.md)。

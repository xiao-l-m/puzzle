# 模式3：第二大问第（1）小问

模式3用于把纵向 A4 上半区随机摆放的 1～4 片白色碎片，自动重建为矩形并搬运到下半区。原模式1、模式2和物理按键功能不变；模式3从网页一键启动。

## 启动与使用

1. 将电磁铁中心放在固定 HOME：A4 坐标 `(5, 61) mm`；Z 位于安全高度，5号轴为0°，电磁铁关闭。
2. 烧录 `30_stepper_5motor_magnet_test`，给电机和电磁铁驱动上电，再复位天猛星。
3. 启动树莓派：

   ```bash
   cd /home/wxm/28_puzzle_vision/raspberry_pi
   sh stop_puzzle_vision.sh
   sh start_puzzle_vision.sh
   ```

4. 浏览器打开 `http://10.29.23.60:8081/`。
5. 移除摄像头遮挡。网页显示 A4 已锁定、分界线稳定、模式3白片稳定、MCU串口已连接且模式3可执行后，点击“模式3：第二大问（1）白片自动拼矩形”。不需要二次确认。

模式3只识别 `Y < 分界线-8 mm` 的白片，只把目标放在 `Y > 分界线+8 mm`。当前机械质心可达范围是 `X=5..205 mm, Y=61..286 mm`；吸取点落在 `Y<61 mm` 时会整套拒绝，不会部分执行。

## 求解结果

网页和 `/api/status` 会显示：

- 1～4片源轮廓、面积、边长、角度、吸取点与吸取方式；
- 自动重建矩形、目标轮廓、放置点及旋转角；
- 片间最小间隙、最大对应顶点距离、解的置信度裕量；
- 每个吸取/放置点的可达性及预计总时间。

默认机械安全缝约4 mm，最大对应顶点距离限制15 mm。算法只允许旋转和平移，绝不镜像或翻面。多解接近、轮廓质量差、相交、目标尺寸不合法、无外边界边、点不可达或预计时间超过110秒时均保持不动并显示原因。

## 串口协议

模式3使用可变片数协议：

```text
PLAN request_id 3 item_count
ITEM request_id index pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
COMMIT request_id
```

`item_count` 必须为1～4。固件拒绝0片、5片、重复ITEM、缺失ITEM、越界坐标及非法角度。第一片开始移动后，MCU只执行已冻结计划，后续画面变化不会修改任务。

## 指示与记录

- 执行中：LED常亮；
- 完成：LED慢闪5秒；
- 视觉/计划拒绝：LED中速闪4秒；
- 运动故障：LED快速闪；
- PB21/网页急停：LED超快速闪，电磁铁立即关闭且位置作废。

每次启动会在 `raspberry_pi/task_logs/` 保存冻结的识别轮廓、拼图方案、动作计划、预计时间、实际完成时间或错误原因，可直接用于测试记录和设计报告。

## 本地回归测试

```bash
cd 28_puzzle_vision/raspberry_pi
python3 -m unittest \
  test_arbitrary_puzzle_solver.py \
  test_motion_plan_stability.py \
  test_puzzle_motion.py \
  test_gantry_serial.py
```

测试覆盖1、2、3、4片随机旋转重建、非重叠、目标范围、偏心吸取映射、不可达拒绝、模式3可变片数动作计划和串口报文。

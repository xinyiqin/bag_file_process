# CVAT 使用指南（track 与框修正）

本文说明如何用 **CVAT** 在视频上查看、修正预标注的 **track 与框**；类别采用单标签 **fish**（第 8 列恒为 1），**摄食/普通在导出后用脚本或模型按帧再算**。

---

## 一、获取 CVAT

### 方式 1：官方在线（适合个人/小团队）

- 打开 **https://app.cvat.ai**（或 https://cvat.ai 再点 Sign in / 注册）。
- 注册账号并登录。免费版通常有任务数/人数限制，用于单段视频标注足够。

### 方式 2：本地/服务器自建（Docker）

- 参考官方：https://opencv.github.io/cvat/docs/administration/basics/installation/
- 安装后通过浏览器访问 `http://localhost:8080`（或你的服务器地址）。

---

## 二、创建项目与任务（先建标签）

1. 登录后进入 **Projects**，点击 **Create new project**。
2. **Name**：例如 `fish_feeding`。
3. **Labels**（标签）：建**一个 rectangle 标签**，名称为 **fish**（单类别；MOT 第 8 列恒为 1，摄食/普通导出后再算）。
4. 保存项目后，点击 **Create new task**（或在该项目下 **Add new task**）。
5. **Task name**：例如 `fish_video`。
6. **Select files**：上传与预标注对应的**同一段视频**（如 `fish_video.mp4`），等待上传完成。

---

## 三、导入预标注（MOT 格式）

1. 进入该 **Task**，打开任务详情页。
2. 点击 **Import annotations**（或 **Upload annotations**）。
3. **Format** 选择 **MOT 1.1**（或 **MOT**，以界面为准）。
4. 选择文件：
   - 推荐使用 **zip**：`python3 export_csv_to_mot.py ... --zip output/mot_fish_video.zip`，解压后为 **gt/gt.txt**、**gt/labels.txt**（labels.txt 仅一行 `fish`）。
5. 上传成功后，时间轴上会出现**轨迹**，每帧上会有框和 ID；播放视频即可看到预标注叠在画面上。

> 若导入失败：确认 zip 解压后仅含 **gt/** 文件夹，gt.txt 为 9 列、第 8 列恒为 1，labels.txt 为 `fish`。

---

## 四、在视频上修正标注

### 4.1 基本操作

- **时间轴**：拖动或点击可跳帧；空格可播放/暂停。
- **左侧/右侧**：通常为**对象列表**（按轨迹 ID 列出）和**属性/标签**。
- **选中框**：点击画面上的框可选中该目标，左侧会显示该轨迹的 **Track ID**、**Label**（fish）等。

### 4.2 修改轨迹 ID（合并/统一同一只鱼）

- **目标**：把「本是同一只鱼但被标成两个 ID」的轨迹合并，或把误标的 ID 改成正确 ID。
- CVAT 中通常**不能直接改 Track ID 数字**，而是通过：
  - **Merge tracks**：选中两条轨迹（在对象列表中多选），右键或菜单里选 **Merge**，合并为一条（保留一个 ID）。
  - 或**删除错误轨迹 + 在正确轨迹上补画**：删除错误 ID 的轨迹，在正确 ID 的轨迹上，到缺失的帧补画框并归到该轨迹。
- **删除轨迹**：在对象列表选中该轨迹，删除；或选中画面上的框后删除。

### 4.3 修改框位置

- 选中框后，拖动四角或边可调整大小，拖动整体可移动。用于修正明显错位的框。

### 4.4 跨帧应用（Propagate）

- 若某一帧改了框的位置或大小，希望**后续帧沿用**：可用 **Propagate**（传播）到后续 N 帧，减少逐帧修改。具体在菜单 **Object** 或右键中查找 **Propagate** / **Propagation**。

---

## 五、导出为 MOT（供转 GT CSV）

1. 在任务页点击 **Export annotations**（或 **Export dataset**）。
2. **Format** 选择 **MOT 1.1**（或与导入时一致的 MOT 格式）。
3. 下载得到的 zip，解压到本地，例如得到：
   ```
   task_fish_video-YYYYMMDD-HHMMSS/
     fish_video/   （或类似名称）
       gt.txt
   ```
4. 在本项目里执行，转成 GT CSV：
   ```bash
   python3 mot_to_gt_csv.py task_fish_video-xxx/fish_video/gt.txt --out output/gt_fish_video.csv
   ```
   得到的 CSV 中 **class 列为 1**（占位）；**摄食/普通需在导出后由脚本或模型按帧再算**（例如对每帧框跑一次分类），再写入 class 0/1 供 `evaluate_tracking.py` 使用。

---

## 六、与本项目流程的对应关系

| 你在 CVAT 里做的 | 对应到 GT CSV / 评估 |
|------------------|----------------------|
| Merge 两条轨迹 或 删除错误轨迹并补画 | 修正 **track_id**（同一只鱼同一 ID） |
| 调整框位置 | 修正 **x_min, y_min, x_max, y_max** |
| 导出 MOT 1.1 | 用 **mot_to_gt_csv.py** 转成 GT CSV（class=1）；摄食/普通由后续脚本或模型按帧再算 |

---

## 七、常见问题

- **导入报错**：确认 zip 解压后仅含 **gt/**（gt.txt、labels.txt），labels.txt 为一行 `fish`，gt.txt 第 8 列恒为 1。
- **关键帧筛选**：导出时可用 `--keyframe-every N`（如 10）使每 N 帧 keyframe=1.0，在 CVAT 中可只显示关键帧进行标注。
- **摄食/普通**：CVAT 仅做 track 与框修正；导出后用脚本或模型按帧对框做分类，再写 class 0/1 供评估。
- **视频太长卡顿**：可先用短视频（如 1 分钟）试通流程，再上整段。

更多操作细节见 CVAT 官方文档：https://opencv.github.io/cvat/docs/manual/basics/

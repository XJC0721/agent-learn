# Pomodoro Desktop

一个基于 Electron + Vue 3 的桌面番茄钟应用，支持任务记录、声音提醒和系统通知。

## 技术栈

| 技术 | 用途 |
|------|------|
| [Electron](https://www.electronjs.org/) | 桌面应用框架，提供窗口管理、系统通知、文件系统访问 |
| [Vue 3](https://vuejs.org/) | 前端框架，使用 Composition API 管理组件状态 |
| [electron-vite](https://electron-vite.org/) | 构建工具，整合 Vite 热重载与 Electron 主/渲染进程 |
| Web Audio API | 浏览器原生音频合成，无需外部音频文件生成提示音 |
| Electron IPC | 主进程与渲染进程通信，安全桥接系统级功能 |

## 功能

- **番茄计时**：25 分钟专注 → 5 分钟短休息，每 4 轮后进入 15 分钟长休息
- **任务记录**：专注开始前输入任务名称，完成后自动保存至本地 JSON 文件
- **声音提醒**：阶段结束时通过 Web Audio API 合成蜂鸣提示音（专注结束三短音，休息结束一长音）
- **系统通知**：调用 Electron Notification API 弹出桌面通知，关闭窗口时也能收到提醒
- **历史记录**：查看所有已完成的专注记录，支持一键清空
- **无边框窗口**：自定义标题栏，支持拖动，三阶段（专注/短休息/长休息）对应不同主题色

## 快速开始

```bash
# 安装依赖
npm install

# 开发模式（热重载）
npm run dev

# 构建生产版本
npm run build
```

## 项目结构

```
src/
├── main/           # Electron 主进程（窗口、IPC、文件读写、系统通知）
├── preload/        # contextBridge 安全桥接层
└── renderer/       # Vue 3 渲染进程
    └── src/
        ├── composables/    # useTimer（计时逻辑）、useSound（音频合成）
        └── components/     # TimerDisplay、ControlBar、HistoryPanel 等
```

## 数据存储

任务记录保存在系统用户数据目录：

```
Windows: C:\Users\<用户名>\AppData\Roaming\pomodoro-desktop\pomodoro-records.json
```

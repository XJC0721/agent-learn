<script setup>
import { ref, computed, onMounted } from 'vue'
import { useTimer } from './composables/useTimer'
import { useSound } from './composables/useSound'
import TimerDisplay from './components/TimerDisplay.vue'
import ControlBar from './components/ControlBar.vue'
import SessionBadges from './components/SessionBadges.vue'
import TaskInput from './components/TaskInput.vue'
import HistoryPanel from './components/HistoryPanel.vue'

const { beep } = useSound()
const taskName = ref('')
const showHistory = ref(false)
const records = ref([])

async function handlePhaseComplete(finishedPhase, nextPhase, rounds) {
  beep(finishedPhase)

  const messages = {
    focus:      { title: '专注结束！', body: `第 ${rounds} 轮完成，休息一下 ☕` },
    shortBreak: { title: '短休息结束', body: '继续加油！' },
    longBreak:  { title: '长休息结束', body: '准备好新的一轮了！' }
  }
  const msg = messages[finishedPhase]
  await window.electronAPI.sendNotification(msg.title, msg.body)

  if (finishedPhase === 'focus') {
    const record = {
      task: taskName.value.trim() || '未命名任务',
      round: rounds,
      completedAt: new Date().toISOString()
    }
    await window.electronAPI.saveRecord(record)
    records.value.unshift(record)
  }
}

const timer = useTimer(handlePhaseComplete)

const phaseClass = computed(() => `phase-${timer.phase.value}`)

onMounted(async () => {
  const loaded = await window.electronAPI.loadRecords()
  records.value = (loaded || []).reverse()
})

async function clearHistory() {
  await window.electronAPI.clearRecords()
  records.value = []
}
</script>

<template>
  <div class="app" :class="phaseClass">
    <!-- 自定义标题栏 -->
    <div class="titlebar">
      <span class="app-name">Pomodoro</span>
      <div class="win-controls">
        <button @click="window.electronAPI.minimizeWindow()">
          <svg width="12" height="12" viewBox="0 0 12 12"><rect y="5.5" width="12" height="1" fill="currentColor"/></svg>
        </button>
        <button class="close-btn" @click="window.electronAPI.closeWindow()">
          <svg width="12" height="12" viewBox="0 0 12 12"><path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="1.5"/></svg>
        </button>
      </div>
    </div>

    <div class="content">
      <SessionBadges :completed="timer.completedRounds.value" :total="4" />

      <TimerDisplay
        :display-time="timer.displayTime.value"
        :progress="timer.progress.value"
        :phase-config="timer.phaseConfig.value"
      />

      <ControlBar
        :is-running="timer.isRunning.value"
        @start="timer.start"
        @pause="timer.pause"
        @reset="timer.reset"
        @skip="timer.skipPhase"
      />

      <TaskInput v-model="taskName" />

      <button class="history-toggle" @click="showHistory = !showHistory">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        {{ showHistory ? '隐藏记录' : '查看记录' }}
        <span class="badge-count">{{ records.length }}</span>
      </button>

      <HistoryPanel
        v-if="showHistory"
        :records="records"
        @clear="clearHistory"
      />
    </div>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(160deg, #1a1a2e 0%, #16213e 100%);
  transition: --phase-accent 0.4s ease;
}

.phase-focus      { --phase-accent: #e94560; }
.phase-shortBreak { --phase-accent: #0f9b8e; }
.phase-longBreak  { --phase-accent: #7b52ab; }

.titlebar {
  -webkit-app-region: drag;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 12px;
  height: 38px;
  background: rgba(0, 0, 0, 0.25);
  flex-shrink: 0;
}

.app-name {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.35);
}

.win-controls {
  -webkit-app-region: no-drag;
  display: flex;
  gap: 4px;
}

.win-controls button {
  width: 28px;
  height: 22px;
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.35);
  cursor: pointer;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s, color 0.15s;
}

.win-controls button:hover { background: rgba(255, 255, 255, 0.1); color: white; }
.win-controls .close-btn:hover { background: #e94560; color: white; }

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4px 0 12px;
  overflow-y: auto;
}

.history-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.45);
  font-size: 12px;
  padding: 6px 14px;
  border-radius: 20px;
  cursor: pointer;
  margin: 8px 0 4px;
  transition: border-color 0.2s, color 0.2s;
}

.history-toggle:hover {
  border-color: var(--phase-accent);
  color: var(--phase-accent);
}

.badge-count {
  background: rgba(255, 255, 255, 0.1);
  padding: 1px 6px;
  border-radius: 10px;
  font-size: 11px;
}
</style>

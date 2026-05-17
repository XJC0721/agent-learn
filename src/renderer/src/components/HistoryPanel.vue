<script setup>
defineProps({ records: { type: Array, default: () => [] } })
defineEmits(['clear'])

function formatDate(iso) {
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<template>
  <div class="history-panel">
    <div class="history-header">
      <span>历史记录</span>
      <button class="clear-btn" @click="$emit('clear')">清空</button>
    </div>
    <div v-if="records.length === 0" class="empty-tip">暂无记录</div>
    <ul v-else class="record-list">
      <li v-for="(r, i) in records" :key="i" class="record-item">
        <div class="record-task">{{ r.task || '未命名任务' }}</div>
        <div class="record-meta">第 {{ r.round }} 轮 · {{ formatDate(r.completedAt) }}</div>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.history-panel {
  margin: 8px 16px 16px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  overflow: hidden;
  max-height: 200px;
  display: flex;
  flex-direction: column;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px 8px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.45);
  letter-spacing: 1px;
  text-transform: uppercase;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  flex-shrink: 0;
}

.clear-btn {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.3);
  cursor: pointer;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 4px;
  transition: color 0.2s ease, background 0.2s ease;
}

.clear-btn:hover {
  color: #e94560;
  background: rgba(233, 69, 96, 0.1);
}

.empty-tip {
  padding: 20px;
  text-align: center;
  color: rgba(255, 255, 255, 0.2);
  font-size: 13px;
}

.record-list {
  list-style: none;
  overflow-y: auto;
  flex: 1;
}

.record-list::-webkit-scrollbar { width: 4px; }
.record-list::-webkit-scrollbar-track { background: transparent; }
.record-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

.record-item {
  padding: 8px 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.record-task {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.8);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.record-meta {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.3);
  flex-shrink: 0;
}
</style>

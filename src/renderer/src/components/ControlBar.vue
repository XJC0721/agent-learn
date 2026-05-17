<script setup>
defineProps({
  isRunning: { type: Boolean, required: true }
})
defineEmits(['start', 'pause', 'reset', 'skip'])
</script>

<template>
  <div class="control-bar">
    <button class="btn-icon" title="重置" @click="$emit('reset')">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
        <path d="M3 3v5h5"/>
      </svg>
    </button>

    <button class="btn-main" @click="isRunning ? $emit('pause') : $emit('start')">
      <template v-if="isRunning">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"/>
          <rect x="14" y="4" width="4" height="16"/>
        </svg>
      </template>
      <template v-else>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5,3 19,12 5,21"/>
        </svg>
      </template>
    </button>

    <button class="btn-icon" title="跳过" @click="$emit('skip')">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5,4 15,12 5,20"/>
        <line x1="19" y1="5" x2="19" y2="19"/>
      </svg>
    </button>
  </div>
</template>

<style scoped>
.control-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  margin: 12px 0;
}

.btn-main {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: var(--phase-accent, #e94560);
  border: none;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.15s ease, opacity 0.15s ease;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.btn-main:hover { transform: scale(1.06); }
.btn-main:active { transform: scale(0.96); }

.btn-icon {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: rgba(255, 255, 255, 0.6);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s ease, color 0.15s ease;
}

.btn-icon:hover {
  background: rgba(255, 255, 255, 0.14);
  color: white;
}
</style>

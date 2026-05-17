<script setup>
defineProps({
  displayTime: { type: String, required: true },
  progress: { type: Number, required: true },
  phaseConfig: { type: Object, required: true }
})

const R = 110
const CIRCUMFERENCE = 2 * Math.PI * R
</script>

<template>
  <div class="timer-display">
    <svg width="280" height="280" viewBox="0 0 280 280">
      <circle
        cx="140" cy="140" :r="R"
        fill="none"
        stroke="rgba(255,255,255,0.07)"
        stroke-width="10"
      />
      <circle
        cx="140" cy="140" :r="R"
        fill="none"
        :stroke="phaseConfig.color"
        stroke-width="10"
        stroke-linecap="round"
        :stroke-dasharray="CIRCUMFERENCE"
        :stroke-dashoffset="CIRCUMFERENCE * (1 - progress)"
        transform="rotate(-90 140 140)"
        class="progress-arc"
      />
      <text x="140" y="128" text-anchor="middle" class="time-text" :fill="phaseConfig.color">
        {{ displayTime }}
      </text>
      <text x="140" y="162" text-anchor="middle" class="phase-label" fill="rgba(255,255,255,0.45)">
        {{ phaseConfig.label }}
      </text>
    </svg>
  </div>
</template>

<style scoped>
.timer-display {
  display: flex;
  justify-content: center;
  margin: 8px 0;
}

.progress-arc {
  transition: stroke-dashoffset 0.9s linear, stroke 0.5s ease;
}

.time-text {
  font-size: 54px;
  font-weight: 300;
  font-family: 'Segoe UI', system-ui, monospace;
  letter-spacing: -2px;
}

.phase-label {
  font-size: 13px;
  letter-spacing: 3px;
  text-transform: uppercase;
  font-family: 'Segoe UI', system-ui, sans-serif;
}
</style>

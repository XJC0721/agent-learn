<script setup>
defineProps({
  completed: { type: Number, required: true },
  total: { type: Number, default: 4 }
})
</script>

<template>
  <div class="session-badges">
    <div
      v-for="i in total"
      :key="i"
      class="badge"
      :class="{ filled: i <= completed % total || (completed > 0 && completed % total === 0 && i <= total) }"
    />
  </div>
</template>

<style scoped>
.session-badges {
  display: flex;
  gap: 10px;
  justify-content: center;
  margin: 6px 0 2px;
}

.badge {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.15);
  border: 1.5px solid rgba(255, 255, 255, 0.25);
  transition: background 0.3s ease, border-color 0.3s ease;
}

.badge.filled {
  background: var(--phase-accent, #e94560);
  border-color: var(--phase-accent, #e94560);
  box-shadow: 0 0 6px var(--phase-accent, #e94560);
}
</style>

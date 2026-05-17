<script setup>
defineProps({ modelValue: { type: String, default: '' } })
defineEmits(['update:modelValue'])
</script>

<template>
  <div class="task-input-wrap">
    <input
      class="task-input"
      type="text"
      placeholder="当前在做什么？"
      maxlength="60"
      :value="modelValue"
      @input="$emit('update:modelValue', $event.target.value)"
    />
  </div>
</template>

<style scoped>
.task-input-wrap {
  padding: 0 32px;
  margin: 8px 0;
}

.task-input {
  width: 100%;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  color: white;
  font-size: 14px;
  padding: 10px 16px;
  outline: none;
  transition: border-color 0.2s ease, background 0.2s ease;
  font-family: 'Segoe UI', system-ui, sans-serif;
}

.task-input::placeholder { color: rgba(255, 255, 255, 0.3); }

.task-input:focus {
  border-color: var(--phase-accent, #e94560);
  background: rgba(255, 255, 255, 0.09);
}
</style>

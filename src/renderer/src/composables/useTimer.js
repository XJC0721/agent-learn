import { ref, computed, onUnmounted } from 'vue'

const PHASES = {
  focus:      { label: '专注',   duration: 25 * 60, color: '#e94560' },
  shortBreak: { label: '短休息', duration:  5 * 60, color: '#0f9b8e' },
  longBreak:  { label: '长休息', duration: 15 * 60, color: '#7b52ab' }
}
const ROUNDS_BEFORE_LONG = 4

export function useTimer(onPhaseComplete) {
  const phase = ref('focus')
  const secondsLeft = ref(PHASES.focus.duration)
  const isRunning = ref(false)
  const completedRounds = ref(0)

  let intervalId = null

  const totalSeconds = computed(() => PHASES[phase.value].duration)
  const progress = computed(() => 1 - secondsLeft.value / totalSeconds.value)
  const phaseConfig = computed(() => PHASES[phase.value])

  const displayTime = computed(() => {
    const m = Math.floor(secondsLeft.value / 60).toString().padStart(2, '0')
    const s = (secondsLeft.value % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  })

  function tick() {
    if (secondsLeft.value > 0) {
      secondsLeft.value--
    } else {
      handlePhaseEnd()
    }
  }

  function handlePhaseEnd() {
    stop()
    const finishedPhase = phase.value

    if (finishedPhase === 'focus') {
      completedRounds.value++
      const nextPhase = completedRounds.value % ROUNDS_BEFORE_LONG === 0
        ? 'longBreak'
        : 'shortBreak'
      onPhaseComplete?.(finishedPhase, nextPhase, completedRounds.value)
      switchPhase(nextPhase)
    } else {
      onPhaseComplete?.(finishedPhase, 'focus', completedRounds.value)
      switchPhase('focus')
    }
  }

  function switchPhase(newPhase) {
    phase.value = newPhase
    secondsLeft.value = PHASES[newPhase].duration
  }

  function start() {
    if (isRunning.value) return
    isRunning.value = true
    intervalId = setInterval(tick, 1000)
  }

  function pause() {
    isRunning.value = false
    clearInterval(intervalId)
    intervalId = null
  }

  function stop() {
    isRunning.value = false
    clearInterval(intervalId)
    intervalId = null
  }

  function reset() {
    stop()
    secondsLeft.value = PHASES[phase.value].duration
  }

  function skipPhase() {
    handlePhaseEnd()
  }

  onUnmounted(() => { clearInterval(intervalId) })

  return {
    phase,
    secondsLeft,
    isRunning,
    completedRounds,
    progress,
    displayTime,
    phaseConfig,
    start,
    pause,
    reset,
    skipPhase
  }
}

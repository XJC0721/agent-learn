export function useSound() {
  function beep(type = 'focus') {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()

    const sequences = {
      focus: [
        { freq: 880,  duration: 0.15, delay: 0 },
        { freq: 880,  duration: 0.15, delay: 0.22 },
        { freq: 1046, duration: 0.35, delay: 0.44 }
      ],
      shortBreak: [
        { freq: 660, duration: 0.5, delay: 0 }
      ],
      longBreak: [
        { freq: 523, duration: 0.3, delay: 0 },
        { freq: 659, duration: 0.6, delay: 0.35 }
      ]
    }

    const notes = sequences[type] || sequences.focus

    notes.forEach(({ freq, duration, delay }) => {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.type = 'sine'
      osc.frequency.setValueAtTime(freq, ctx.currentTime + delay)
      gain.gain.setValueAtTime(0, ctx.currentTime + delay)
      gain.gain.linearRampToValueAtTime(0.3, ctx.currentTime + delay + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + duration)
      osc.start(ctx.currentTime + delay)
      osc.stop(ctx.currentTime + delay + duration + 0.05)
    })

    const maxEnd = Math.max(...notes.map(n => n.delay + n.duration)) + 0.3
    setTimeout(() => ctx.close(), maxEnd * 1000 + 100)
  }

  return { beep }
}
